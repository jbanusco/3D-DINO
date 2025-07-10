# Author: Tony Xu
#
# This code is adapted from the original DINOv2 repository: https://github.com/facebookresearch/dinov2
# This code is licensed under the CC BY-NC-ND 4.0 license
# found in the LICENSE file in the root directory of this source tree.

import argparse
import logging
import math
import os
from functools import partial
from monai.transforms import Compose, LoadImaged, ScaleIntensityRangePercentilesd, Lambdad
import random
import wandb

from fvcore.common.checkpoint import PeriodicCheckpointer
import torch

from dinov2.data import SamplerType, make_data_loader, make_dataset_3d
from dinov2.data import collate_data_and_cast, DataAugmentationDINO3d, MaskingGenerator3d, CropForegroundSwapSliceDims, Printer
import dinov2.distributed as distributed
from dinov2.fsdp import FSDPCheckpointer
from dinov2.logging import MetricLogger
from dinov2.utils.config import setup_3d
from dinov2.utils.utils import CosineScheduler

from dinov2.train.ssl_meta_arch import SSLMetaArch

torch.backends.cuda.matmul.allow_tf32 = True  # PyTorch 1.12 sets this to False by default
logger = logging.getLogger("dinov2")


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("3DINO training", add_help=add_help)
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not attempt to resume from the checkpoint directory. ",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--eval", type=str, default="", help="Eval type to perform")
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default="",
        type=str,
        help="Output directory to save logs and checkpoints",
    )
    parser.add_argument("--local-rank", default=0, type=int, help="Variable for distributed computing.")
    parser.add_argument(
        "--cache-dir",
        default=None,
        type=str,
        help="path to cache directory for monai persistent dataset"
    )

    parser.add_argument(
        "--entity",
        default='benoit-gerin',
        type=str,
        help="wandb entity"
    )

    parser.add_argument(
        "--project",
        default='fomo2025',
        type=str,
        help="wandb project"
    )

    parser.add_argument(
        "--mode",
        default='disabled',
        type=str,
        choices=['disabled', 'online'],
        help="wandb mode"
    )

    return parser


def build_optimizer(cfg, params_groups):
    return torch.optim.AdamW(params_groups, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))


def build_schedulers(cfg):
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    lr = dict(
        base_value=cfg.optim["lr"],
        final_value=cfg.optim["min_lr"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=0,
    )
    wd = dict(
        base_value=cfg.optim["weight_decay"],
        final_value=cfg.optim["weight_decay_end"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    momentum = dict(
        base_value=cfg.teacher["momentum_teacher"],
        final_value=cfg.teacher["final_momentum_teacher"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    teacher_temp = dict(
        base_value=cfg.teacher["teacher_temp"],
        final_value=cfg.teacher["teacher_temp"],
        total_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=cfg.teacher["warmup_teacher_temp"],
    )

    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)
    last_layer_lr_schedule = CosineScheduler(**lr)

    last_layer_lr_schedule.schedule[
        : cfg.optim["freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH
    ] = 0  # mimicking the original schedules

    logger.info("Schedulers ready.")

    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    )


def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    for param_group in optimizer.param_groups:
        is_last_layer = param_group["is_last_layer"]
        lr_multiplier = param_group["lr_multiplier"]
        wd_multiplier = param_group["wd_multiplier"]
        param_group["weight_decay"] = wd * wd_multiplier
        param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier


def do_test(cfg, model, iteration):
    new_state_dict = model.teacher.state_dict()

    if distributed.is_main_process():
        iterstring = str(iteration)
        eval_dir = os.path.join(cfg.train.output_dir, "eval", iterstring)
        os.makedirs(eval_dir, exist_ok=True)
        # save teacher checkpoint
        teacher_ckp_path = os.path.join(eval_dir, "teacher_checkpoint.pth")
        torch.save({"teacher": new_state_dict}, teacher_ckp_path)


def check_data(cfg):
    inputs_dtype = torch.half
    # checkpointer
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    max_iter = OFFICIAL_EPOCH_LENGTH
    start_iter = 0
    # setup data preprocessing
    img_size = cfg.crops.global_crops_size
    patch_size = cfg.student.patch_size
    n_tokens = (img_size // patch_size) ** 3
    mask_generator = MaskingGenerator3d(
        input_size=(img_size // patch_size, img_size // patch_size, img_size // patch_size)
    )

    def random_select_time(x):
        # if time axis exists, select random time slice
        if x.shape[0] > 1:
            t = random.randint(0, x.shape[0] - 1)
            x = x[t:t + 1]
        return x

    # Compose the loading and intensity scaling here to cache transforms in monai persistent dataset
    data_transform = Compose(
            [
                #Printer(),
                LoadImaged(keys=["image"], ensure_channel_first=True),
                Lambdad(keys=["image"], func=random_select_time),
                Lambdad(
                    keys=["image"], func=lambda x: torch.nan_to_num(x, torch.nanmean(x).item())
                ),  # replace NaNs with mean
                #Printer(),
                ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=-1, b_max=1, clip=True),
                CropForegroundSwapSliceDims(select_fn=lambda x: x > -1),
                DataAugmentationDINO3d(
                    cfg.crops.global_crops_in_slice_scale,
                    cfg.crops.global_crops_cross_slice_scale,
                    cfg.crops.local_crops_in_slice_scale,
                    cfg.crops.local_crops_cross_slice_scale,
                    cfg.crops.local_crops_number,
                    global_crops_size=cfg.crops.global_crops_size,
                    local_crops_size=cfg.crops.local_crops_size,
                )
            ]
        )

    # data collate
    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        dtype=inputs_dtype,
    )

    # setup data loader
    dataset = make_dataset_3d(
        dataset_path=cfg.train.dataset_path,
        cache_path=cfg.train.cache_dir,
        data_min_axis_size=cfg.train.data_min_axis_size,
        transform=data_transform
    )
    sampler_type = SamplerType.SHARDED_INFINITE
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        seed=start_iter,
        sampler_type=sampler_type,
        sampler_advance=0,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # training loop
    iteration = start_iter
    header = "Checking data"

    import time

    print("Num iter in one epoch: ", OFFICIAL_EPOCH_LENGTH)
    for data in data_loader:

        current_batch_size = data["collated_global_crops"].shape[0] / 2
        if iteration > max_iter:
            return

        # Here, check data

        keys = list(data.keys())
        for key in keys:
            tensor = data[key]
            if isinstance(tensor, torch.Tensor):
                n_nans = torch.sum(torch.isnan(tensor)).item()
                n_bg = torch.sum(tensor == -1).item()

                if tensor.numel() < 25:
                    print("Small tensor found in: ", key, " with shape: ", tensor.shape, " and numel: ", tensor.numel())

                if n_nans > 0:
                    print("NaNs found in key: ", key, " with shape: ", tensor.shape, " and n_nans: ", n_nans, "    prop = ", n_nans / tensor.numel())

                if (n_bg / tensor.numel() > 0.9):
                    print("too much background in key: ", key, " with shape: ", tensor.shape, " and n_bg: ", n_bg, "    prop = ", n_bg / tensor.numel())


        iteration = iteration + 1




def main(args):
    cfg = setup_3d(args)

    check_data(cfg)







if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)
