# Author: Tony Xu
#
# This code is adapted from the original DINOv2 repository: https://github.com/facebookresearch/dinov2
# This code is licensed under the CC BY-NC-ND 4.0 license
# found in the LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics import MetricCollection
from monai.inferers import sliding_window_inference
from typing import Dict, List, Tuple, Literal

from dinov2.data import DictDatasetWithEnumeratedTargets, SamplerType, make_data_loader
import dinov2.distributed as distributed
from dinov2.logging import MetricLogger
from dinov2.eval.segmentation_3d.vit_adapter import ViTAdapter


logger = logging.getLogger("dinov2")
Reduce = Literal["mean", "max", "median"]


def create_linear_input(x_tokens_list, use_n_blocks, use_avgpool):
    intermediate_output = x_tokens_list[-use_n_blocks:]
    output = torch.cat([class_token for _, class_token in intermediate_output], dim=-1)
    if use_avgpool:
        output = torch.cat(
            (
                output,
                torch.mean(intermediate_output[-1][0], dim=1),  # patch tokens
            ),
            dim=-1,
        )
        output = output.reshape(output.shape[0], -1)
    return output.float()


class ModelWithNormalize(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, samples):
        return nn.functional.normalize(self.model(samples), dim=1, p=2)


class ModelWithIntermediateLayers(nn.Module):
    def __init__(self, feature_model, n_last_blocks, autocast_ctx):
        super().__init__()
        self.feature_model = feature_model
        self.feature_model.eval()
        self.n_last_blocks = n_last_blocks
        self.autocast_ctx = autocast_ctx

    def forward(self, images):
        with torch.inference_mode():
            with self.autocast_ctx():
                features = self.feature_model.get_intermediate_layers(
                    images, self.n_last_blocks, return_class_token=True
                )
        return features


class MultiChannelFeatureModel(nn.Module):
    def __init__(self, vit_model, input_channels, n_last_blocks, autocast_ctx):
        super().__init__()
        self.vit = vit_model
        self.input_channels = input_channels
        self.n_last_blocks = n_last_blocks
        self.autocast_ctx = autocast_ctx

    def forward(self, x):
        # B, C, H, W, D → B*C, 1, H, W, D
        B, C = x.shape[0], x.shape[1]
        x = x.reshape(B * C, 1, *x.shape[2:])
        with torch.inference_mode():
            with self.autocast_ctx():
                class_tokens, patch_tokens = self.vit.get_intermediate_layers(
                    x, self.n_last_blocks, return_class_token=True
                )

        # Use the patch tokens
        features = patch_tokens
        
        # [B*C] → [B] - back to per sample features
        features = [f.reshape(B, C, *f.shape[1:]) for f in features]
        
        # Aggregate: e.g., mean over channels
        features = [f.mean(dim=1) for f in features]  # Now shape: [B, ...]
        
        return features


class ViTAdapterFeatureWrapper(nn.Module):
    def __init__(self, vit_model, input_channels, n_last_blocks, autocast_ctx):
        super().__init__()
        self.adapter = ViTAdapter(vit_model, input_channels)
        self.n_last_blocks = n_last_blocks
        self.autocast_ctx = autocast_ctx

    def forward(self, x):
        with torch.inference_mode():
            # with self.autocast_ctx():
            features = self.adapter(x)  # returns [f1, f2, f3, f4]

        # Simulate (patch_tokens, class_tokens) format
        outputs = []
        for feat in features[-self.n_last_blocks:]:
            B, C, H, W, D = feat.shape

            # Flatten spatial dims -- grid-like fts, spatial
            patch_tokens = feat.flatten(2).transpose(1, 2)  # [B, N_patches, C]

            # Use mean of all spatial tokens as the "class token" -- summarises context
            # class_token = patch_tokens.mean(dim=1, keepdim=True)  # [B, 1, C]
            class_token = patch_tokens.mean(dim=1)  # [B, C]

            outputs.append((patch_tokens, class_token))

        return outputs




@torch.inference_mode()
def evaluate(
    model: nn.Module,
    data_loader,
    postprocessors: Dict[str, nn.Module],
    metrics: Dict[str, MetricCollection],
    device: torch.device,
    criterion: Optional[nn.Module] = None,
):
    model.eval()
    if criterion is not None:
        criterion.eval()

    for metric in metrics.values():
        metric = metric.to(device)

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    for samples, targets, *_ in metric_logger.log_every(data_loader, 10, header):
        outputs = model(samples.to(device))
        targets = targets.to(device)

        if criterion is not None:
            loss = criterion(outputs, targets)
            metric_logger.update(loss=loss.item())

        for k, metric in metrics.items():
            metric_inputs = postprocessors[k](outputs, targets)
            metric.update(**metric_inputs)

    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")

    stats = {k: metric.compute() for k, metric in metrics.items()}
    metric_logger_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return metric_logger_stats, stats


def _pad_to_roi(patch: torch.Tensor, roi: Tuple[int, ...]) -> torch.Tensor:
    # patch: (1, C, H, W[, D]); roi: (RH, RW[, RD])
    spatial = patch.shape[2:]
    pad_spatial = [max(r - s, 0) for s, r in zip(spatial, roi)]
    if len(roi) == 2:
        # (W_left, W_right, H_left, H_right)
        pad = (0, pad_spatial[1], 0, pad_spatial[0])
    else:
        # (D_left, D_right, W_left, W_right, H_left, H_right)
        pad = (0, pad_spatial[2], 0, pad_spatial[1], 0, pad_spatial[0])
    return F.pad(patch, pad, value=-1)

def _positions(S: int, R: int, ov: float) -> List[int]:
    if S <= R:
        return [0]
    step = max(1, int(R * (1.0 - ov)))
    pos = list(range(0, S - R + 1, step))
    if pos[-1] != S - R:
        pos.append(S - R)
    return pos

def _coords(spatial: Tuple[int, ...], roi: Tuple[int, ...], overlap: float):
    axes = [_positions(S, R, overlap) for S, R in zip(spatial, roi)]
    if len(roi) == 2:
        for h in axes[0]:
            for w in axes[1]:
                yield (h, w)
    else:
        for h in axes[0]:
            for w in axes[1]:
                for d in axes[2]:
                    yield (h, w, d)

@torch.inference_mode()
def predict_reduce_tokens(
    backbone: nn.Module,                         # returns x_tokens_list
    heads: Dict[str, nn.Module],                 # {name: LinearRegressor}
    x: torch.Tensor,                             # (B,C,H,W[,D])
    roi: Tuple[int, ...],                        # (RH,RW[,RD])
    overlap: float = 0.5,
    sw_bs: int = 1,
    reduce: Reduce = "mean",
    create_linear_input_fn=None,                 # pass your create_linear_input
) -> Dict[str, torch.Tensor]:
    assert create_linear_input_fn is not None, "Pass create_linear_input_fn=create_linear_input"

    B, spatial = x.shape[0], x.shape[2:]
    coords = list(_coords(spatial, roi, overlap))
    device = x.device

    # Accumulators per head
    if reduce == "mean":
        # running mean of predictions: (B, num_outputs)
        pred_mean: Dict[str, torch.Tensor] = {}
        counts = torch.zeros(B, dtype=torch.long)
    elif reduce == "max":
        pred_max: Dict[str, torch.Tensor] = {}
        seen = torch.zeros(B, dtype=torch.bool)
    else:  # median
        # store per-patch predictions on CPU to save VRAM
        pred_lists: Dict[str, List[List[torch.Tensor]]] = {k: [[] for _ in range(B)] for k in heads.keys()}

    batch_patches: List[torch.Tensor] = []
    owners: List[int] = []

    def flush():
        nonlocal batch_patches, owners, counts, seen
        if not batch_patches:
            return
        batch = torch.cat(batch_patches, 0)  # (Npatch, C, *roi)

        # 1) Backbone forward on this mini-batch
        x_tokens_list = backbone(batch)  # list/tuple per block, each with batch dim

        # 2) For each head, build linear input and compute patch predictions
        for name, head in heads.items():
            li = create_linear_input_fn(x_tokens_list, head.use_n_blocks, head.use_avgpool)
            if li.ndim > 2:
                li = li.reshape(li.shape[0], -1)
            preds_patch = head.linear(li)  # (Npatch, num_outputs)

            if reduce == "mean":
                # streaming mean per owner
                if name not in pred_mean:
                    pred_mean[name] = torch.zeros((B, preds_patch.shape[1]), device=device, dtype=preds_patch.dtype)
                # update per-row
                for row_idx, owner in enumerate(owners):
                    n = counts[owner].item()
                    current = pred_mean[name][owner]
                    pred_mean[name][owner] = current + (preds_patch[row_idx] - current) / (n + 1)
                # update counts once per owner occurrence
                for owner in owners:
                    counts[owner] += 1

            elif reduce == "max":
                if name not in pred_max:
                    pred_max[name] = torch.empty((B, preds_patch.shape[1]), device=device, dtype=preds_patch.dtype)
                for row_idx, owner in enumerate(owners):
                    if not seen[owner]:
                        pred_max[name][owner] = preds_patch[row_idx]
                    else:
                        pred_max[name][owner] = torch.maximum(pred_max[name][owner], preds_patch[row_idx])
                for owner in owners:
                    seen[owner] = True

            else:  # median
                # move to CPU immediately to save GPU mem
                preds_cpu = preds_patch.detach().cpu()
                for row_idx, owner in enumerate(owners):
                    pred_lists[name][owner].append(preds_cpu[row_idx])

        batch_patches, owners = [], []

    # Build mini-batches of patches
    for i in range(B):
        for c in coords:
            if x.ndim == 4:
                h, w = c
                patch = x[i:i+1, :, h:h+roi[0], w:w+roi[1]]
            else:
                h, w, d = c
                patch = x[i:i+1, :, h:h+roi[0], w:w+roi[1], d:d+roi[2]]
            patch = _pad_to_roi(patch, roi)
            batch_patches.append(patch)
            owners.append(i)
            if len(batch_patches) >= sw_bs:
                flush()
    flush()

    # 3) Finalize
    outs: Dict[str, torch.Tensor] = {}
    if reduce == "mean":
        outs = pred_mean
    elif reduce == "max":
        outs = pred_max
    else:  # median
        for name in heads.keys():
            preds_k = []
            for i in range(B):
                stack = torch.stack(pred_lists[name][i], dim=0)  # (Npatch_i, num_outputs) on CPU
                med = torch.median(stack, dim=0).values
                preds_k.append(med)
            outs[name] = torch.stack(preds_k, dim=0).to(device)

    return outs  # dict[name] -> (B, num_outputs)



@torch.inference_mode()
def evaluate_dict(
    model: nn.Module,
    data_loader,
    linear_regressors,
    postprocessors: Dict[str, nn.Module],
    metrics: Dict[str, MetricCollection],
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    return_preds: bool = False,
    reduce: Literal = "median",
):    
    if criterion is not None:
        criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    pred_dict = {}

    model.eval(); linear_regressors.eval()
    for k,m in metrics.items(): metrics[k] = m.to(device)

    for samples in metric_logger.log_every(data_loader, 10, header):
        x = samples["image"].to(device)
        tgt = samples["label"].to(device)
        batch_size = x.shape[0]
        roi = (112,112) if x.ndim==4 else (112,112,112)

        outs = predict_reduce_tokens(
            backbone=model,
            heads=linear_regressors.regressors_dict,
            x=x,
            roi=(112,112) if x.ndim==4 else (112,112,112),
            overlap=0.5,
            # sw_bs=1,                      # raise for speed if you have headroom
            sw_bs=batch_size,
            reduce=reduce,              # "mean" | "max" | "median"
            create_linear_input_fn=create_linear_input,
        )

        # if criterion is not None:
        #     loss = criterion(outputs, targets)
        #     metric_logger.update(loss=loss.item())

        for k, metric in metrics.items():            
            preds = outs[k]                          # (B, C)
            if preds.ndim == 1:                       # just in case
                preds = preds.unsqueeze(0)            # (1, C)
        
            target = tgt if not isinstance(tgt, dict) else tgt[k]
            target = target.long().view(-1)           # (B,)
            
            if preds.shape[1] == 1:
                target = target.squeeze()
                preds = preds.squeeze()
                
            metric.update(preds=preds, target=target)

            if k not in pred_dict:
                pred_dict[k] = {'preds': [], 'target': []}
            pred_dict[k]['preds'].append(preds.detach().cpu())
            pred_dict[k]['target'].append(target.detach().cpu())

    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")

    stats = {k: metric.compute() for k, metric in metrics.items()}
    metric_logger_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    if return_preds:
        return metric_logger_stats, stats, pred_dict
    return metric_logger_stats, stats


def all_gather_and_flatten(tensor_rank):
    tensor_all_ranks = torch.empty(
        distributed.get_global_size(),
        *tensor_rank.shape,
        dtype=tensor_rank.dtype,
        device=tensor_rank.device,
    )
    tensor_list = list(tensor_all_ranks.unbind(0))
    torch.distributed.all_gather(tensor_list, tensor_rank.contiguous())
    return tensor_all_ranks.flatten(end_dim=1)


def extract_features_dict(model, dataset, batch_size, num_workers, gather_on_cpu=False):
    dataset_with_enumerated_targets = DictDatasetWithEnumeratedTargets(dataset)
    sample_count = len(dataset_with_enumerated_targets)
    data_loader = make_data_loader(
        dataset=dataset_with_enumerated_targets,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
        shuffle=False,
    )
    return extract_features_with_dataloader(model, data_loader, sample_count, gather_on_cpu)


@torch.inference_mode()
def extract_features_with_dataloader(model, data_loader, sample_count, gather_on_cpu=False):
    gather_device = torch.device("cpu") if gather_on_cpu else torch.device("cuda")
    metric_logger = MetricLogger(delimiter="  ")
    features, all_labels = None, None
    for samples, (index, labels_rank) in metric_logger.log_every(data_loader, 10):
        samples = samples.cuda(non_blocking=True)
        labels_rank = labels_rank.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        features_rank = model(samples).float()

        # init storage feature matrix
        if features is None:
            features = torch.zeros(sample_count, features_rank.shape[-1], device=gather_device)
            labels_shape = list(labels_rank.shape)
            labels_shape[0] = sample_count
            all_labels = torch.full(labels_shape, fill_value=-1, device=gather_device)
            logger.info(f"Storing features into tensor of shape {features.shape}")

        # share indexes, features and labels between processes
        index_all = all_gather_and_flatten(index).to(gather_device)
        features_all_ranks = all_gather_and_flatten(features_rank).to(gather_device)
        labels_all_ranks = all_gather_and_flatten(labels_rank).to(gather_device)

        # update storage feature matrix
        if len(index_all) > 0:
            features.index_copy_(0, index_all, features_all_ranks)
            all_labels.index_copy_(0, index_all, labels_all_ranks)

    logger.info(f"Features shape: {tuple(features.shape)}")
    logger.info(f"Labels shape: {tuple(all_labels.shape)}")

    assert torch.all(all_labels > -1)

    return features, all_labels
