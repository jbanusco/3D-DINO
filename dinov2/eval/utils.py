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

from dinov2.data import DictDatasetWithEnumeratedTargets, SamplerType, make_data_loader
import dinov2.distributed as distributed
from dinov2.logging import MetricLogger
from dinov2.eval.segmentation_3d.vit_adapter import ViTAdapter


logger = logging.getLogger("dinov2")


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


def _pad_to_roi(patch, roi):
    # patch: (1, C, H, W[, D]); roi is (RH, RW[, RD])
    spatial = patch.shape[2:]
    pad_spatial = [max(r - s, 0) for s, r in zip(spatial, roi)]
    if len(roi) == 2:
        # F.pad uses (W_left, W_right, H_left, H_right)
        pad = (0, pad_spatial[1], 0, pad_spatial[0])
    else:
        # (D_left, D_right, W_left, W_right, H_left, H_right)
        pad = (0, pad_spatial[2], 0, pad_spatial[1], 0, pad_spatial[0])
    return F.pad(patch, pad)


class PatchReducer:
    @staticmethod
    def _positions(S, R, ov):
        if S <= R: return [0]
        step = max(1, int(R * (1-ov)))
        pos = list(range(0, S - R + 1, step))
        if pos[-1] != S - R: pos.append(S - R)
        return pos

    @staticmethod
    def coords(spatial, roi, overlap):
        axes = [PatchReducer._positions(S, R, overlap) for S, R in zip(spatial, roi)]
        if len(roi) == 2:
            return [(h,w) for h in axes[0] for w in axes[1]]
        else:
            return [(h,w,d) for h in axes[0] for w in axes[1] for d in axes[2]]

    @staticmethod
    @torch.inference_mode()
    def predict_reduce(backbone, heads: dict[str, nn.Module], x: torch.Tensor,
                       roi, overlap=0.5, sw_bs=1, reduce="mean"):
        B, spatial = x.shape[0], x.shape[2:]
        coords = PatchReducer.coords(spatial, roi, overlap)

        # collect per-sample per-patch features once
        per_sample_feats: list[list[torch.Tensor]] = [[] for _ in range(B)]
        batch_patches, owners = [], []

        def flush():
            nonlocal batch_patches, owners
            if not batch_patches: return
            batch = torch.cat(batch_patches, 0)
            feats = backbone(batch)  # (Npatch, feat_dim, ...)
            # split back
            idx = 0
            for i in owners:
                per_sample_feats[i].append(feats[idx].detach())
                idx += 1
            batch_patches, owners = [], []

        for i in range(B):
            for c in coords:
                if x.ndim == 4:
                    h,w = c
                    patch = x[i:i+1, :, h:h+roi[0], w:w+roi[1]]
                else:
                    h,w,d = c
                    patch = x[i:i+1, :, h:h+roi[0], w:w+roi[1], d:d+roi[2]]
                patch = _pad_to_roi(patch, roi)
                batch_patches.append(patch)
                owners.append(i)
                if len(batch_patches) >= sw_bs: flush()
        flush()

        # reduce features then apply heads (or apply heads per patch then reduce — both ok)
        outs: dict[str, torch.Tensor] = {}
        for i in range(B):
            stack = torch.stack(per_sample_feats[i], 0)  # (Npatch, feat_dim, ...)
            feats_red = stack.mean(0) if reduce=="mean" else stack.max(0).values
            per_sample_feats[i] = feats_red

        # run heads once per sample
        for k, head in heads.items():
            preds = []
            for i in range(B):
                p = head(per_sample_feats[i].unsqueeze(0))  # (1, *)
                preds.append(p.squeeze(0))
            outs[k] = torch.stack(preds, 0)  # (B, *)
        return outs



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
        roi = (112,112) if x.ndim==4 else (112,112,112)

        outs = PatchReducer.predict_reduce(
            backbone=model,
            heads=linear_regressors.regressors_dict,
            x=x,
            roi=roi,
            overlap=0.5,
            sw_bs=1,
            reduce="mean",
        )  # dict[k] -> (B, *)

        # if criterion is not None:
        #     loss = criterion(outputs, targets)
        #     metric_logger.update(loss=loss.item())

        for k, metric in metrics.items():            
            # metric_inputs = postprocessors[k](outputs, targets)            
            mi = postprocessors[k](outs[k], tgt if not isinstance(tgt, dict) else tgt[k])
            preds, target = mi["preds"], mi["target"]
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
