# Author: Tony Xu
#
# This code is adapted from the original DINOv2 repository: https://github.com/facebookresearch/dinov2
# This code is licensed under the CC BY-NC-ND 4.0 license
# found in the LICENSE file in the root directory of this source tree.

import logging
from enum import Enum
from typing import Any, Callable, List, Optional, TypeVar
import os
from copy import deepcopy
import random

import torch
from torch.utils.data import Sampler
from monai.data import CacheNTransDataset, PersistentDataset
import json

from .samplers import EpochSampler, InfiniteSampler, ShardedInfiniteSampler


logger = logging.getLogger("dinov2")


class SamplerType(Enum):
    DISTRIBUTED = 0
    EPOCH = 1
    INFINITE = 2
    SHARDED_INFINITE = 3
    SHARDED_INFINITE_NEW = 4


def _make_bool_str(b: bool) -> str:
    return "yes" if b else "no"


def _make_sample_transform(image_transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
    def transform(sample):
        image, target = sample
        if image_transform is not None:
            image = image_transform(image)
        if target_transform is not None:
            target = target_transform(target)
        return image, target

    return transform


def make_dataset_3d(
    *,
    dataset_path: str,
    cache_path: str,
    data_min_axis_size: int,
    transform: Optional[Callable] = None,
):
    """
    Creates a 3d input dataset with the specified parameters.

    Args:
        dataset_path: A path to a list of sample paths for MONAI datasets.
        cache_path: A path to a directory to cache the dataset.
        data_min_axis_size: The minimum size of the smallest axis of the data.
        transform: A transform to apply to images.
    Returns:
        The created dataset.
    """
    logger.info(f'creating 3d dataset from datalist: {dataset_path}')

    # load datalist
    with open(dataset_path, 'r') as json_f:
        datalist = json.load(json_f)

    # filter overly small data
    datalist = [x for x in datalist if min(x['shape'][:3]) > data_min_axis_size]
    dataset = CacheNTransDataset(datalist, transform=transform, cache_n_trans=5, cache_dir=cache_path)

    # Aggregated datasets do not expose (yet) these attributes, so add them.
    if not hasattr(dataset, "transform"):
        setattr(dataset, "transform", transform)

    return dataset


def make_segmentation_dataset_3d(
    dataset_name: str,
    dataset_percent: int,
    base_directory: str,
    train_transforms: Callable,
    val_transforms: Callable,
    cache_path: str,
    batch_size: int,
):
    """
    Creates a 3d segmentation dataset with the specified parameters.

    Args:
        dataset_name: Name of the segmentation dataset (BTCV, BraTS, LA-SEG, TDSC-ABUS).
        dataset_percent: Percentage of the dataset to use for training.
        base_directory: Base directory where dataset json files are stored.
        train_transforms: Training transforms to apply to images.
        val_transforms: Validation transforms to apply to images.
        cache_path: A path to a directory to cache the dataset, used in PersistentDataset.
        batch_size: Batch size for the dataset.
    Returns:
        Created train, val, and test datasets, number of input channels, and number of classes for the dataset.
    """

    if dataset_name == 'BTCV':
        datalist_path = os.path.join(base_directory, 'BTCV_100_datalist.json')
        class_num = 14
        input_channels = 1
    elif dataset_name == 'BraTS':
        datalist_path = os.path.join(base_directory, 'BraTS_100_datalist.json')
        class_num = 3
        input_channels = 4
    elif dataset_name == 'LA-SEG':
        datalist_path = os.path.join(base_directory, 'LA-SEG_100_datalist.json')
        class_num = 2
        input_channels = 1
    elif dataset_name == 'TDSC-ABUS':
        datalist_path = os.path.join(base_directory, 'TDSC-ABUS_100_datalist.json')
        class_num = 2
        input_channels = 1
    else:
        raise ValueError(f'Unsupported dataset "{dataset_name}"')

    with open(datalist_path, 'r') as json_f:
        datalist = json.load(json_f)

    train_data_ind = int(round(len(datalist['training']) * (dataset_percent / 100)))

    train_datalist = datalist['training'][:train_data_ind]
    val_datalist = datalist['validation']
    test_datalist = datalist['test']
    logger.info(f"# of train samples: {len(train_datalist):,d}")
    logger.info(f"# of val samples: {len(val_datalist):,d}")
    logger.info(f"# of test samples: {len(test_datalist):,d}")

    if len(train_datalist) < batch_size:
        logger.info(f"copying train samples to match batch size: {batch_size:,d}")
        copied_datalist = []
        for i in range(batch_size // len(train_datalist)):
            copied_datalist.extend(deepcopy(train_datalist))
        assert len(copied_datalist) == batch_size
        train_datalist = copied_datalist

    train_dataset = PersistentDataset(train_datalist, transform=train_transforms, cache_dir=cache_path)
    val_dataset = PersistentDataset(val_datalist, transform=val_transforms, cache_dir=cache_path)
    test_dataset = PersistentDataset(test_datalist, transform=val_transforms, cache_dir=cache_path)

    return train_dataset, val_dataset, test_dataset, input_channels, class_num


def make_classification_dataset_3d(
    dataset_name: str,
    dataset_percent: int,
    base_directory: str,
    train_transforms: Callable,
    val_transforms: Callable,
    cache_path: str,
    dataset_seed: int,
):
    """
    Creates a 3d classification dataset with the specified parameters.

    Args:
        dataset_name: Name of the classification dataset (ICBM, COVID-CT-MD).
        dataset_percent: Percentage of the dataset to use for training.
        base_directory: Base directory where dataset json files are stored.
        train_transforms: Training transforms to apply to images.
        val_transforms: Validation transforms to apply to images.
        cache_path: A path to a directory to cache the dataset, used in PersistentDataset.
        dataset_seed: Seed for random shuffling of the dataset.
    Returns:
        Created train, val, and test datasets, and number of classes for the dataset.
    """

    if dataset_name == 'ICBM':
        datalist_path = os.path.join(base_directory, 'ICBM_cls_datalist.json')
        class_num = 4
    elif dataset_name == 'COVID-CT-MD':
        datalist_path = os.path.join(base_directory, 'COVID-CT-MD_cls_datalist.json')
        class_num = 3
    else:
        raise ValueError(f'Unsupported dataset "{dataset_name}"')

    with open(datalist_path, 'r') as json_f:
        datalist = json.load(json_f)

    # filter ages for icbm
    if dataset_name == 'ICBM':

        for k in datalist:
            for item in datalist[k]:
                item['image'] = item['image'].replace('.nii.gz', '_mask.nii.gz')

        datalist['training'] = [x for x in datalist['training'] if 20 <= x['label'] <= 60]
        datalist['validation'] = [x for x in datalist['validation'] if 20 <= x['label'] <= 60]
        datalist['test'] = [x for x in datalist['test'] if 20 <= x['label'] <= 60]

    # ensure reproducible shuffling
    random.Random(dataset_seed).shuffle(datalist['training'])
    print(f'Shuffled with seed: {dataset_seed}')

    train_data_ind = int(round(len(datalist['training']) * (dataset_percent / 100)))
    train_datalist = datalist['training'][:train_data_ind]
    val_datalist = datalist['validation']
    test_datalist = datalist['test']

    logger.info(f"# of train samples: {len(train_datalist):,d}")
    logger.info(f"# of val samples: {len(val_datalist):,d}")
    logger.info(f"# of test samples: {len(test_datalist):,d}")

    train_dataset = PersistentDataset(train_datalist, transform=train_transforms, cache_dir=cache_path)
    val_dataset = PersistentDataset(val_datalist, transform=val_transforms, cache_dir=cache_path)
    test_dataset = PersistentDataset(test_datalist, transform=val_transforms, cache_dir=cache_path)

    return train_dataset, val_dataset, test_dataset, class_num


def make_regression_dataset_3d(
    dataset_name: str,
    dataset_percent: int,
    base_directory: str,
    train_transforms: Callable,
    val_transforms: Callable,
    cache_path: str,
    dataset_seed: int,
):
    """
    Creates a 3D regression dataset from a datalist JSON file.

    Args:
        dataset_name: Name of the regression dataset (e.g., ICBM).
        dataset_percent: Percent of training dataset to use.
        base_directory: Path to the directory containing datalist JSON.
        train_transforms: Transforms to apply to training samples.
        val_transforms: Transforms to apply to validation/test samples.
        cache_path: Path to directory for MONAI PersistentDataset cache.
        dataset_seed: Random seed to ensure reproducibility.

    Returns:
        train_dataset, val_dataset, test_dataset, and input_channels
    """

    if dataset_name == 'ICBM':
        datalist_path = os.path.join(base_directory, 'ICBM_regression_datalist.json')
        input_channels = 1
    elif dataset_name == 'FOMO':
        datalist_path = os.path.join(base_directory, 'fomo_fold_0.json')  # Or use fold name format
        input_channels = 3  # since you have image1, image2, image3
    else:
        raise ValueError(f'Unsupported regression dataset: "{dataset_name}"')

    with open(datalist_path, 'r') as json_f:
        datalist = json.load(json_f)

    # Optional: filter by a numeric range if needed (e.g. age 20–60)
    # datalist['training'] = [x for x in datalist['training'] if 20 <= x['label'] <= 60]

    random.Random(dataset_seed).shuffle(datalist['training'])
    logger.info(f"Shuffled training data with seed {dataset_seed}")

    train_data_ind = int(round(len(datalist['training']) * (dataset_percent / 100)))
    train_datalist = datalist['training'][:train_data_ind]
    val_datalist = datalist['validation']
    test_datalist = datalist['test']

    logger.info(f"# of train samples: {len(train_datalist):,d}")
    logger.info(f"# of val samples: {len(val_datalist):,d}")
    logger.info(f"# of test samples: {len(test_datalist):,d}")

    train_dataset = PersistentDataset(train_datalist, transform=train_transforms, cache_dir=cache_path)
    val_dataset = PersistentDataset(val_datalist, transform=val_transforms, cache_dir=cache_path)
    test_dataset = PersistentDataset(test_datalist, transform=val_transforms, cache_dir=cache_path)

    return train_dataset, val_dataset, test_dataset, input_channels




def _make_sampler(
    *,
    dataset,
    type: Optional[SamplerType] = None,
    shuffle: bool = False,
    seed: int = 0,
    size: int = -1,
    advance: int = 0,
) -> Optional[Sampler]:
    sample_count = len(dataset)

    if type == SamplerType.INFINITE:
        logger.info("sampler: infinite")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        return InfiniteSampler(
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
            advance=advance,
        )
    elif type in (SamplerType.SHARDED_INFINITE, SamplerType.SHARDED_INFINITE_NEW):
        logger.info("sampler: sharded infinite")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        # TODO: Remove support for old shuffling
        use_new_shuffle_tensor_slice = type == SamplerType.SHARDED_INFINITE_NEW
        return ShardedInfiniteSampler(
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
            advance=advance,
            use_new_shuffle_tensor_slice=use_new_shuffle_tensor_slice,
        )
    elif type == SamplerType.EPOCH:
        logger.info("sampler: epoch")
        if advance > 0:
            raise NotImplementedError("sampler advance > 0 is not supported")
        size = size if size > 0 else sample_count
        logger.info(f"# of samples / epoch: {size:,d}")
        return EpochSampler(
            size=size,
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
        )
    elif type == SamplerType.DISTRIBUTED:
        logger.info("sampler: distributed")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        if advance > 0:
            raise ValueError("sampler advance > 0 is invalid")
        return torch.utils.data.DistributedSampler(
            dataset=dataset,
            shuffle=shuffle,
            seed=seed,
            drop_last=False,
        )

    logger.info("sampler: none")
    return None


T = TypeVar("T")


def make_data_loader(
    *,
    dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    seed: int = 0,
    sampler_type: Optional[SamplerType] = SamplerType.INFINITE,
    sampler_size: int = -1,
    sampler_advance: int = 0,
    drop_last: bool = True,
    persistent_workers: bool = False,
    collate_fn: Optional[Callable[[List[T]], Any]] = None,
):
    """
    Creates a data loader with the specified parameters.

    Args:
        dataset: A dataset (third party, LaViDa or WebDataset).
        batch_size: The size of batches to generate.
        num_workers: The number of workers to use.
        shuffle: Whether to shuffle samples.
        seed: The random seed to use.
        sampler_type: Which sampler to use: EPOCH, INFINITE, SHARDED_INFINITE, SHARDED_INFINITE_NEW, DISTRIBUTED or None.
        sampler_size: The number of images per epoch (when applicable) or -1 for the entire dataset.
        sampler_advance: How many samples to skip (when applicable).
        drop_last: Whether the last non-full batch of data should be dropped.
        persistent_workers: maintain the workers Dataset instances alive after a dataset has been consumed once.
        collate_fn: Function that performs batch collation
    """

    sampler = _make_sampler(
        dataset=dataset,
        type=sampler_type,
        shuffle=shuffle,
        seed=seed,
        size=sampler_size,
        advance=sampler_advance,
    )

    logger.info("using PyTorch data loader")
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )

    try:
        logger.info(f"# of batches: {len(data_loader):,d}")
    except TypeError:  # data loader has no length
        logger.info("infinite data loader")
    return data_loader
