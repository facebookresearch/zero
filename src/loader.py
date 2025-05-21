# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, List
import json
import math
import os
import re
import ast
import torch
import torch.distributed as dist

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset
from torchvision.transforms import v2 as T

"""
simply register the dataset here
"""

DATASETS_META = {
    "llava-1.5-558k-captions": {
        "root": "llava-1.5-558k-captions/train_split",
        "json_shuffled": "blip_laion_cc_sbu_558k.json",
    },
    "llava-1.5-665k-instructions": {
        "root": "llava-1.5-665k-instructions/train_split",
        "json_shuffled": "llava_v1_5_mix665k_shuffled.json",
    },
    "llava-1.5-665k-genqa-500k-instructions": {
        "root": "llava-1.5-665k-instructions/train_split",
        "json_shuffled": "llava-665k_genqa-500k_shuffled.json",
    },
}


"""
classes
"""


class JsonDataset(Dataset):
    """
    for distributed sampler, the size and order of the dataset is the same across all ranks
    """

    def __init__(
        self,
        ds_root,
        ds_name,
        image_processor: object = None,
        image_input_size: int = 336,
        image_ratio_type: str = "pad",
        enable_image_aug: bool = False,
        enable_sampler_shuffle: bool = False,
        image_aug_rotation_degree: int = 5,
        max_input_image_size: Optional[int] = None,
        min_input_image_size: Optional[int] = None,
        anyres_grid_pinpoints: Optional[str] = None,
        simple_raw_image_mode: bool = False,
    ):
        super(JsonDataset, self).__init__()

        if isinstance(ds_name, str):
            ds_name = [ds_name]

        if image_ratio_type not in ["pad", "anyres"]:
            raise ValueError(
                f"unknown image_ratio_type: {image_ratio_type}, "
                f"should be one of ['pad', 'anyres']"
            )

        self.ds_root = ds_root

        # if False, prefer to load the json with a fixed order of samples
        self.enable_sampler_shuffle = enable_sampler_shuffle

        ds_items = []
        for ds in ds_name:
            assert (
                ds in DATASETS_META
            ), f"dataset {ds} not found in {DATASETS_META.keys()}"

            data_fold = DATASETS_META[ds]["root"]
            if self.enable_sampler_shuffle:
                json_path = DATASETS_META[ds]["json"]
            else:
                # fix the order of samples for training
                json_path = DATASETS_META[ds]["json_shuffled"]
            json_file = os.path.join(ds_root, ds, json_path)
            print(f"loading {json_file}")
            with open(json_file, "r") as f:
                d = json.load(f)  # List[Dict[str, Any]]
            ds_items += [dict(item, fold=data_fold) for item in d]

        self.ds_items = ds_items
        self.enable_image_aug = enable_image_aug
        self.image_input_size = image_input_size
        self.image_ratio_type = image_ratio_type
        self.transform = image_processor

        if not simple_raw_image_mode:
            self.bg_value = tuple(int(x * 255) for x in self.transform.image_mean)

        # control the number of image tokens for anyres
        self.max_input_image_size = max_input_image_size
        self.min_input_image_size = min_input_image_size

        # grid points from llava-next for anyres
        self.anyres_grid_pinpoints = anyres_grid_pinpoints

        self.img_augs = None
        self.image_aug_rotation_degree = image_aug_rotation_degree
        if self.enable_image_aug:
            self.img_augs = T.Compose(
                [
                    T.RandomRotation(
                        degrees=self.image_aug_rotation_degree,
                        interpolation=Image.BICUBIC,
                        fill=self.bg_value,
                    ),
                ]
            )

        # in this mode, the image is not processed by the image processor
        self.simple_raw_image_mode = simple_raw_image_mode

    def __getitem__(self, idx):
        item = self.ds_items[idx]

        id = item["id"]

        conversations = item["conversations"]
        _conv = []
        for i, t in enumerate(conversations):
            if t["from"] == "human":
                role = "user"
            elif t["from"] == "gpt":
                role = "model"
            else:
                raise ValueError(f"unknown role: {t['from']}")

            _conv.append({"role": role, "content": t["value"]})
        conversations = _conv

        if "image" in item.keys():
            image_path = os.path.join(
                self.ds_root,
                item["fold"],
                item["image"],
            )
            image = Image.open(image_path).convert("RGB")

            if self.enable_image_aug:
                image = self.img_augs(image)

            if self.simple_raw_image_mode:
                return image, conversations, id

            if self.image_ratio_type == "pad":
                image = expand2square(image, self.bg_value)
                image = self.transform(image, return_tensors="pt")[
                    "pixel_values"
                ]  # [patches, 3, h, w], where patches is 1
            elif self.image_ratio_type == "anyres":
                # control the number of total image tokens
                if (
                    self.max_input_image_size is not None
                    or self.min_input_image_size is not None
                ):
                    ori_size = image.size
                    image = T.functional.resize(
                        image,
                        size=self.min_input_image_size,
                        max_size=self.max_input_image_size,
                    )
                    # print(f"resized image from {ori_size} to {image.size}")

                image = process_anyres_image(
                    image,
                    self.transform,
                    self.anyres_grid_pinpoints,
                    self.bg_value,
                )
            else:
                raise ValueError(f"unknown image_ratio_type: {self.image_ratio_type}")

        else:
            if self.simple_raw_image_mode:
                # return empty PIL image
                image = Image.new("RGB", (336, 336), (0, 0, 0))
            else:
                # handle pure text samples without image
                image_size = self.transform.size["shortest_edge"]
                image = torch.zeros(1, 3, image_size, image_size)  # [patches, 3, h, w]

        return image, conversations, id

    def __len__(self):
        return len(self.ds_items)

    def __repr__(self):
        return (
            f"JsonDataset(num_samples={len(self.ds_items)}, "
            f"image_input_size={self.image_input_size}, "
            f"image_ratio_type={self.image_ratio_type}, "
            f"transform={self.transform}, "
            f"enable_sampler_shuffle={self.enable_sampler_shuffle}, "
            f"enable_image_aug={self.enable_image_aug}, "
            f"image_aug_rotation_degree={self.image_aug_rotation_degree}, "
            f"max_input_image_size={self.max_input_image_size}, "
            f"min_input_image_size={self.min_input_image_size}, "
            f"anyres_grid_pinpoints={self.anyres_grid_pinpoints}, "
            f"simple_raw_image_mode={self.simple_raw_image_mode}, "
            f"ds_root={self.ds_root})"
        )

    @property
    def lengths(self) -> List[int]:
        length_list = []
        for item in self.ds_items:
            length = sum(len(t["value"].split()) for t in item["conversations"])
            length = length if "image" in item.keys() else -length

            length_list.append(length)
        return length_list


def build_dataloader(
    cfg,
    global_rank: int = 0,
    world_size: int = 1,
    is_train: bool = True,
    image_processor: object = None,
):
    ds = JsonDataset(
        ds_root=cfg.data_root,
        ds_name=cfg.data_name,
        enable_sampler_shuffle=cfg.enable_sampler_shuffle,
        image_processor=image_processor,
        image_input_size=cfg.image_input_size,
        image_ratio_type=cfg.image_ratio_type,
        enable_image_aug=cfg.enable_image_aug,
        image_aug_rotation_degree=cfg.image_aug_rotation_degree,
        max_input_image_size=cfg.max_input_image_size,
        min_input_image_size=cfg.min_input_image_size,
        anyres_grid_pinpoints=cfg.anyres_grid_pinpoints,
        simple_raw_image_mode=cfg.simple_raw_image_mode,
    )
    print(f"dataset: {ds}")

    enable_shuffle = is_train and cfg.enable_sampler_shuffle
    print(f"shuffle samples: {enable_shuffle}")

    if cfg.enable_dist_length_grouped_sampler:
        sampler = DistributedLengthGroupedSampler(
            ds,
            batch_size=cfg.batch_size,  # per GPU
            mega_batch_mult=world_size * cfg.gradient_accumulation_steps,
            num_replicas=world_size,
            rank=global_rank,
            shuffle=enable_shuffle,
            seed=cfg.seed,  # need to be identical across all ranks
            drop_last=False,
            enable_length_group_by_modality=cfg.enable_length_group_by_modality,
        )
    else:
        sampler = DistributedSampler(
            ds,
            num_replicas=world_size,
            rank=global_rank,
            shuffle=enable_shuffle,
            seed=cfg.seed,  # need to be identical across all ranks
            drop_last=False,
        )
    print(f"sampler: {sampler}")

    def _collate_fn(data):
        imgs = []
        caps = []
        keys = []
        for d in data:
            imgs.append(d[0])
            caps.append(d[1])
            keys.append(d[2])
        return imgs, caps, keys

    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        sampler=sampler,
        collate_fn=_collate_fn,
    )
    return dl, sampler


"""
loader for hf datasets
"""

DATASETS_TEXT_META = {
    "genqa": "tomg-group-umd/GenQA_rebalanced",
    "cvbench": "nyu-visionx/CV-Bench",
}


def build_hf_dataloader(cfg, global_rank, world_size, is_train=True, **kwargs):
    ds_name = cfg.data_name
    if isinstance(ds_name, str):
        ds_name = [ds_name]

    ds_items = []
    for ds in ds_name:
        assert (
            ds in DATASETS_TEXT_META
        ), f"dataset {ds} not found in {DATASETS_TEXT_META.keys()}"

        ds = DATASETS_TEXT_META[ds]  # hf dataset name
        print(f"loading {ds} from hf")
        ds = load_dataset(
            ds,
            cache_dir=f"{cfg.hf_cache_dir}/datasets",
        )
        ds_items += [ds[k] for k in ds.keys()]

    ds = torch.utils.data.ConcatDataset(ds_items)
    sampler = DistributedSampler(
        ds,
        num_replicas=world_size,
        rank=global_rank,
        shuffle=is_train,
        seed=cfg.seed,  # need to be identical across all ranks
        drop_last=is_train,
    )

    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        sampler=sampler,
        collate_fn=lambda x: x,
    )
    return dl, sampler


"""
helper functions
"""


class DistributedLengthGroupedSampler(DistributedSampler):
    """
    https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_pt_utils.py#L668
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        mega_batch_mult: Optional[int] = None,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
        shuffle: bool = False,
        drop_last: bool = False,
        enable_length_group_by_modality: bool = False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )

        self.dataset = dataset
        self.batch_size = batch_size
        self.mega_batch_mult = mega_batch_mult
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.enable_length_group_by_modality = enable_length_group_by_modality

        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)

        self.epoch = 0
        self.total_size = self.num_samples * self.num_replicas
        self.lengths = self.dataset.lengths

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        if self.enable_length_group_by_modality:
            indices = get_modality_length_grouped_indices(
                self.lengths,
                self.batch_size,
                mega_batch_mult=self.mega_batch_mult,
                generator=g,
                shuffle=self.shuffle,
            )
        else:
            indices = get_length_grouped_indices(
                self.lengths,
                self.batch_size,
                mega_batch_mult=self.mega_batch_mult,
                generator=g,
                shuffle=self.shuffle,
            )

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            indices += indices[: (self.total_size - len(indices))]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int):
        self.epoch = epoch


def split_to_even_chunks(indices, lengths, num_chunks):
    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_length_grouped_indices(
    lengths,
    batch_size,
    mega_batch_mult=None,
    generator=None,
    shuffle=False,
):
    if mega_batch_mult is None:
        mega_batch_mult = min(len(lengths) // (batch_size * 4), 64)

    if shuffle:
        indices = torch.randperm(len(lengths), generator=generator)
    else:
        indices = torch.arange(len(lengths))

    megabatch_size = mega_batch_mult * batch_size
    megabatches = [
        indices[i : i + megabatch_size].tolist()
        for i in range(0, len(lengths), megabatch_size)
    ]
    megabatches = [
        sorted(megabatch, key=lambda i: lengths[i], reverse=True)
        for megabatch in megabatches
    ]

    # ensure the longest sample is in the first megabatch
    megabatch_maximums = [lengths[megabatch[0]] for megabatch in megabatches]
    max_idx = torch.argmax(torch.tensor(megabatch_maximums)).item()
    megabatches[0][0], megabatches[max_idx][0] = (
        megabatches[max_idx][0],
        megabatches[0][0],
    )

    # split to even chunks following llava
    megabatches = [
        split_to_even_chunks(megabatch, lengths, mega_batch_mult)
        for megabatch in megabatches
    ]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


def get_modality_length_grouped_indices(
    lengths,
    batch_size,
    mega_batch_mult=None,
    generator=None,
    shuffle=False,
):
    # edge case
    assert all(l != 0 for l in lengths)
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        mod = "image" if all(l > 0 for l in lengths) else "text"
        print(
            f"all samples are of the same modality: {mod}, "
            f"falling back to `get_length_grouped_indices`."
        )
        return get_length_grouped_indices(
            lengths,
            batch_size,
            mega_batch_mult=mega_batch_mult,
            generator=generator,
            shuffle=shuffle,
        )

    if mega_batch_mult is None:
        mega_batch_mult = min(len(lengths) // (batch_size * 4), 64)

    img_indices, img_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    txt_indices, txt_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l < 0])

    img_indices = [
        img_indices[i]
        for i in get_length_grouped_indices(
            img_lengths,
            batch_size,
            mega_batch_mult=mega_batch_mult,
            generator=generator,
            shuffle=shuffle,
        )
    ]
    txt_indices = [
        txt_indices[i]
        for i in get_length_grouped_indices(
            txt_lengths,
            batch_size,
            mega_batch_mult=mega_batch_mult,
            generator=generator,
            shuffle=shuffle,
        )
    ]

    megabatch_size = mega_batch_mult * batch_size

    img_megabatches = [
        img_indices[i : i + megabatch_size]
        for i in range(0, len(img_indices), megabatch_size)
    ]
    txt_megabatches = [
        txt_indices[i : i + megabatch_size]
        for i in range(0, len(txt_indices), megabatch_size)
    ]

    last_img_megabatch = img_megabatches[-1]
    last_txt_megabatch = txt_megabatches[-1]
    last_megabatch = last_img_megabatch + last_txt_megabatch

    megabatches = img_megabatches[:-1] + txt_megabatches[:-1]
    if shuffle:
        megabatches_indices = torch.randperm(len(megabatches), generator=generator)
    else:
        megabatches_indices = torch.arange(len(megabatches))
    megabatches = [megabatches[i] for i in megabatches_indices]

    if len(last_megabatch) > 0:
        megabatches.append(sorted(last_megabatch))

    # ensure the longest sample is in the first megabatch
    megabatch_maximums = [lengths[megabatch[0]] for megabatch in megabatches]
    max_idx = torch.argmax(torch.tensor(megabatch_maximums)).item()
    megabatches[0][0], megabatches[max_idx][0] = (
        megabatches[max_idx][0],
        megabatches[0][0],
    )

    return [i for megabatch in megabatches for i in megabatch]


"""
utils borrowed from llava-next
"""


def expand2square(pil_img, bg_value):
    w, h = pil_img.size
    if w == h:
        return pil_img
    elif w > h:
        x = Image.new(pil_img.mode, (w, w), bg_value)
        x.paste(pil_img, (0, (w - h) // 2))
        return x
    else:
        x = Image.new(pil_img.mode, (h, h), bg_value)
        x.paste(pil_img, ((h - w) // 2, 0))
        return x


def process_anyres_image(image, processor, grid_pinpoints, bg_value=None):
    """
    Process an image with variable resolutions.

    Args:
        image (PIL.Image.Image): The input image to be processed.
        processor: The image processor object.
        grid_pinpoints (str): A string representation of a list of possible resolutions.

    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """
    # convert grid_pinpoints from string to list
    if isinstance(grid_pinpoints, str) and "x" in grid_pinpoints:
        try:
            patch_size = processor.size[0]
        except Exception as e:
            patch_size = processor.size["shortest_edge"]
        assert patch_size in [
            224,
            336,
            384,
            448,
            512,
        ], "patch_size should be in [224, 336, 384, 448, 512]"
        # use regex to extract the range from the input string
        matches = re.findall(r"\((\d+)x(\d+)\)", grid_pinpoints)
        range_start = tuple(map(int, matches[0]))
        range_end = tuple(map(int, matches[-1]))
        # generate a matrix of tuples from (range_start[0], range_start[1]) to (range_end[0], range_end[1])
        grid_pinpoints = [
            (i, j)
            for i in range(range_start[0], range_end[0] + 1)
            for j in range(range_start[1], range_end[1] + 1)
        ]
        # multiply all elements by patch_size
        grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]

    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)

    best_resolution = select_best_resolution(image.size, possible_resolutions)
    image_padded = resize_and_pad_image(image, best_resolution, bg_value)
    patches = divide_to_patches(image_padded, processor.crop_size["height"])

    if isinstance(processor.size, dict):
        shortest_edge = processor.size["shortest_edge"]
    else:
        shortest_edge = min(processor.size)

    # FIXME: this seems to be a bug that it resizes instead of pad.
    # but to keep it consistent with previous, i will keep it as it is
    image_original_resize = image.resize((shortest_edge, shortest_edge))
    # TODO: uncomment below to ablate with the padding
    # image_original_resize = expand2square(
    #     image, tuple(int(x * 255) for x in processor.image_mean)
    # ).resize((processor.size["shortest_edge"], processor.size["shortest_edge"]))

    image_patches = [image_original_resize] + patches
    image_patches = [
        processor.preprocess(image_patch, return_tensors="pt")["pixel_values"]
        for image_patch in image_patches
    ]
    return torch.cat(image_patches, dim=0)


def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for width, height in possible_resolutions:
        # calculate the downscaled size to keep the aspect ratio
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(
            original_height * scale
        )

        # calculate effective and wasted resolutions
        effective_resolution = min(
            downscaled_width * downscaled_height, original_width * original_height
        )
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution
            and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


def resize_and_pad_image(image, target_resolution, bg_value=(0, 0, 0)):
    """
    Resize and pad an image to a target resolution while maintaining aspect ratio.

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    # determine which dimension (width or height) to fill
    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        # width will be filled completely
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        # height will be filled completely
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # resize the image
    resized_image = image.resize((new_width, new_height))

    # create a new image with the target size and paste the resized image onto it
    new_image = Image.new("RGB", (target_width, target_height), bg_value)
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image


def divide_to_patches(image, patch_size):
    """
    Divides an image into patches of a specified size.

    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The size of each patch.

    Returns:
        list: A list of PIL.Image.Image objects representing the patches.
    """
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)

    return patches
