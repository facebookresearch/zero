#!/usr/bin/env python3

from typing import List

import argparse
import os
import json
import random
from collections import defaultdict
from datasets import load_dataset
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--hf-cache-dir", type=str, default=None)
parser.add_argument("--llava-665k-path", type=str, default=None)
parser.add_argument("--save-fold", type=str, default=None)
args = parser.parse_args()

"""
GenQA 500k
https://huggingface.co/datasets/tomg-group-umd/GenQA
"""


def load_genqa_500k(args) -> List[dict]:
    print("loading genqa dataset")
    genqa = load_dataset(
        "tomg-group-umd/GenQA_rebalanced",
        cache_dir=args.hf_cache_dir,
    )["train"]
    print(f"genqa total {genqa.num_rows} samples")

    genqa = genqa.shuffle(seed=42)
    genqa = genqa.shuffle(seed=0)
    genqa = genqa.shuffle(seed=42)

    sample_cnt = 0
    genqa_samples = []
    pbar = tqdm(enumerate(genqa), total=int(genqa.num_rows))
    for i, d in pbar:
        if random.random() < 0.3:
            continue

        messages = d["messages"]

        # conver the format to the same as llava
        convs = []
        for m in messages:
            if m["role"] == "user":
                convs.append({"from": "human", "value": m["content"]})
            elif m["role"] == "assistant":
                convs.append({"from": "gpt", "value": m["content"]})
            else:
                raise ValueError(f"unknown role: {m['role']}")

        d = {
            "id": str(i).zfill(7),
            "conversations": convs,
            "ds": "genqa-random-500k",
        }
        genqa_samples.append(d)
        sample_cnt += 1
        pbar.set_description(f"sample_cnt: {sample_cnt}")

        if sample_cnt == 500000:
            break

        if i == 10:
            # break
            pass

    return genqa_samples


"""
llava mix 665k
https://huggingface.co/datasets/kaiyuyue/llava-1.5-665k-instructions
"""


def load_llava_mix_665k(args) -> List[dict]:
    file_path = args.llava_665k_path

    fold_to_images = defaultdict(list)
    with open(file_path, "r") as f:
        data = json.load(f)

        for d in data:
            if "image" in d:
                fold = d["image"].split("/")[0]
                d["ds"] = "llava-mix-665k"
                fold_to_images[fold].append(d)

    for fold, images in fold_to_images.items():
        random.shuffle(images)
        random.shuffle(images)
        random.shuffle(images)

        fold_to_images[fold] = images
        # print(f"{fold}: {len(images)}")

    return fold_to_images


"""
mix the two datasets
"""

dst = load_genqa_500k(args)
src = load_llava_mix_665k(args)

random.shuffle(dst)
src["genqa"] = dst
sorted_folds = sorted(src.items(), key=lambda x: len(x[1]), reverse=True)

for k, v in sorted_folds:
    print(f"{k}: {len(v)}")

# balance the dataset
total_size = sum([len(images) for images in src.values()])
proportions = [len(images) / total_size for images in src.values()]

iters = [iter(images) for images in src.values()]
new_data = []
indices = [0] * len(iters)  # current index for each fold

print("mixing ...")
for i in tqdm(range(total_size)):
    strides = [ind / prop for ind, prop in zip(indices, proportions)]
    fold_idx = strides.index(min(strides))
    try:
        d = next(iters[fold_idx])
        d["index"] = i
        new_data.append(d)
        indices[fold_idx] += 1
    except StopIteration:
        print(f"fold {fold_idx} is exhausted")
        proportions[fold_idx] = float("inf")


print(len(new_data))
save_path = os.path.join(args.save_fold, "llava-665k_genqa-500k_shuffled.json")
with open(save_path, "w") as f:
    json.dump(new_data, f, indent=2)
print(f"saved to {save_path}")
