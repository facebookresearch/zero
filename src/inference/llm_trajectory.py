# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
usage:
    python3 inference/llm_trajectory.py configs/config_base.py
"""

import sys
import os
import random
import numpy as np
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
import transformers

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from tqdm import tqdm
from utils import (
    load_config,
    setup_model_parallel,
    setup_for_distributed,
    print_model_numel,
    format_metric_to_gb,
    set_dtype,
    set_seed,
)

from loader import build_hf_dataloader


# ====== global settings ======
MODEL_FAMILY = "llama"  # llama | gemma | qwen

# if True, use torchrun to launch
# if False, use python3 to launch
LOAD_BIG_MODEL = bool(1)  # default: True

# for plotting
color = {
    "llama": "#E57439",
    "gemma": "#479A5F",
    "qwen": "#4F7B9D",
}[MODEL_FAMILY]
# =============================


def main(cfg):
    # load function before overwriting cfg
    cfg = cfg.config
    cfg.batch_size = cfg.eval_batch_size

    # overwrite cfg
    cfg.dtype = "bfloat16"  # float16, bfloat16 or float32
    cfg.force_to_use_fp16 = bool(0)

    # overwrite hf cache dir
    # cfg.hf_cache_dir = "none"

    # hyper params
    cfg.eval_ckpt_path = []
    cfg.enable_kv_cache = bool(1)
    cfg.enable_sdpa = bool(1)

    # init ranks
    local_rank, global_rank, world_size, device = setup_model_parallel(
        mute_non_master_ranks=cfg.mute_non_master_ranks,
        force_to_single_world=LOAD_BIG_MODEL,
    )
    cfg.local_rank = local_rank
    cfg.global_rank = global_rank
    cfg.world_size = world_size
    cfg.device = device
    master_process = global_rank == 0
    print(f"hyper params:")
    for k, v in cfg.__dict__.items():
        print(f"- {k}: {v}")

    # set print with timestamp
    if cfg.mute_non_master_ranks:
        setup_for_distributed(master_process)

    # set seed
    cfg.seed = random.randint(0, 10000)
    set_seed(cfg.seed)

    # set tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # gpu memory
    device_properties = torch.cuda.get_device_properties(device)
    gpu_mem = format_metric_to_gb(device_properties.total_memory)
    print(f"gpu memory: {gpu_mem:.2f}GB")

    # set dtype
    cfg = set_dtype(cfg)

    # build decoder
    if MODEL_FAMILY == "llama":
        model_name = (
            "meta-llama/Llama-3.2-3B-Instruct"
            # "meta-llama/Llama-3.1-8B-Instruct"
            # "meta-llama/Llama-3.1-70B-Instruct"
        )
    elif MODEL_FAMILY == "gemma":
        model_name = (
            # "google/gemma-2-2b-it"
            # "google/gemma-3-1b-it"
            # "google/gemma-3-4b-it"
            "google/gemma-3-12b-it"
            # "google/gemma-3-27b-it"
        )
    elif MODEL_FAMILY == "qwen":
        model_name = (
            "Qwen/Qwen2.5-0.5B-Instruct"
            # "Qwen/Qwen2.5-1.5B-Instruct"
            # "Qwen/Qwen2.5-3B-Instruct"
            # "Qwen/Qwen2.5-7B-Instruct"
            # "Qwen/Qwen2.5-14B-Instruct"
            # "Qwen/Qwen2.5-32B-Instruct"
            # "Qwen/Qwen2.5-72B-Instruct"
            # "Qwen/Qwen3-0.6B"
            # "Qwen/Qwen3-1.7B"
            # "Qwen/Qwen3-4B"
            # "Qwen/Qwen3-8B"
            # "Qwen/Qwen3-14B"
            # "Qwen/Qwen3-32B"
        )
    else:
        raise ValueError(f"invalid model family: {MODEL_FAMILY}")

    attn_impl = "sdpa" if cfg.enable_sdpa else None
    if MODEL_FAMILY == "qwen" and "Qwen2.5-" in model_name:
        # to avoid sliding window attention warning
        # even though it is not used in default
        attn_impl = "flash_attention_2"

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="balanced" if LOAD_BIG_MODEL else cfg.device,
        attn_implementation=attn_impl,
        torch_dtype=cfg.ptdtype,
        cache_dir=cfg.hf_cache_dir,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cfg.hf_cache_dir,
    )
    print(model.config)

    # clean chat template
    if MODEL_FAMILY == "llama":
        chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|start_header_id|>' + message['role'] + '<|end_header_id|>' + '\n' + message['content'] + '<|eot_id|>'}}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n' }}{% endif %}"
    elif MODEL_FAMILY == "gemma":
        chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<start_of_turn>' + message['role'] + '\n' + message['content'] + '<end_of_turn>'}}{% endfor %}{% if add_generation_prompt %}{{ '<start_of_turn>model\n' }}{% endif %}"
    elif MODEL_FAMILY == "qwen":
        chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    else:
        print("might need to consider removing system prompt from chat template")
        chat_template = None

    # overwrite chat template
    if chat_template is not None:
        tokenizer.chat_template = chat_template

    # build adapter
    model.eval()
    model.requires_grad_(False)
    print_model_numel(model, model_name=model_name)
    print(model)

    # get language model
    if (
        MODEL_FAMILY == "gemma"
        and "3-1b" not in model_name
        and "2-2b" not in model_name
    ):
        language_model = model.language_model
    else:
        language_model = model

    # get norm and head layer
    lm_norm = language_model.model.norm
    lm_head = language_model.lm_head

    # load genqa loader
    cfg.data_name = "genqa"
    loader, sampler = build_hf_dataloader(
        cfg,
        global_rank,
        world_size,
        is_train=True,  # hack to shuffle
    )

    """
    plotting trajectory
    """

    # init canvas
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    probs_hidden_states_out = []
    num_samples_to_plot = 300

    sample_cnt = -1
    random_pick = bool(0)
    pbar = tqdm(enumerate(loader), desc="plotting trajectory")
    for i, sample in pbar:
        if random.random() > 0.3 and random_pick:
            continue

        sample_cnt += 1

        # apply template
        if MODEL_FAMILY in ["llama", "qwen"]:
            usr_role, ast_role = "user", "assistant"

        elif MODEL_FAMILY == "gemma":
            usr_role, ast_role = "user", "model"

        sample = [d for d in sample[0]["messages"] if d["content"] != None]
        _sample = []
        for s in sample:
            if s["role"] == "user":
                _sample.append({"role": usr_role, "content": s["content"].strip()})
            else:
                _sample.append({"role": ast_role, "content": s["content"].strip()})
        sample = _sample

        # forward
        text = tokenizer.apply_chat_template(
            sample,
            add_generation_prompt=False,
            tokenize=False,
        )
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=False,
        ).to(device)
        token_ids = inputs["input_ids"]

        # check
        # print()
        # print(i, tokenizer.decode(token_ids[0]))
        # exit(0)

        # all intermediate states will be cached in debugging_tensors
        output = model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = output["hidden_states"][1:]  # remove input embeddings

        # recycle memory
        torch.cuda.empty_cache()

        # plot trajectory
        ps_out = []
        last_x = hidden_states[-1]
        target_out = token_ids[:, 1:]

        # NOTE: the last final hidden state is already normalized
        # when going through the forward pass, so no need to norm
        # last_x = lm_norm(last_x)

        last_x = lm_head(last_x)
        last_x = last_x[:, :-1, :]
        last_p = torch.softmax(last_x, dim=-1)

        last_p = last_p.to("cuda")
        target_out = target_out.to("cuda")

        last_p = last_p.gather(dim=2, index=target_out.unsqueeze(-1)).squeeze(-1)
        last_p_norm = last_p / last_p.sum(dim=-1, keepdim=True)
        last_p_norm = last_p_norm.float()

        epsilon = 1e-6

        for i, v in enumerate(hidden_states):
            """
            shift ratio from each layer to the output space
            """

            if i != len(hidden_states) - 1:
                x_out = lm_norm(v.clone())
            else:
                x_out = v.clone()
            x_out = lm_head(x_out)
            x_out = x_out[:, :-1, :]  # [bs, seq_len - 1, vocab_size]
            p_out = torch.softmax(x_out, dim=-1)

            p_out = p_out.to("cuda")
            p_out = p_out.gather(dim=2, index=target_out.unsqueeze(-1)).squeeze(-1)
            p_out_norm = p_out / p_out.sum(dim=-1, keepdim=True)
            p_out_norm = p_out_norm.float()

            # avoid log(0)
            last_p_norm = torch.clamp(last_p_norm, min=epsilon)
            p_out_norm = torch.clamp(p_out_norm, min=epsilon)

            # compute kl divergence
            kl_out = p_out_norm * (p_out_norm / last_p_norm).log()
            kl_out = kl_out.sum(dim=-1)

            ps_out.append(kl_out.item())

        probs_hidden_states_out.append(ps_out)

        if sample_cnt == num_samples_to_plot:
            break

    # clean up
    dist.destroy_process_group()

    # plot trajectory
    probs_hidden_states_out = np.array(
        probs_hidden_states_out
    )  # [num_samples, num_layers]

    for i in range(num_samples_to_plot):
        ax.plot(probs_hidden_states_out[i], alpha=0.1, color=color)

    ax.grid(color="gray", linestyle=":", linewidth=0.1)
    ax.set_xlim(0, probs_hidden_states_out.shape[1] - 1)
    ax.set_ylim(0, 12.5)
    model_name = model_name.replace("google/", "")
    ax.set_title(f"trajectory of {model_name}", fontsize=14)
    ax.set_xlabel("layers", fontsize=14)
    ax.set_ylabel("kl distance", fontsize=14)
    plt.tight_layout()

    # save trajectory
    save_fold = "saved_plots"
    os.makedirs(save_fold, exist_ok=True)

    model_name = model_name.replace("/", "_")
    path = f"{save_fold}/trajetory_{model_name}"
    plt.savefig(f"{path}.pdf", bbox_inches="tight")
    np.save(f"{path}.npy", probs_hidden_states_out)


if __name__ == "__main__":
    cfg = load_config(sys.argv)
    main(cfg)
