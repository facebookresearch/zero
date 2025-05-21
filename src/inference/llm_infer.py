# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
usage:
    ./scripts/run inference/llm_infer.py configs/zero/infer_llama_3b.py
"""

import sys
import os
import torch
import warnings

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from copy import deepcopy
from torch.distributed import barrier
from utils import (
    load_config,
    setup_model_parallel,
    print_model_numel,
    format_metric_to_gb,
    set_dtype,
    set_seed,
    load_checkpoint,
)
from build_encoder import load_clip
from build_decoder import load_hf_llama3 as load_decoder
from models.adapter import Zero
from inference.checkpoints import BASE_REPO_URL, HF_HUB_LINKS

# suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch*")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers*")


def main(cfg):
    # load function before overwriting cfg
    cfg = cfg.config
    cfg.batch_size = cfg.eval_batch_size

    # overwrite cfg
    cfg.dtype = "bfloat16"  # float16, bfloat16 or float32
    cfg.force_to_use_fp16 = bool(0)

    cfg.enable_translator = bool(1)
    cfg.enable_kv_cache = bool(1)

    # init ranks
    local_rank, global_rank, world_size, device = setup_model_parallel(
        mute_non_master_ranks=cfg.mute_non_master_ranks,
        force_to_single_world=cfg.force_to_single_world,
    )
    cfg.local_rank = local_rank
    cfg.global_rank = global_rank
    cfg.world_size = world_size
    cfg.device = device
    master_process = global_rank == 0
    print(f"hyper params:")
    for k, v in cfg.__dict__.items():
        print(f"- {k}: {v}")

    if cfg.llama_model_scale == "3":
        cfg.eval_ckpt_path = os.path.join(
            cfg.ckpt_root,
            "llama3.2-3b_adapter_translator",
            "ckpt_01_0003568.pth",
        )
    elif cfg.llama_model_scale == "8":
        cfg.eval_ckpt_path = os.path.join(
            cfg.ckpt_root,
            "llama3.1-8b_adapter_translator",
            "ckpt_01_0004393.pth",
        )
    elif cfg.llama_model_scale == "70":
        cfg.eval_ckpt_path = os.path.join(
            cfg.ckpt_root,
            "llama3.1-70b_adapter_translator",
            "ckpt_01_0004393.pth",
        )
    else:
        raise ValueError(f"unsupported llama model scale '{cfg.llama_model_scale}'")

    # download the encoder checkpoint
    if not os.path.exists(cfg.eval_ckpt_path):
        os.makedirs(os.path.dirname(cfg.eval_ckpt_path), exist_ok=True)
        url = BASE_REPO_URL + HF_HUB_LINKS[cfg.llama_model_scale]["translator"]
        print(f"downloading adapter and translator from {url} ...")
        if master_process:
            os.system(f"wget {url} -O {cfg.eval_ckpt_path}")
            print(": done")
    barrier()
    print(f"translator checkpoint path: {cfg.eval_ckpt_path}")

    # set seed
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

    # build encoder
    encoder = None

    # build decoder
    cfg.llama_model_name = f"{cfg.llama_model_scale}B-Instruct"
    cfg.llama_model_dir = os.path.join(
        cfg.ckpt_root,
        f"llama-{cfg.llama_model_version}",
        f"Llama-{cfg.llama_model_version}-{cfg.llama_model_name}",
    )

    # build decoder
    num_embd, decoder, tokenizer, _, _ = load_decoder(cfg)
    cfg.num_embd = num_embd

    # build adapter
    model = Zero(
        cfg=cfg,
        encoder=encoder,
        decoder=decoder,
    )
    model.to(cfg.ptdtype)
    model.eval()
    model.requires_grad_(False)
    print_model_numel(model, model_name="zero")
    print(model)

    # load checkpoint
    if isinstance(cfg.eval_ckpt_path, str):
        cfg.eval_ckpt_path = [cfg.eval_ckpt_path]

    if cfg.enable_translator:
        # load surrogate translator
        for ckpt_path in cfg.eval_ckpt_path:
            load_checkpoint(
                ckpt_path,
                model,
                strict=False,
                verbose=True,
                ignore_nonzero_unexpected_keys=True,
                force_to_use_raw_param_name=True,
            )

    # generation
    question = [
        {
            "role": "user",
            "content": "Explain E=mc^2 in simple terms.",
        },
    ]

    # clean chat template
    chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|start_header_id|>' + message['role'] + '<|end_header_id|>' + '\n' + message['content'] + '<|eot_id|>'}}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n' }}{% endif %}"
    tokenizer = deepcopy(tokenizer)
    tokenizer.chat_template = chat_template

    model.decoder.generation_config.pad_token_id = tokenizer.eos_token_id

    token_ids = tokenizer.apply_chat_template(
        question, add_generation_prompt=True
    )  # [bs, seq_len]
    print("input token ids:", token_ids)
    print("input text:", tokenizer.decode(token_ids))

    token_ids = torch.tensor([token_ids]).long().to(device)  # [bs, seq_len]
    input_len = token_ids.shape[1]

    # perform top-1 sampling
    output_token_ids = []  # bs = 1
    max_gen_toks = 4096

    with torch.inference_mode():
        attention_mask = torch.ones(1, input_len).to(device)
        output_token_ids = model.decoder.generate(
            input_ids=token_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=max_gen_toks,
            use_cache=cfg.enable_kv_cache,
        )
        output_token_ids = output_token_ids.cpu().numpy().tolist()  # [bs, seq_len]

    output_token_str = tokenizer.decode(output_token_ids[0])
    print("\n------- output -------\n")
    print(output_token_str)
    print("\n----------------------\n")


if __name__ == "__main__":
    cfg = load_config(sys.argv)
    main(cfg)
