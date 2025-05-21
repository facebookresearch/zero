"""
usage:
    ./scripts/run inference/vlm_infer.py configs/zero/infer_llama_3b.py
"""

from typing import List

import sys
import os
import torch
import warnings

from copy import deepcopy
from torch.distributed import barrier
from torch.nn.utils.rnn import pad_sequence
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

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
from loader import expand2square
from inference.checkpoints import BASE_REPO_URL, HF_HUB_LINKS
from methods._questions_prompt_for_caption import template_for_image_tokens

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

    cfg.enable_translator = bool(0)
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
        cfg.load_pretrained_path = os.path.join(
            cfg.ckpt_root,
            "llama3.2-3b_adapter_translator",
            "ckpt_01_0003568.pth",
        )
        cfg.eval_ckpt_path = os.path.join(
            cfg.ckpt_root,
            "llama3.2-3b_surrogate-trained-encoder",
            "ckpt_01_0004880.pth",
        )
    elif cfg.llama_model_scale == "8":
        cfg.load_pretrained_path = os.path.join(
            cfg.ckpt_root,
            "llama3.1-8b_adapter_translator",
            "ckpt_01_0004393.pth",
        )
        cfg.eval_ckpt_path = os.path.join(
            cfg.ckpt_root,
            "llama3.1-8b_surrogate-trained-encoder",
            "ckpt_01_0004880.pth",
        )
    elif cfg.llama_model_scale == "70":
        cfg.load_pretrained_path = os.path.join(
            cfg.ckpt_root,
            "llama3.1-70b_adapter_translator",
            "ckpt_01_0004393.pth",
        )
        cfg.eval_ckpt_path = os.path.join(
            cfg.ckpt_root,
            "llama3.1-70b_surrogate-trained-encoder",
            "ckpt_01_0004880.pth",
        )
    else:
        raise ValueError(f"unsupported llama model scale '{cfg.llama_model_scale}'")

    # download the encoder checkpoint
    for i, p in enumerate([cfg.load_pretrained_path, cfg.eval_ckpt_path]):
        if not os.path.exists(p):
            os.makedirs(os.path.dirname(p), exist_ok=True)
            checkpoint_type = "translator" if i == 0 else "encoder"
            url = BASE_REPO_URL + HF_HUB_LINKS[cfg.llama_model_scale][checkpoint_type]
            print(f"downloading encoder from {url} ...")
            if master_process:
                os.system(f"wget {url} -O {p}")
                print(": done")
    barrier()

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
    encoder, image_processor, image_input_size = load_clip(
        cfg=cfg, model_name=cfg.encoder_model_name
    )
    encoder = encoder.to(cfg.ptdtype)
    cfg.image_input_size = image_input_size

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
        cfg.eval_ckpt_path.insert(0, cfg.load_pretrained_path)

    for ckpt_path in cfg.eval_ckpt_path:
        load_checkpoint(
            ckpt_path,
            model,
            strict=False,
            verbose=True,
            ignore_nonzero_unexpected_keys=cfg.ignore_nonzero_unexpected_keys,
            force_to_use_raw_param_name=True,
        )

    if cfg.enable_lora:
        model.decoder.merge_and_unload()
        print("merge and unload LoRA weights")
        print(model)

    # input image and question
    question = ["What is the text in the image? And where is it?"]
    img_path = "../.github/shark.jpg"

    # preprocess
    imgs = Image.open(img_path).convert("RGB")
    imgs = expand2square(imgs, tuple(int(v * 255) for v in image_processor.image_mean))
    imgs = image_processor(imgs, return_tensors="pt")["pixel_values"]
    imgs = imgs.to(device)  # [1, 3, h, w]

    # prepare input embeddings
    embds_imgs: List[torch.Tensor] = model._forward([imgs])

    # tokenizer ids
    img_token_str = "<|reserved_special_token_0|>"
    img_id = tokenizer.convert_tokens_to_ids(img_token_str)
    bot_id = tokenizer.convert_tokens_to_ids("<|begin_of_text|>")

    # clean chat template
    chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|start_header_id|>' + message['role'] + '<|end_header_id|>' + '\n' + message['content'] + '<|eot_id|>'}}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n' }}{% endif %}"
    tokenizer = deepcopy(tokenizer)
    tokenizer.chat_template = chat_template

    model.decoder.generation_config.pad_token_id = tokenizer.eos_token_id
    wte = model.decoder.model.get_input_embeddings()

    # prepare input ids
    input_ids = []
    for q in question:
        ids = [bot_id]
        q = (
            template_for_image_tokens[0]
            + img_token_str
            + f", {q[0].lower()}{q[1:].strip()}"
        )
        ids += tokenizer.apply_chat_template(
            [{"role": "user", "content": q}],
            add_generation_prompt=True,
        )
        input_ids.append(ids)

    # prepare input embeddings
    input_embds = []
    for i, input_id in enumerate(input_ids):
        img_pad_idx = input_id.index(img_id)

        inp1_ids = input_id[:img_pad_idx]
        inp2_ids = input_id[img_pad_idx + 1 :]

        inp1_ids = torch.tensor(inp1_ids, dtype=torch.long, device=device)
        inp2_ids = torch.tensor(inp2_ids, dtype=torch.long, device=device)

        embds_inp1 = wte(inp1_ids)  # [seq_len, dim]
        embds_inp2 = wte(inp2_ids)  # [seq_len, dim]

        input_embds.append(
            torch.cat(
                [
                    embds_inp1,
                    embds_imgs[i],
                    embds_inp2,
                ],
                dim=0,
            )
        )

    # pad input embeddings
    input_embds = pad_sequence(input_embds, batch_first=True, padding_value=0.0)

    with torch.inference_mode():
        attention_mask = torch.ones(
            input_embds.shape[0],
            input_embds.shape[1],
        ).to(device)
        output_token_ids = model.decoder.generate(
            inputs_embeds=input_embds,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=1024,
            use_cache=True,
        )

    input_token_str = tokenizer.decode(input_ids[0])
    output_token_str = tokenizer.decode(output_token_ids[0])
    print("\n------- output -------\n")
    print(f"input: {input_token_str}")
    print(f"output: {output_token_str}")
    print("\n----------------------\n")


if __name__ == "__main__":
    cfg = load_config(sys.argv)
    main(cfg)
