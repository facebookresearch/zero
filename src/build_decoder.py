# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from transformers import AutoTokenizer, AutoProcessor
from peft import LoraConfig, get_peft_model
from copy import deepcopy

from utils import print_model_numel


def load_hf_llama3(cfg):
    """
    load llama models from https://huggingface.co/meta-llama
    """
    from transformers.models.llama.modeling_llama import (
        LlamaForCausalLM,
        LlamaDecoderLayer,
    )

    if cfg.llama_model_version not in ["3.3", "3.2", "3.1", "3"]:
        raise ValueError(f"invalid llama model version: {cfg.llama_model_version}")

    if int(cfg.llama_model_scale) not in [1, 3, 8, 70]:
        raise ValueError(f"invalid llama model scale: {cfg.llama_model_scale}")

    model_name = (
        f"meta-llama"
        + "/"
        + f"Llama-{cfg.llama_model_version}-{cfg.llama_model_scale}B-Instruct"
    )
    print(f"loading hf model: {model_name}")

    # register hf model name for decoder
    # to be used in vllm or other places
    cfg.decoder_hf_model_name = model_name

    # sdpa | eager | flash_attention_2
    attn_implementation = "sdpa" if cfg.enable_sdpa else None

    if int(cfg.llama_model_scale) >= 70:
        # if "auto" does not work, set device_map to "balanced"
        # https://huggingface.co/docs/accelerate/en/concept_guides/big_model_inference#designing-a-device-map
        device_map = "balanced"
        is_giant_model = bool(1)
    else:
        device_map = cfg.device  # duplicate model across devices
        is_giant_model = bool(0)

    if cfg.fsdp_mode:
        device_map = None

    if (
        cfg.eval_enable_vllm
        and not cfg.force_to_single_world
        and "cuda" not in str(device_map)
    ):
        # NOTE: for running giant models using vllm,
        # we need to set device_map to None
        # since it needs to be launched using torchrun
        device_map = None

    print(f"device_map: {device_map}")

    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        use_cache=False if cfg.fsdp_mode else None,
        attn_implementation=attn_implementation,
        device_map=device_map,
        torch_dtype=cfg.ptdtype,
        cache_dir=cfg.hf_cache_dir,
    )
    devices = check_model_devices(model)
    if len(devices) == 1:
        model.to(cfg.ptdtype)

    print(f"llama model.config: {model.config}")
    print(f"llama model.device: {devices}")
    print(f"llama model.dtype: {model.dtype}")

    num_embd = model.config.hidden_size  # for adapter's output dim
    num_hidden_layers = model.config.num_hidden_layers

    # build surrogate model if needed
    model = build_surrogate_model(cfg, model, num_hidden_layers=num_hidden_layers)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cfg.hf_cache_dir,
        use_fast=False,
        padding_side="right",
    )
    print("llama tokenizer:", tokenizer)

    if cfg.enable_lora:
        print("applying LoRA to the model")

        # recipe from llama-recipies
        # ref: https://github.com/meta-llama/llama-cookbook/blob/main/src/llama_cookbook/configs/peft.py#L8-L15
        peft_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=cfg.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
            lora_dropout=0.05,
            inference_mode=cfg.inference_mode,
        )

        # # recipe from llava-13B-lora
        # # ref: https://huggingface.co/liuhaotian/llava-v1.5-13b-lora/blob/main/adapter_config.json
        # peft_config = LoraConfig(
        #     r=128,
        #     lora_alpha=256,
        #     target_modules=[
        #         "q_proj",
        #         "k_proj",
        #         "v_proj",
        #         "o_proj",
        #         "down_proj",
        #         "gate_proj",
        #         "up_proj",
        #     ],
        #     bias="none",
        #     lora_dropout=0.05,
        #     task_type="CAUSAL_LM",
        #     inference_mode=cfg.inference_mode,
        # )

        print(f"peft_config: {peft_config}")
        model = get_peft_model(model, peft_config)
        model.to(cfg.ptdtype)
        model.print_trainable_parameters()

    # this layer is used for fsdp training
    # which is wrapped by FSDP in trian.py
    FSDP_DECODER_LAYER = LlamaDecoderLayer

    return num_embd, model, tokenizer, FSDP_DECODER_LAYER, is_giant_model


def load_hf_qwen3(cfg):
    """
    load qwen3 models from https://huggingface.co/Qwen
    """
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3ForCausalLM,
        Qwen3DecoderLayer,
    )

    if cfg.qwen_model_version not in ["3", "2.5"]:
        raise ValueError(f"invalid qwen model version: {cfg.qwen_model_version}")

    if float(cfg.qwen_model_scale) not in [0.6, 1.7, 4, 8, 14, 32, 72]:
        raise ValueError(
            f"invalid qwen model scale: {cfg.qwen_model_scale} "
            f"with version: {cfg.qwen_model_version}"
        )

    # load instruct model in default
    model_name = f"Qwen" + "/" + f"Qwen{cfg.qwen_model_version}-{cfg.qwen_model_scale}B"
    if cfg.qwen_model_type == "base":
        model_name += "-Base"
    print(f"loading hf model: {model_name}")

    # register hf model name for decoder
    # to be used in vllm or other places
    cfg.decoder_hf_model_name = model_name

    # sdpa | eager | flash_attention_2
    attn_implementation = "sdpa" if cfg.enable_sdpa else None

    if float(cfg.qwen_model_scale) >= 32:
        # if "auto" does not work, set device_map to "balanced"
        # https://huggingface.co/docs/accelerate/en/concept_guides/big_model_inference#designing-a-device-map
        device_map = "balanced"
        is_giant_model = bool(1)
    else:
        device_map = cfg.device  # duplicate model across devices
        is_giant_model = bool(0)

    if cfg.fsdp_mode:
        device_map = None

    if (
        cfg.eval_enable_vllm
        and not cfg.force_to_single_world
        and "cuda" not in str(device_map)
    ):
        # NOTE: for running giant models using vllm,
        # we need to set device_map to None
        # since it needs to be launched using torchrun
        device_map = None

    print(f"device_map: {device_map}")

    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py
    model = Qwen3ForCausalLM.from_pretrained(
        model_name,
        use_cache=False if cfg.fsdp_mode else None,
        attn_implementation=attn_implementation,
        device_map=device_map,
        torch_dtype=cfg.ptdtype,
        cache_dir=cfg.hf_cache_dir,
    )
    devices = check_model_devices(model)
    if len(devices) == 1:
        model.to(cfg.ptdtype)

    print(f"qwen model.config: {model.config}")
    print(f"qwen model.device: {devices}")
    print(f"qwen model.dtype: {model.dtype}")

    num_embd = model.config.hidden_size  # for adapter's output dim
    num_hidden_layers = model.config.num_hidden_layers

    # build surrogate model if needed
    model = build_surrogate_model(cfg, model, num_hidden_layers=num_hidden_layers)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name.replace("-Base", ""),
        cache_dir=cfg.hf_cache_dir,
        use_fast=False,
        padding_side="right",
    )
    print("qwen tokenizer:", tokenizer)

    if cfg.enable_lora:
        print("applying LoRA to the model")

        # recipe from llama-recipies
        # ref: https://github.com/meta-llama/llama-cookbook/blob/main/src/llama_cookbook/configs/peft.py#L8-L15
        peft_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=cfg.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
            lora_dropout=0.05,
            inference_mode=cfg.inference_mode,
        )

        # # recipe from llava-13B-lora
        # # ref: https://huggingface.co/liuhaotian/llava-v1.5-13b-lora/blob/main/adapter_config.json
        # peft_config = LoraConfig(
        #     r=128,
        #     lora_alpha=256,
        #     target_modules=[
        #         "q_proj",
        #         "k_proj",
        #         "v_proj",
        #         "o_proj",
        #         "down_proj",
        #         "gate_proj",
        #         "up_proj",
        #     ],
        #     bias="none",
        #     lora_dropout=0.05,
        #     task_type="CAUSAL_LM",
        #     inference_mode=cfg.inference_mode,
        # )

        print(f"peft_config: {peft_config}")
        model = get_peft_model(model, peft_config)
        model.to(cfg.ptdtype)
        model.print_trainable_parameters()

    # this layer is used for fsdp training
    # which is wrapped by FSDP in trian.py
    FSDP_DECODER_LAYER = Qwen3DecoderLayer

    return num_embd, model, tokenizer, FSDP_DECODER_LAYER, is_giant_model


"""
helper functions for building surrogate model by inserting translators
"""


def build_surrogate_model(cfg, model, model_attr="model", num_hidden_layers=-1):
    """
    build surrogate model by inserting translators
    """
    model_name = cfg.decoder_hf_model_name

    if cfg.enable_translator:
        print("start converting to surrogate model")
        print_model_numel(model, model_name="[orig] " + model_name)

        translators_range = deepcopy(cfg.translators_range)
        for range_idx, trans_range in enumerate(translators_range):
            trans_range.sort()

            assert (
                np.sum(np.diff(trans_range) == 1) == len(trans_range) - 1
            ), "translators_range should be consecutive without skipping layers"
            assert np.all(np.array(trans_range) >= 0)
            assert 0 <= min(trans_range) < max(trans_range) < num_hidden_layers, (
                "translators_range should be in the range of "
                f"[0, {num_hidden_layers})"
            )
            assert cfg.num_translators <= len(trans_range), (
                "num_translators should be less than or equal to "
                f"the number of layers in translators_range: {len(trans_range)}"
            )

            index_offset = 0
            if range_idx > 0:
                index_offset += sum(len(translators_range[i]) for i in range(range_idx))

            # rename the layer to translator
            for i in range(cfg.num_translators):
                translator_layer_id = trans_range[i]
                print(
                    f"- mv layer {translator_layer_id:>3} to translator "
                    f"-- actual layer {translator_layer_id - index_offset:>3}"
                )
                model = rename_layer_to_translator(
                    model,
                    translator_layer_id - index_offset,
                    model_attr=model_attr,
                )
                trans_range.pop(i)

            # remove the rest of the layers
            _empty_str = "".join([" "] * 14)
            for layer_id in trans_range:
                print(
                    f"- rm layer {layer_id:>3} {_empty_str}"
                    f"-- actual layer {layer_id - index_offset:>3}"
                )
                model = remove_layer_from_model(
                    model,
                    layer_id - index_offset,
                    model_attr=model_attr,
                )
                index_offset += 1

        print_model_numel(model, model_name="[srgt] " + model_name)
    else:
        print_model_numel(model, model_name=model_name)
    return model


def rename_layer_to_translator(
    model: torch.nn.Module, layer_id: int, model_attr="model"
):
    """
    after this, the original layer name will be changed from
    LlamaDecoderLayer to Translator
    """
    # create a new layer that inherits from the original layer
    Translator = type(
        "Translator", (type(getattr(model, model_attr).layers[layer_id]),), {}
    )

    # create a new instance with the same parameters but new class
    layer = getattr(model, model_attr).layers[layer_id]
    translator = Translator.__new__(Translator)
    translator.__dict__.update(layer.__dict__)

    # replace the layer in the model
    getattr(model, model_attr).layers[layer_id] = translator
    return model


def remove_layer_from_model(model: torch.nn.Module, layer_id: int, model_attr="model"):
    getattr(model, model_attr).layers.pop(layer_id)
    return model


def check_model_devices(model):
    # mainly for cpu offloading case with auto device_map
    devices = set()
    for param in model.parameters():
        devices.add(param.device.type)
    print(f"model devices: {devices}")
    return devices
