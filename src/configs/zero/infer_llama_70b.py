# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
import os.path as osp

from configs.config_base import config
from configs.zero.train_llama import func_load_decoder, func_load_encoder

cfg = copy.deepcopy(config)  # unwrap to override
# -----------------------------------------------

# overwrite cfg
cfg.llama_model_scale = "70"
cfg.llama_model_version = "3.1"

cfg.enable_translator = bool(1)
cfg.translators_range = (list(range(40, 79)),)

# NOTE: this is for loading giant models for inference and evaluation
cfg.force_to_single_world = bool(1)

# =========== [ cfg for evaluation ] ============
EVAL_LANGUAGE_MODEL = bool(0)  # set to 1 for evaluating language model

cfg.exp_code = "llama3.1-70b_surrogate-trained-encoder"
cfg.eval_batch_size = 1

cfg.eval_ckpt_path = [
    osp.join(
        cfg.ckpt_root,
        "llama3.1-70b_surrogate-trained-encoder",
        "ckpt_01_0004880.pth",
    ),
]
if cfg.enable_translator:
    cfg.load_pretrained_path = osp.join(
        cfg.ckpt_root,
        "llama3.1-70b_adapter_translator",
        "ckpt_01_0004393.pth",
    )
    cfg.eval_ckpt_path.insert(0, cfg.load_pretrained_path)

cfg.mute_non_master_ranks = bool(1)
cfg.eval_print_info = bool(1)
cfg.eval_batch_size = 1
cfg.enable_image_prompt = bool(1)

cfg.eval_enable_thinking_mode = bool(0)
cfg.eval_enable_vllm = bool(0)
cfg.eval_vllm_gpu_memory_utilization = 1.0
cfg.eval_vllm_tensor_parallel_size = 2
cfg.eval_vllm_max_model_len = 4096 * 8

if cfg.eval_enable_vllm:
    cfg.force_to_single_world = bool(0)

if EVAL_LANGUAGE_MODEL:
    cfg.ignore_nonzero_unexpected_keys = bool(1)
else:
    cfg.ignore_nonzero_unexpected_keys = bool(0)
cfg.eval_save_original_results = bool(0)
cfg.eval_save_images = bool(0)
cfg.eval_save_folder = f"tmp/BENCHMARK_NAME/testdev_images"

cfg.num_fewshot = None
cfg.max_gen_len = 16
cfg.eval_force_to_set_max_gen_len = bool(0)
cfg.enable_kv_cache = bool(1)
cfg.inference_mode = bool(1)

if cfg.enable_translator:
    if EVAL_LANGUAGE_MODEL:
        _save_fn = f"results_srgt_lang_NUM_FEWSHOT-shot.json"
    else:
        _save_fn = f"results_{cfg.num_fewshot}-shot.json"
else:
    if EVAL_LANGUAGE_MODEL:
        _save_fn = f"results_lang_NUM_FEWSHOT-shot.json"
    else:
        _save_fn = f"results_zero-shot-transfer_{cfg.num_fewshot}-shot.json"

if len(cfg.eval_ckpt_path) > 0:
    _model_name = cfg.eval_ckpt_path[-1].split("/")[-2]
    _model_ckpt = (
        cfg.eval_ckpt_path[-1].split("/")[-1].replace("ckpt_", "").replace(".pth", "")
    )
else:
    _model_name = cfg.exp_code
    _model_ckpt = "original"

_save_fn = _save_fn.replace(".json", f"_{_model_name}_{_model_ckpt}.json")

cfg.eval_result_file_path = osp.join(cfg.result_dir, cfg.exp_code, _save_fn)
print(f"cfg.eval_result_file_path: {cfg.eval_result_file_path}")


# -----------------------------------------------
for k, v in cfg.__dict__.items():
    if k not in config.__dict__:
        raise ValueError(f"config '{k}' is not registered in the base config")
config = cfg  # wrap back

# =========== [ encoder and decoder ] ===========
func_load_encoder = func_load_encoder
func_load_decoder = func_load_decoder


# ====== [ func to load generator class ] =======
def func_load_gen_cls():
    from models.generator_llama import Llama3EvalLM, Llama3EvalVLM

    return Llama3EvalLM if EVAL_LANGUAGE_MODEL else Llama3EvalVLM


# ====== [ func to load benchmarks ] ============
def func_load_benchmark():
    if EVAL_LANGUAGE_MODEL:
        """
        for language benchmarks
        """
        # format: [benchmark, num_fewshot]
        return [
            # ["mmlu", 5],
            # ["hellaswag", 10],
            # ["arc_easy", 0],
            ["arc_challenge", 25],
            # ["winogrande", 5],
            # ["piqa", 0],
            # ["boolq", 0],
            # ["openbookqa", 0],
        ]
    else:
        """
        for vision-language benchmarks
        """
        return [
            # "mmbench_en_dev",
            # "mme_binary",
            # "mme",
            # "pope_adv_binary",
            # "pope_pop_binary",
            # "pope_random_binary",
            # "pope_adv",
            # "pope_pop",
            # "pope_random",
            # # "seedbench_lite",
            # "llava_in_the_wild",
            # "mmvet",
            # "vizwiz_vqa_val",
            # "gqa",
            # "cvbench",
            # "seedbench",
            "textvqa_val",
            # "docvqa_val",
            # "chartqa",
            # "infovqa_val",
            # "ai2d",
        ]
