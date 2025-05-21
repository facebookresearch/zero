from typing import List, Tuple, Optional

import os
import os.path as osp
import torch
import transformers

from dataclasses import dataclass, field


@dataclass
class ConfigBase:
    torch_version: str = torch.__version__
    transformers_version: str = transformers.__version__

    # hparams for directories
    dev_root: str = osp.join(os.environ["HOME"], "dev")
    proj_tag: str = "zero"

    data_root: str = osp.join(dev_root, proj_tag, "data")
    cache_root: str = osp.join(dev_root, proj_tag, "cache")
    ckpt_root: str = osp.join(dev_root, proj_tag, "checkpoints")

    # hparams for hf
    hf_cache_dir: str = osp.join(cache_root, "huggingface")

    # hparams for uploading checkpoints to hf
    ckpt_upload_to_hf: bool = bool(0)
    hf_repo_id: str = None
    hf_access_token: str = None

    # hparams for experiments
    exp_code: str = None
    data_name: Tuple[str] = ("none",)
    training_stage: str = None

    ckpt_dir: str = osp.join(ckpt_root)
    result_dir: str = osp.join(dev_root, proj_tag, "results")

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    # hparams for decoder
    decoder: str = "llama3"  # 'llama3' | "qwen"
    decoder_hf_model_name: str = None

    # hparams for methods
    enable_kv_cache: bool = bool(0)
    max_seq_len: int = 2048  # max length for input sequence
    max_gen_len: int = 4096  # max length for generation

    # hparams for input images
    image_ratio_type: str = "pad"  # pad | anyres
    anyres_grid_pinpoints: str = "(1x1),...,(6x6)"
    enable_image_aug: bool = bool(0)
    image_aug_rotation_degree: int = 0
    simple_raw_image_mode: bool = bool(0)

    # hparams for dataloader
    enable_dist_length_grouped_sampler: bool = bool(0)
    enable_sampler_shuffle: bool = bool(0)
    enable_length_group_by_modality: bool = bool(0)

    # hparams for input prompts
    enable_plain_concat_mode: bool = bool(0)
    shuffle_conversation_turns: bool = bool(0)
    enable_random_image_prompt: bool = bool(0)
    enable_image_prompt: bool = bool(0)
    enable_system_prompt: bool = bool(0)

    # hparams for model
    enable_attention_mask: bool = bool(0)
    adapter_mlp_depth: int = 2
    append_rmsnorm_in_adapter: bool = bool(0)

    # hparams for input instructions
    ignore_instruction_token_ids: bool = bool(0)
    include_special_tokens_in_masking_instruction_tokens: bool = bool(0)
    split_image_tokens_in_prompt_with_newline: bool = bool(0)

    enable_translator: bool = bool(0)
    translators_range: list = field(default_factory=lambda: list(range(1, 2)))
    num_translators: int = 1

    # hparams for surrogate llama-3
    hf_force_download: bool = bool(0)  # force download again for new checkpoints

    # hparams for llama-3
    llama_model_scale: str = "8"
    llama_model_version: str = "3"  # 3, 3.1, 3.2
    llama_model_name: str = f"{llama_model_scale}B-Instruct"  # 8B, 8B-Instruct
    llama_model_dir: str = osp.join(
        ckpt_root,
        f"llama-{llama_model_version}",
        f"Llama{llama_model_version}-{llama_model_name}",
    )

    # hparams for qwen3
    qwen_model_scale: str = "4b"
    qwen_model_version: str = "3"
    qwen_model_type: str = "instruct"  # "base" | "instruct"

    # hparams for clip
    encoder_model_name: str = "openai/clip-vit-large-patch14-336"
    input_img_resolution: int = 336
    input_img_patch_size: int = 14
    min_input_image_size: Optional[int] = None
    max_input_image_size: Optional[int] = None

    encoder_lr_scaler: float = 1.0
    encoder_ft_layers_after: int = 0
    encoder_shave_last_n_layers: int = 0
    ignore_encoder_cls_token: bool = bool(0)

    # hparams for training
    device: torch.device = None
    seed: int = 42
    batch_size: int = 1
    num_workers: int = 0
    pin_memory: bool = bool(0)
    mute_non_master_ranks: bool = bool(1)
    force_to_single_world: bool = bool(0)

    lr: float = 1e-3
    min_lr: float = 0.0
    wd: float = 0.0
    grad_clip: float = 0.0

    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.999

    reduce_batch_loss: bool = bool(1)

    enable_balance_loss_weights: bool = bool(0)
    balance_loss_exp_ord: float = 1.0  # exp ord for loss weight balancing
    balance_loss_exp_avg_norm: bool = bool(0)

    epochs: int = 1
    gradient_accumulation_steps: int = 1

    enable_warmup: bool = bool(1)
    warmup_ratio: float = 3  # % of total steps
    warmup_steps: int = 0
    dtype: str = "bfloat16"  # float16, bfloat16 or float32
    force_to_use_fp16: bool = bool(
        0
    )  # avoid switching to bf16 automatically when supported
    compile_model: bool = bool(0)

    ckpt_save_params_only: bool = bool(0)
    ckpt_save_interval_ratio: int = 10  # % : save ckpt every % of the total steps
    log_interval: int = 1
    force_to_use_raw_param_name: bool = bool(0)

    enable_debug_mode: bool = bool(0)

    # hprams for fsdp
    fsdp_mode: bool = bool(0)
    fsdp_activation_checkpointing: bool = bool(0)
    fsdp_cpu_offload: bool = bool(0)
    enable_sdpa: bool = bool(0)

    # hparams for peft
    enable_lora: bool = bool(0)
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # hparams for loading pretrained weights
    load_pretrained: bool = bool(0)
    load_pretrained_path: str = None

    # hparams for resuming
    resume: bool = bool(0)
    resume_ckpt_path: str = None
    from_scratch: bool = bool(1)  # if True, training from scratch with resumed weights

    ignore_nonzero_unexpected_keys: bool = bool(0)
    ignore_adapter_keys_pretrained: bool = bool(0)
    ignore_adapter_keys_resume: bool = bool(0)

    # hprams for evaluation
    num_fewshot: int = 0
    benchmark: str = "none"

    eval_print_info: bool = bool(0)
    eval_batch_size: int = 1
    eval_ckpt_path: str = None
    eval_data_root: str = None
    eval_result_file_path: str = None
    eval_force_to_set_max_gen_len: bool = bool(0)
    eval_save_original_results: bool = bool(0)
    eval_res_upload_to_hf: bool = bool(0)
    eval_enable_chat_template: bool = bool(0)

    eval_enable_thinking_mode: bool = bool(0)
    eval_enable_vllm: bool = bool(0)
    eval_vllm_gpu_memory_utilization: float = 0.9
    eval_vllm_tensor_parallel_size: int = 1
    eval_vllm_max_model_len: Optional[int] = None

    eval_save_images: bool = bool(0)
    eval_save_folder: str = None

    # hparams for inference
    inference_mode: bool = bool(0)
    input_img_path: str = None
    input_txt: List[str] = None

    # hparams for logging
    enable_wandb: bool = bool(0)
    wandb_project: str = None
    wandb_entity: str = None
    wandb_track_name: str = None
    wandb_key: str = None

    # fallback for attributes dynamically set by setattr
    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )


config = ConfigBase()
