import copy
import os.path as osp
from configs.config_base import config

# unwrap base config to override
cfg = copy.deepcopy(config)
# -----------------------------------------------


# ============= [ global config ] ===============
TOTAL_NUM_GPUS_PLAN_TO_USE = 4  # gpu cards
QWEN_MODEL_SCALE = "4"
QWEN_MODEL_VERSION = "3"
QWEN_MODEL_TYPE = "instruct"
IS_SURROGATE = bool(1)
TRAINING_STAGE = (
    "pretrain"
    # "finetune-decoder"
)


# ============= [ overwrite cfg ] ===============
# for doing sth. different according the training stage
cfg.training_stage = TRAINING_STAGE

cfg.decoder = "qwen"
cfg.qwen_model_scale = QWEN_MODEL_SCALE
cfg.qwen_model_version = QWEN_MODEL_VERSION
cfg.qwen_model_type = QWEN_MODEL_TYPE

# for inserting the translator in language model (decoder)
# cfg.translators_range tuple can have more than one range,
# for example: = (list(range(16, 27)), list(range(40, 79)))
# in this case, two translators will be inserted to replace those two ranged layers
cfg.enable_translator = IS_SURROGATE
if cfg.enable_translator:
    if cfg.qwen_model_scale == "4":
        cfg.translators_range = (list(range(22, 35)),)
    else:
        raise ValueError(f"unknown qwen model scale: {cfg.qwen_model_scale}")

# can be more than one dataset in the tuple
cfg.data_name = ("genqa",)  # from huggingface

total_batch_size = 256
cfg.batch_size = 4
cfg.gradient_accumulation_steps = (
    total_batch_size // cfg.batch_size // TOTAL_NUM_GPUS_PLAN_TO_USE
)
cfg.lr = 1e-4
cfg.epochs = 1

cfg.log_interval = 1
cfg.ckpt_save_interval_ratio = 25  # % : save ckpt every % of the total steps
cfg.dtype = "bfloat16"
cfg.seed = 1337
cfg.num_workers = 8
cfg.mute_non_master_ranks = bool(1)
cfg.enable_warmup = bool(1)
cfg.warmup_ratio = 3.05  # % of the total steps

cfg.min_lr = 0.0  # for cosine decay scheduler
cfg.wd = 0.0  # weight decay
cfg.grad_clip = 1.0
cfg.reduce_batch_loss = bool(1)
cfg.max_seq_len = 4096 * 1  # max input sequence length

cfg.fsdp_mode = bool(1)
cfg.fsdp_cpu_offload = bool(0)
cfg.fsdp_activation_checkpointing = bool(1)

cfg.enable_sdpa = bool(1)
cfg.enable_lora = bool(0)

# set False to save parameters and optimizer states
# set True to save only parameters
cfg.ckpt_save_params_only = bool(1)

# compute the loss only on the response tokens
cfg.ignore_instruction_token_ids = bool(1)
cfg.include_special_tokens_in_masking_instruction_tokens = bool(0)

# NOTE: using pretrained weights
cfg.load_pretrained = bool(0)
cfg.load_pretrained_path = osp.join(cfg.ckpt_dir, "none", "none")

# NOTE: resuming training
cfg.resume = bool(0)
cfg.from_scratch = bool(1)

# for saving checkpoints
training_stage = (
    TRAINING_STAGE.lower()
    .replace("encoder", "enc")
    .replace("decoder", "dec")
    .replace("pretrain", "pt")
    .replace("finetune", "ft")
)
is_surrogate = "srgt" if cfg.enable_translator else ""
cfg.exp_code = (
    f"lang_{cfg.decoder}{cfg.qwen_model_version}_{cfg.qwen_model_scale}b-{is_surrogate}"
    f"_{training_stage}"
)
cfg.exp_code = cfg.exp_code.lower().replace(".", "")

# -----------------------------------------------
for k, v in cfg.__dict__.items():
    if k not in config.__dict__:
        raise ValueError(f"config '{k}' is not registered in the base config")
config = cfg  # wrap back


# ========== [ func to load loader ] ============
def func_load_loader():
    import loader

    return loader.build_hf_dataloader


# ========== [ func to load decoder ] ===========
def func_load_decoder():
    import build_decoder

    if cfg.decoder == "qwen":
        return build_decoder.load_hf_qwen3
    else:
        raise ValueError(f"decoder '{cfg.decoder}' is not supported")


# ========== [ func to load encoder ] ===========
def func_load_encoder():
    import build_encoder

    return build_encoder.load_dummy_encoder


# ========== [ func to load forward ] ===========
# this function aims to decouple the forward function from the main code
# so that the forward function can be easily replaced by different methods
def func_load_forward():
    if cfg.decoder == "qwen":
        import methods.engine_ar_instruction_qwen as method
    else:
        raise ValueError(f"decoder '{cfg.decoder}' is not supported")

    return method.forward_hf_lang


# ========== [ func to filter params ] ==========
# the optimizer function can be customized in the config file
# it is must be provided in the config file
def func_optim_filter(cfg, model):
    # for decoder params
    decay_params = []
    other_params = []

    global_lr = cfg.lr
    global_wd = cfg.wd

    max_n_len = max([len(n) for n, _ in model.named_parameters()])

    for n, p in model.named_parameters():
        p.requires_grad = False

        # in pretraining stage, we train two parts:
        # 1. the translator in decoder
        if TRAINING_STAGE == "pretrain":
            if "decoder" in n and "layers" in n:
                layer_id_idx = n.split(".").index("layers") + 1
                layer_id = int(n.split(".")[layer_id_idx])

                # compute the actual translator layer ids
                # after converting the original model to surrogate one
                num_translators = cfg.num_translators
                translator_layer_ids = []
                for i, r in enumerate(cfg.translators_range):
                    o = 0  # offset
                    if i > 0:
                        o += sum(len(cfg.translators_range[i]) for i in range(i)) - 1
                    ids = [n - o for n in r[:num_translators]]
                    translator_layer_ids.append(ids)

                translator_layer_ids = sum(translator_layer_ids, [])
                if layer_id in translator_layer_ids:
                    p.requires_grad = True
                    if p.ndim >= 2:
                        decay_params.append(p)
                    else:
                        other_params.append(p)

        # in finetune-decoder stage, we train the decoder
        if TRAINING_STAGE == "finetune-decoder":
            if "decoder" in n:
                p.requires_grad = True
                if p.ndim >= 2:
                    decay_params.append(p)
                else:
                    other_params.append(p)

        print(
            f"p.requires_grad: {str(p.requires_grad):<5}, param: {n:<{max_n_len}}, {p.shape}"
        )

    optim_groups = [
        {
            "params": decay_params,
            "lr": global_lr,
            "weight_decay": global_wd,
        },
        {
            "params": other_params,
            "lr": global_lr,
            "weight_decay": 0.0,
        },
    ]

    return optim_groups
