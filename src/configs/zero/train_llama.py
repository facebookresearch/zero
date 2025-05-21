import copy
import os.path as osp
from configs.config_base import config

# unwrap base config to override
cfg = copy.deepcopy(config)
# -----------------------------------------------


# ============= [ global config ] ===============
TOTAL_NUM_GPUS_PLAN_TO_USE = 4  # gpu cards
IS_SURROGATE = bool(1)
TRAINING_STAGE = (
    "pretrain"
    # "finetune-encoder"
    # "finetune-decoder"
)


# ============= [ overwrite cfg ] ===============
# for doing sth. different according the training stage
cfg.training_stage = TRAINING_STAGE

cfg.llama_model_scale = "3"
cfg.llama_model_version = "3.2"

# for inserting the translator in language model (decoder)
# cfg.translators_range tuple can have more than one range,
# for example: = (list(range(16, 27)), list(range(40, 79)))
# in this case, two translators will be inserted to replace those two ranged layers
cfg.enable_translator = IS_SURROGATE
if cfg.enable_translator:
    if cfg.llama_model_scale == "3":
        cfg.translators_range = (list(range(16, 27)),)
    elif cfg.llama_model_scale == "8":
        cfg.translators_range = (list(range(17, 31)),)
    elif cfg.llama_model_scale == "70":
        cfg.translators_range = (list(range(40, 79)),)
    else:
        raise ValueError(f"unknown llama model scale: {cfg.llama_model_scale}")

if cfg.enable_translator:
    # can be more than one dataset in the tuple
    cfg.data_name = ("llava-1.5-665k-genqa-500k-instructions",)
else:
    cfg.data_name = ("llava-1.5-665k-instructions",)

# hyper-parameters for training
if cfg.training_stage == "pretrain":
    total_batch_size = 256
    cfg.batch_size = 8
    cfg.lr = 1e-4
elif cfg.training_stage == "finetune-encoder":
    total_batch_size = 128
    cfg.batch_size = 8
    if cfg.enable_translator:
        cfg.lr = 5e-5
    else:
        cfg.lr = 2e-4
elif cfg.training_stage == "finetune-decoder":
    total_batch_size = 128
    cfg.batch_size = 4
    cfg.lr = 5e-5
else:
    raise ValueError(f"unknown training stage: {cfg.training_stage}")

cfg.gradient_accumulation_steps = (
    total_batch_size // cfg.batch_size // TOTAL_NUM_GPUS_PLAN_TO_USE
)
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

cfg.adapter_mlp_depth = 2

cfg.fsdp_mode = bool(1)
cfg.fsdp_cpu_offload = bool(0)
cfg.fsdp_activation_checkpointing = bool(1)

cfg.enable_sdpa = bool(1)
cfg.enable_lora = bool(0)

# set False to save parameters and optimizer states
# set True to save only parameters
cfg.ckpt_save_params_only = bool(1)

cfg.image_ratio_type = "pad"  # pad | anyres

# NOTE: enable anyres
# this code supports simple anyres training but not used in our paper
# cfg.image_ratio_type = "anyres"
# cfg.anyres_grid_pinpoints = "(1x1),...,(6x6)"
# cfg.min_input_image_size = 336 * 1  # to control total number of image tokens
# cfg.max_input_image_size = 336 * 2

cfg.enable_image_prompt = bool(1)  # for wrapping the image tokens
cfg.enable_random_image_prompt = bool(1)

# compute the loss only on the response tokens
cfg.ignore_instruction_token_ids = bool(1)
cfg.include_special_tokens_in_masking_instruction_tokens = (
    TRAINING_STAGE != "finetune-decoder"
)

# for partially training the encoder
cfg.encoder_ft_layers_after = 16  # -1 freeze all layers | 16
cfg.encoder_lr_scaler = 1.0  # 0.1 | 0.5 | 1.0 [default]
cfg.encoder_shave_last_n_layers = -1

# NOTE: using pretrained weights
cfg.load_pretrained = bool(0)
cfg.load_pretrained_path = osp.join(cfg.ckpt_dir, "none", "none")
if TRAINING_STAGE == "finetune-encoder" and IS_SURROGATE:
    # this is our 2nd training stage in the paper,
    # we train the vision encoders on top of surrogate models
    # thus, we need to load the translators from the pretraining stage
    cfg.load_pretrained = bool(1)

    if cfg.llama_model_scale == "3":
        cfg.load_pretrained_path = osp.join(
            cfg.ckpt_dir,
            "llama3.2-3b_adapter_translator",
            "ckpt_01_0003568.pth",
        )
    elif cfg.llama_model_scale == "8":
        cfg.load_pretrained_path = osp.join(
            cfg.ckpt_dir,
            "llama3.1-8b_adapter_translator",
            "ckpt_01_0004393.pth",
        )
    elif cfg.llama_model_scale == "70":
        cfg.load_pretrained_path = osp.join(
            cfg.ckpt_dir,
            "llama3.1-70b_adapter_translator",
            "ckpt_01_0004393.pth",
        )

# NOTE: resuming training
cfg.resume = bool(0)
cfg.from_scratch = bool(1)
if TRAINING_STAGE == "finetune-decoder" and not IS_SURROGATE:
    # this is our 3rd training stage in the paper,
    # we finetune the decoder on top of the pretrained vision encoders
    # thus, we need to load the surrogate-trained encoders from the 2nd stage
    cfg.resume = bool(1)

    if cfg.llama_model_scale == "3":
        cfg.resume_ckpt_path = osp.join(
            cfg.ckpt_dir,
            "llama3.2-3b_surrogate-trained-encoder",
            "ckpt_01_0004880.pth",
        )
    elif cfg.llama_model_scale == "8":
        cfg.resume_ckpt_path = osp.join(
            cfg.ckpt_dir,
            "llama3.1-8b_surrogate-trained-encoder",
            "ckpt_01_0004880.pth",
        )
    elif cfg.llama_model_scale == "70":
        cfg.resume_ckpt_path = osp.join(
            cfg.ckpt_dir,
            "llama3.1-70b_surrogate-trained-encoder",
            "ckpt_01_0004880.pth",
        )

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
    f"{cfg.decoder}{cfg.llama_model_version}_{cfg.llama_model_scale}b-{is_surrogate}"
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

    return loader.build_dataloader


# ========== [ func to load decoder ] ===========
def func_load_decoder():
    import build_decoder

    if cfg.decoder == "llama3":
        return build_decoder.load_hf_llama3
    else:
        raise ValueError(f"decoder '{cfg.decoder}' is not supported")


# ========== [ func to load encoder ] ===========
def func_load_encoder():
    import build_encoder

    return build_encoder.load_clip


# ========== [ func to load forward ] ===========
# this function aims to decouple the forward function from the main code
# so that the forward function can be easily replaced by different methods
def func_load_forward():
    if cfg.decoder == "llama3":
        import methods.engine_ar_instruction_llama as method
    else:
        raise ValueError(f"decoder '{cfg.decoder}' is not supported")

    return method.forward_hf


# ========== [ func to filter params ] ==========
# the optimizer function can be customized in the config file
# it is must be provided in the config file
def func_optim_filter(cfg, model):
    # for decoder params
    decay_params = []
    other_params = []

    # for encoder params
    encoder_params = []

    global_lr = cfg.lr
    global_wd = cfg.wd

    max_n_len = max([len(n) for n, _ in model.named_parameters()])

    for n, p in model.named_parameters():
        p.requires_grad = False

        # we always train the adapter in encoder
        if "adapter" in n:
            p.requires_grad = True
            if p.ndim >= 2:
                decay_params.append(p)
            else:
                other_params.append(p)

        # we always train the special tokens in adapter
        if "embd_soi" in n or "embd_eoi" in n or "special_tokens" in n:
            p.requires_grad = True
            if p.ndim >= 2:
                decay_params.append(p)
            else:
                other_params.append(p)

        # in pretraining stage, we train two parts:
        # 1. the adapter in encoder
        # 2. the translator in decoder
        if TRAINING_STAGE == "pretrain":
            if "vision_model" in n and "post_layernorm" in n:
                p.requires_grad = True
                encoder_params.append(p)

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

        # in finetune-encoder stage, we train the encoder
        # but only the last 8 layers, including the post_layernorm
        if TRAINING_STAGE == "finetune-encoder":
            if "vision_model" in n:
                if "post_layernorm" in n:
                    p.requires_grad = True
                    other_params.append(p)

                if "layers" in n:
                    layer_id_idx = n.split(".").index("layers") + 1
                    layer_id = int(n.split(".")[layer_id_idx])

                    if layer_id >= cfg.encoder_ft_layers_after:
                        p.requires_grad = True
                        encoder_params.append(p)

        # in finetune-decoder stage, we train the decoder
        if TRAINING_STAGE == "finetune-decoder":
            if "decoder" in n:
                p.requires_grad = True
                if p.ndim >= 2:
                    decay_params.append(p)
                else:
                    other_params.append(p)

            # we also train the last 8 layers of the encoder
            if "vision_model" in n:
                if "post_layernorm" in n:
                    p.requires_grad = True
                    other_params.append(p)

                if "layers" in n:
                    layer_id_idx = n.split(".").index("layers") + 1
                    layer_id = int(n.split(".")[layer_id_idx])

                    if layer_id >= cfg.encoder_ft_layers_after:
                        p.requires_grad = True
                        encoder_params.append(p)

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

    if len(encoder_params) > 0:
        encoder_lr = global_lr * cfg.encoder_lr_scaler
        optim_groups.append(
            {
                "params": encoder_params,
                "lr": encoder_lr,
                "weight_decay": 0.0,
            }
        )
        print(
            f"insert encoder_params with "
            f"lr: {encoder_lr}, "
            f"wd: 0.0 in the optimizer"
        )

    return optim_groups
