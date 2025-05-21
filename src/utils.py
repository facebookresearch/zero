# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import os
import importlib
import builtins
import datetime
import random
import dill
import hashlib

import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import huggingface_hub

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType,
    MixedPrecision,
)
from enum import Enum
from torchvision import transforms
from huggingface_hub import HfApi


"""
helper classes for training
"""


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


"""
helper functions for debugging
"""


def save_tensor_images(
    images: torch.Tensor,
    count: int = 0,
    save_folder: str = "saved_images",
):
    """
    bs x c x h x w -> save images
    """
    assert images.ndim == 4, f"images should be 4D, but got {images.ndim}D"
    bs, c, h, w = images.shape

    os.makedirs(save_folder, exist_ok=True)
    rank = dist.get_rank() if torch.distributed.is_initialized() else 0

    for i in range(bs):
        x = images[i].float()
        x = (x - x.min()) / (x.max() - x.min())
        x = transforms.ToPILImage()(x)
        x.save(
            os.path.join(
                save_folder, f"image_{rank}_{str(i).zfill(3)}_{str(count).zfill(5)}.png"
            )
        )


"""
helper functions for training
"""


def load_config(argv):
    argv = argv[1:]  # rm the script self name
    if len(argv) == 0:
        raise ValueError("please pass the config file")
    if len(argv) > 1:
        raise ValueError(
            f"only accept one argument for the config file, but got {argv}"
        )

    cfg = None
    if argv[0].endswith(".py"):
        # config file from manual created python file
        print(argv[0])
        cfg_path = str(argv[0]).replace("/", ".").replace(".py", "")
        print(cfg_path)
        cfg = importlib.import_module(cfg_path)
    elif argv[0].endswith(".dill"):
        # config file from tools/_oh_my_submitit.py
        with open(argv[0], "rb") as f:
            cfg = dill.load(f)
    assert cfg is not None, f"config file not found: {argv[0]}"
    return cfg


def set_seed(seed, offset=0):
    seed = seed + offset
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


def set_dtype(args):
    assert args.dtype in ["float32", "bfloat16", "float16"]

    cuda_is_available = torch.cuda.is_available()
    if not args.force_to_use_fp16 and args.dtype == "float16":
        if cuda_is_available and torch.cuda.is_bf16_supported():
            args.dtype = "bfloat16"
            print("bfloat16 is supported: float16 -> bfloat16")

    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]

    # floating-point dtype for torch.tensor()
    fpdtype = {
        "float32": torch.cuda.FloatTensor if cuda_is_available else torch.FloatTensor,
        "float16": torch.cuda.HalfTensor if cuda_is_available else torch.HalfTensor,
        "bfloat16": (
            torch.cuda.BFloat16Tensor if cuda_is_available else torch.BFloat16Tensor
        ),
    }[args.dtype]

    args.ptdtype = ptdtype
    args.fpdtype = fpdtype
    return args


def set_fsdp_amp_policy(args):
    assert args.dtype in ["float32", "bfloat16", "float16"]

    cuda_is_available = torch.cuda.is_available()
    if not args.force_to_use_fp16 and args.dtype == "float16":
        if cuda_is_available and torch.cuda.is_bf16_supported():
            args.dtype = "bfloat16"
            print("bfloat16 is supported: float16 -> bfloat16")

    amp_policy = None

    if args.dtype == "float32":
        print(f"fsdp amp mode: fp32")

    if args.dtype == "float16":
        amp_policy = MixedPrecision(
            param_dtype=args.ptdtype,
            reduce_dtype=args.ptdtype,
            buffer_dtype=args.ptdtype,
        )
        print(f"fsdp amp mode: fp16")

    if args.dtype == "bfloat16":
        amp_policy = MixedPrecision(
            param_dtype=args.ptdtype,
            reduce_dtype=args.ptdtype,  # grad communication precision
            buffer_dtype=args.ptdtype,  # buffer precision
            cast_forward_inputs=True,
        )
        print(f"fsdp amp mode: bf16 with the policy {amp_policy}")

    return amp_policy


def setup_model_parallel(
    mute_non_master_ranks=False,
    force_to_single_world=False,
):
    if force_to_single_world:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "11705"
        os.environ["LOCAL_RANK"] = "0"
        print(f"> single world mode is on")

    # torchrun mode
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    if local_rank == -1 or world_size == -1:
        raise ValueError(
            "LOCAL_RANK or WORLD_SIZE is not set in the environment variables"
        )

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()

    dist.init_process_group(
        "nccl",
        timeout=datetime.timedelta(hours=2),
        device_id=device,
    )

    global_rank = dist.get_rank()
    print(
        f"> "
        f"local_rank: {local_rank}, "
        f"global_rank: {global_rank}, "
        f"world_size: {world_size}, "
        f"device: {device}"
    )

    if mute_non_master_ranks and global_rank > 0:
        sys.stdout = open(os.devnull, "w")
    return local_rank, global_rank, world_size, device


def setup_for_distributed(is_master, plain_print_mode=False):
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        force = force or (dist.get_world_size() > 8)
        if is_master or force:
            if plain_print_mode:
                builtin_print(*args, **kwargs)
            else:
                now = datetime.datetime.now().time().strftime("%H:%M:%S.%f")[:-3]
                builtin_print("[{}] ".format(now), end="")  # print with time stamp
                builtin_print(*args, **kwargs)

    builtins.print = print
    pass


def print_model_numel(model, model_name="model"):
    numel = sum(param.numel() for param in model.parameters()) / 1e6
    print(f"total params of {model_name}: {numel:.2f} M, {numel / 1e3:.2f} B")


"""
helper functions for saving results
"""


def _get_raw_param_name(n: str):
    n = n.replace("_fsdp_wrapped_module.", "")  # from fsdp
    n = n.replace("_checkpoint_wrapped_module.", "")  # from act checkpointing
    n = n.replace("_orig_mod.", "")  # from compile
    n = n.replace("module.", "")  # from ddp
    return n


def save_checkpoint(
    args,
    model,
    optimizer,
    scheduler,
    epoch,
    global_step,
    only_save_trainable_params=bool(1),
    is_master=True,
):
    if is_master:
        dir = os.path.join(args.ckpt_dir, args.exp_code)
        os.makedirs(dir, exist_ok=True)
        ckpt_path = os.path.join(
            dir,
            f"ckpt_{str(epoch).zfill(2)}_{str(global_step).zfill(7)}.pth",
        )
        print(f"saving checkpoint to {ckpt_path}")

    if args.fsdp_mode:
        # in fsdp_mode, we need to call the state_dict on each rank
        # then stream the overall states on the master rank to save
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            model_state_dict = model.state_dict()
    else:
        # ddp mode
        if is_master:
            model_state_dict = model.state_dict()

    raw_model = model.module if hasattr(model, "module") else model

    if only_save_trainable_params and is_master:
        # only save the trained model state dict for reducing the file size
        _state_dict = {}
        for n, p in raw_model.named_parameters():
            n = _get_raw_param_name(n)

            if p.requires_grad:
                try:
                    _state_dict[n] = model_state_dict[n]
                except KeyError:
                    _state_dict[n] = model_state_dict["module." + n]
        model_state_dict = _state_dict

    if optimizer is not None:
        optimizer_state_dict = optimizer.state_dict()
    else:
        optimizer_state_dict = None

    if scheduler is not None:
        scheduler_state_dict = scheduler.state_dict()
    else:
        scheduler_state_dict = None

    if is_master:
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model": model_state_dict,
                "optimizer": optimizer_state_dict,
                "scheduler": scheduler_state_dict,
            },
            ckpt_path,
        )
        print(f"checkpoint saved to {ckpt_path}")

        if args.ckpt_upload_to_hf:
            # upload the checkpoint to huggingface
            upload_to_hf(args, ckpt_path)


def load_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    strict=True,
    verbose=False,
    ignore_nonzero_unexpected_keys=False,
    ignore_adapter_keys=False,
    force_to_use_raw_param_name=False,
    dry_load_mode=False,
):
    rank = dist.get_rank()
    if dry_load_mode:
        print(f"dry load mode on rank {rank}")
        return 0, 0

    assert os.path.exists(path), f"ckpt not found: {path}"
    print(f"loading checkpoint from {path} on rank {rank}")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)

    has_ddp_module = not force_to_use_raw_param_name  # handle fsdp wrapped model
    ddp_module_pos = "prefix"  # 'prefix' | 'suffix'
    has_orig_mod = False  # handle compiled model
    idx_orig_mod = -1
    setting_per_param = {}
    for n, _ in model.named_parameters():
        if "fsdp" in n:
            has_ddp_module = False

        if "_orig_mod" in n:
            has_orig_mod = True
            idx_orig_mod = n.split(".").index("_orig_mod") - 1

        n = _get_raw_param_name(n)
        setting_per_param[n] = (
            has_ddp_module,
            ddp_module_pos,
            has_orig_mod,
            idx_orig_mod,
        )

        # reset for the next param
        has_ddp_module, has_orig_mod, idx_orig_mod = (
            (not force_to_use_raw_param_name) and has_ddp_module,
            False,
            -1,
        )

    if verbose:
        print("loading keys:")
    _state_dict = {}

    states = ckpt["model"] if "model" in ckpt else ckpt
    max_n_len = max([len(n) for n in states.keys()])
    for n, v in states.items():
        n_raw = _get_raw_param_name(n)
        if n_raw in setting_per_param:
            has_ddp_module, ddp_module_pos, has_orig_mod, idx_orig_mod = (
                setting_per_param[n_raw]
            )
        else:
            has_ddp_module, ddp_module_pos, has_orig_mod, has_orig_mod = (
                False,
                "prefix",
                False,
                -1,
            )

        if has_ddp_module and "module" not in n:
            ns = n.split(".")
            if ddp_module_pos == "prefix":
                ns.insert(0, "module")
            elif ddp_module_pos == "suffix":
                ns.insert(1, "module")
            else:
                raise ValueError(f"invalid ddp_module_pos: {ddp_module_pos}")
            n = ".".join(ns)

        if not has_ddp_module and "module" in n:
            n = n.replace("module.", "")

        if has_orig_mod and has_orig_mod > 0:
            ns = n.split(".")
            ns.insert(idx_orig_mod, "_orig_mod")  # for torch.compile each
            n = ".".join(ns)

            if has_ddp_module and "module" not in n:
                ns = n.split(".")
                if ddp_module_pos == "prefix":
                    ns.insert(0, "module")
                elif ddp_module_pos == "suffix":
                    ns.insert(1, "module")
                else:
                    raise ValueError(f"invalid ddp_module_pos: {ddp_module_pos}")
                n = ".".join(ns)

        if ignore_adapter_keys:
            if "adapter" in n:
                print(f". [ignore] adapter keys: {n:<{max_n_len}}: {v.shape}")
                continue
            if "embd_soi" in n or "embd_eoi" in n:
                print(f". [ignore] adapter keys: {n:<{max_n_len}}: {v.shape}")
                continue

        _state_dict[n] = v
        if verbose:
            print(f"- {n:<{max_n_len}}: {v.shape}")

    msgs = model.load_state_dict(_state_dict, strict=strict)
    del _state_dict

    if verbose and len(msgs.missing_keys) > 0:
        print("loading api message:\n", msgs)

    if len(msgs.unexpected_keys) > 0 and not ignore_nonzero_unexpected_keys:
        print("unexpected keys:", msgs.unexpected_keys)
        raise ValueError(f"unexpected keys found in the model state dict: {path}")

    if optimizer is not None:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception as e:
            print(f"failed to load optimizer state dict: {e}")

    if scheduler is not None:
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
        except Exception as e:
            print(f"failed to load scheduler state dict: {e}")

    epoch = int(ckpt["epoch"]) if "epoch" in ckpt else 1
    global_step = ckpt["global_step"] if "global_step" in ckpt else 1
    del ckpt

    return epoch, global_step


def upload_to_hf(args, save_path):
    """
    for large files or checkpoints, please consider using
    https://huggingface.co/docs/huggingface_hub/v0.29.3/en/package_reference/hf_api#huggingface_hub.HfApi.upload_large_folder
    """
    api = HfApi()

    assert REPO_ID is not None and ACCESS_TOKEN is not None

    REPO_ID = args.hf_repo_id
    ACCESS_TOKEN = args.hf_access_token
    assert REPO_ID is not None and ACCESS_TOKEN is not None
    huggingface_hub.login(token=ACCESS_TOKEN, write_permission=True)

    api.create_repo(
        repo_id=REPO_ID,
        token=ACCESS_TOKEN,
        private=True,
        exist_ok=True,
    )
    api.upload_file(
        path_or_fileobj=save_path,
        path_in_repo=os.path.basename(save_path),
        repo_id=REPO_ID,
        repo_type="model",
        token=ACCESS_TOKEN,
        commit_message=f"upload {REPO_ID}/{os.path.basename(save_path)}",
    )


"""
helper functions for misc
"""


def format_metric_to_gb(item):
    return round(item / (1024**3), ndigits=4)


def verify_min_gpu_count(min_gpus=2):
    """
    at least 2 gpus to run fsdp training
    """
    has_gpu = torch.cuda.is_available()
    num_gpu = torch.cuda.device_count()
    return has_gpu and num_gpu >= min_gpus


def assert_cfg(cfg):
    assert cfg.enable_kv_cache is False, "kv cache should be disabled for training"
    assert cfg.training_stage in [
        "pretrain-lang",
        "pretrain",
        "finetune-encoder",
        "finetune-decoder",
    ], (
        f"training stage {cfg.training_stage} not supported, "
        f"should be one of ['pretrain-lang', 'pretrain', 'finetune-encoder', 'finetune-decoder']"
    )

    torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
    if torch_ver >= [2, 3]:  # torch 2.3.0+
        torch.utils.data.datapipes.utils.common.DILL_AVAILABLE = (
            torch.utils._import_utils.dill_available()
        )


def get_md5sum(file_path):
    """
    Calculate the MD5 checksum of a file.
    """
    md5_hash = hashlib.md5()

    # Read the file in chunks to handle large files efficiently
    with open(file_path, "rb") as file:
        # Read in 4MB chunks
        for chunk in iter(lambda: file.read(4096 * 1024), b""):
            md5_hash.update(chunk)

    return md5_hash.hexdigest()
