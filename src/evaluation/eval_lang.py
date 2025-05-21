import sys
import os
import json
import wandb
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from torch.distributed import destroy_process_group
from utils import (
    load_config,
    setup_model_parallel,
    setup_for_distributed,
    print_model_numel,
    format_metric_to_gb,
    set_dtype,
    set_seed,
    load_checkpoint,
)

from models.adapter import Zero

from lm_eval.evaluator import simple_evaluate
from lm_eval.tasks import TaskManager


def main(cfg):
    """
    cfg contains three objectives:
        - cfg.config (Class): the arguments
        - cfg.func_load_encoder (func): the function for returning the encoder
        - cfg.func_load_decoder (func): the function for returning the decoder
        - cfg.func_load_forward (func): the method engine for training, import from methods
        - cfg.func_optim_filter (func): the param filter for optimizer
    """
    # decompose cfg into functions and config-params
    load_decoder = cfg.func_load_decoder()
    load_gen_cls = cfg.func_load_gen_cls()
    cfg.config.benchmark = cfg.func_load_benchmark()

    # load function before overwriting cfg
    cfg = cfg.config
    cfg.batch_size = cfg.eval_batch_size

    # init ranks
    # NOTE: for large model inference (e.g., llama-3.1-70B),
    # multi-gpu inference with model parallelism requires launching this script
    # directly with `python3` rather than `torchrun`.
    # to enable this, set `cfg.jupyter_mode = True` as a workaround.
    # ref: https://huggingface.co/docs/accelerate/en/usage_guides/big_modeling
    local_rank, global_rank, world_size, device = setup_model_parallel(
        mute_non_master_ranks=cfg.mute_non_master_ranks,
        force_to_single_world=cfg.force_to_single_world,
    )
    cfg.local_rank = local_rank
    cfg.global_rank = global_rank
    cfg.world_size = world_size
    cfg.device = device
    master_process = global_rank == 0

    # set wandb
    if cfg.enable_wandb and master_process:
        wandb.login(key=cfg.wandb_key)
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name="eval_" + cfg.wandb_track_name,
            config=vars(cfg),
        )

    print(f"hyper params:")
    for k, v in cfg.__dict__.items():
        print(f"- {k}: {v}")

    # set print with timestamp
    if cfg.mute_non_master_ranks:
        setup_for_distributed(master_process)

    # set seed
    set_seed(cfg.seed)

    # set tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # set cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # gpu memory
    device_properties = torch.cuda.get_device_properties(device)
    gpu_mem = format_metric_to_gb(device_properties.total_memory)
    print(f"gpu memory: {gpu_mem:.2f}GB")

    # set dtype
    cfg = set_dtype(cfg)

    # build decoder
    num_embd, decoder, tokenizer, _, _ = load_decoder(cfg)
    cfg.num_embd = num_embd

    # build adapter
    model = Zero(
        cfg=cfg,
        encoder=None,
        decoder=decoder,
    )
    model.to(cfg.ptdtype)
    model.eval()
    model.requires_grad_(False)
    print_model_numel(model, model_name="zero")
    print(model)

    # compile model
    if cfg.compile_model:
        # torch._dynamo.config.accumulated_cache_size_limit = 128
        model = torch.compile(model)

    # load checkpoint
    if isinstance(cfg.eval_ckpt_path, str):
        cfg.eval_ckpt_path = [cfg.eval_ckpt_path]
    if cfg.enable_translator:
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

    if master_process:
        if os.path.exists(cfg.eval_result_file_path):
            pass
        os.makedirs(os.path.dirname(cfg.eval_result_file_path), exist_ok=True)

    # evaluation
    task_manager = TaskManager()

    # for json serialization
    def _handle_non_serializable(o):
        if isinstance(o, np.int64) or isinstance(o, np.int32):
            return int(o)
        elif isinstance(o, set):
            return list(o)
        else:
            return str(o)

    assert cfg.batch_size == 1, "batch size must be 1 for evaluation"

    for benchmark, num_fewshot in cfg.benchmark:
        print(f"evaluating benchmark: {benchmark} ...")

        # reset save path for each benchmark
        save_path = (
            cfg.eval_result_file_path.replace(".json", f"_{benchmark}.json")
            .replace("NUM_FEWSHOT", f"{num_fewshot}")
            .lower()
        )
        cfg.output_path = save_path

        # call simple_evaluate
        # https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/evaluator.py#L48
        results = simple_evaluate(
            model=load_gen_cls(cfg, model, tokenizer),
            tasks=[benchmark],
            num_fewshot=num_fewshot,
            task_manager=task_manager,
            batch_size=cfg.batch_size,
        )

        # wait for all processes to complete
        if world_size > 1:
            torch.distributed.barrier()

        # save results
        if master_process:
            with open(save_path, "w") as f:
                json.dump(
                    results,
                    f,
                    indent=2,
                    default=_handle_non_serializable,
                    ensure_ascii=True,
                )
            print(f"results saved to {save_path}")

            # print the first 20 lines of the result file
            with open(save_path, "r") as f:
                lines = f.readlines()
                print("".join(lines[:20]))

    if cfg.enable_wandb and master_process:
        wandb.finish()

    destroy_process_group()


if __name__ == "__main__":
    cfg = load_config(sys.argv)
    main(cfg)
