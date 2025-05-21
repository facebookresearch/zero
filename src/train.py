# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import functools
import sys
import time
import wandb

import torch
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import (
    _or_policy,
    lambda_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    checkpoint_wrapper,
    apply_activation_checkpointing,
)

from utils import (
    AverageMeter,
    load_config,
    setup_model_parallel,
    setup_for_distributed,
    set_dtype,
    set_seed,
    set_fsdp_amp_policy,
    save_checkpoint,
    load_checkpoint,
    print_model_numel,
    assert_cfg,
)
from models.adapter import Zero


def main(cfg):
    """
    cfg object contains two groups:

    group-1 for config-params
        - cfg.config (Class): the arguments

    group-2 for functions
        - cfg.func_load_decoder (func): the function for returning the decoder
        - cfg.func_load_forward (func): the method engine for training, import from methods
        - cfg.func_optim_filter (func): the param filter for optimizer
    """
    # decompose cfg into functions and config-params
    load_encoder = cfg.func_load_encoder()
    load_decoder = cfg.func_load_decoder()
    func_forward = cfg.func_load_forward()
    load_loader = cfg.func_load_loader()
    optim_filter = cfg.func_optim_filter

    # load function before overwriting cfg
    cfg = cfg.config

    # assert cfg
    assert_cfg(cfg)

    # set process groups
    local_rank, global_rank, world_size, device = setup_model_parallel(
        mute_non_master_ranks=cfg.mute_non_master_ranks,
    )
    cfg.device = device
    master_process = global_rank == 0

    # set print with timestamp
    if cfg.mute_non_master_ranks:
        setup_for_distributed(master_process)

    # set wandb
    if cfg.enable_wandb and master_process:
        wandb.login(key=cfg.wandb_key)
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.wandb_track_name,
            config=vars(cfg),
        )

    # log hyper params after wandb init
    print(f"rank {global_rank:3d} - world size: {world_size}")
    print(f"hyper params:")
    for k, v in cfg.__dict__.items():
        print(f"- {k}: {v}")

    # set seed
    set_seed(cfg.seed, offset=0 if cfg.fsdp_mode else global_rank)

    # set tf32
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    # set dtype
    cfg = set_dtype(cfg)
    print(f"ptdtype: {cfg.ptdtype}, fpdtype: {cfg.fpdtype}")

    # load encoder from huggingface
    encoder, image_processor, image_input_size = load_encoder(
        cfg=cfg, model_name=cfg.encoder_model_name
    )
    cfg.image_input_size = image_input_size

    # build decoder
    num_embd, decoder, tokenizer, FSDP_DECODER_LAYER, is_giant_model = load_decoder(cfg)
    cfg.num_embd = num_embd

    # build adapter
    model = Zero(
        cfg=cfg,
        encoder=encoder,
        decoder=decoder,
    )  # fsdp will handle the model device

    # load pretrained weights before wrapping model with ddp/fsdp
    if cfg.load_pretrained:
        dry_load_mode = is_giant_model and not master_process
        if dry_load_mode:
            print("dry_load_mode is enabled due to giant model training")

        load_checkpoint(
            cfg.load_pretrained_path,
            model,
            optimizer=None,
            scheduler=None,
            strict=False,
            verbose=True,
            ignore_nonzero_unexpected_keys=cfg.ignore_nonzero_unexpected_keys,
            ignore_adapter_keys=cfg.ignore_adapter_keys_pretrained,
            force_to_use_raw_param_name=True,
            dry_load_mode=dry_load_mode,
        )
        print(f"pretrained weights loaded from {cfg.load_pretrained_path}")

    global_step = 1
    start_epoch = 1
    pgs = -1  # previous global step for saving ckpt
    rgs = -1  # resumed global step for training

    if cfg.resume_ckpt_path and cfg.resume:
        dry_load_mode = is_giant_model and not master_process
        if dry_load_mode:
            print("dry_load_mode is enabled due to giant model training")

        # rse: resumed start epoch
        # rgs: resumed global step
        rse, rgs = load_checkpoint(
            cfg.resume_ckpt_path,
            model,
            optimizer=None,
            scheduler=None,
            strict=False,
            verbose=True,
            ignore_nonzero_unexpected_keys=cfg.ignore_nonzero_unexpected_keys,
            ignore_adapter_keys=cfg.ignore_adapter_keys_resume,
            force_to_use_raw_param_name=True,
            dry_load_mode=dry_load_mode,
        )
        print(f"resume ckpt loaded from {cfg.resume_ckpt_path}")

        if is_giant_model:
            # the giant model is only loaded on rank 0, so we need to broadcast
            # wait for the rank 0 to finish loading the ckpt
            dist.barrier()

            # init rse and rgs tensors
            rse_tensor = torch.tensor(rse if master_process else 0, device=cfg.device)
            rgs_tensor = torch.tensor(rgs if master_process else 0, device=cfg.device)

            # broadcast from rank 0
            dist.broadcast(rse_tensor, src=0)
            dist.broadcast(rgs_tensor, src=0)

            # update the variables on all ranks
            rse = int(rse_tensor.item())
            rgs = int(rgs_tensor.item())
            print(
                f"broadcasted resumed start epoch (rse): {rse}, "
                f"resumed global step (rgs): {rgs}"
            )

            # another barrier to ensure all ranks have received values
            dist.barrier()

        # NOTE: for now, we don't load the states of optimizer and lr_scheduler
        # # the optimizer might have its state tensors on a different device
        # # than the model, so we need to move them accordingly
        # for state in optimizer.state.values():
        #     for k, v in state.items():
        #         if isinstance(v, torch.Tensor):
        #             state[k] = v.to(device)

    # set pure bfloat16 in default
    model.to(cfg.ptdtype)
    print(f"move model to {cfg.ptdtype}")

    # register forward from methods
    _forward = func_forward.__get__(model, model.__class__)
    setattr(model, "forward", _forward)

    # call before wrapping model with ddp/fsdp
    print_model_numel(model, model_name="zero")

    # compile model
    if cfg.compile_model:
        model = torch.compile(model)

    # fsdp wrap policy for peft models
    def _fsdp_peft_wrap_policy(model, transformer_layer_names):
        """
        wrap the peft modules separate from the transformer layer in auto_wrapping policy to keep
        peft models having require_grad=True while the rest of the model is require_grad=False
        """

        def lambda_policy_fn(module):
            if (
                len(list(module.named_children())) == 0
                and getattr(module, "weight", None) is not None
                and module.weight.requires_grad
            ):
                return True
            return False

        lambda_policy = functools.partial(
            lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn
        )
        transformer_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=set(transformer_layer_names),
        )
        auto_wrap_policy = functools.partial(
            _or_policy, policies=[lambda_policy, transformer_wrap_policy]
        )
        return auto_wrap_policy

    # calculate device mesh for fsdp
    def _fsdp_calc_device_mesh(num_shard_nodes: float = 1.0):
        """
        perform model sharding within single node or across multiple nodes

        Args:
            num_shard_nodes (float): number of shard nodes per device, support float like
                0.25 or 0.5 for a subset of gpus in a node
        """
        num_gpus = torch.cuda.device_count()
        num_shard_groups = int(num_shard_nodes * num_gpus)
        num_replica_groups = world_size // num_shard_groups
        assert num_replica_groups > 0 and num_shard_groups > 0
        assert world_size == num_replica_groups * num_shard_groups

        mesh_2d = init_device_mesh(
            "cuda",
            (num_replica_groups, num_shard_groups),
            mesh_dim_names=("replica", "shard"),
        )
        print(
            f"fsdp device mesh:",
            f"num_replica_groups = {num_replica_groups}, "
            f"num_shard_groups = {num_shard_groups}",
        )
        return mesh_2d

    # init ddp/fsdp
    if cfg.fsdp_mode:
        print(f"fsdp target layer: {FSDP_DECODER_LAYER}")

        # set wrap policy
        if cfg.enable_lora:
            wrap_policy = _fsdp_peft_wrap_policy(model, [FSDP_DECODER_LAYER])
        else:
            wrap_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls=set([FSDP_DECODER_LAYER]),
            )
        print(f"fsdp wrap policy: {wrap_policy}")

        # set precision policy
        if cfg.ptdtype == torch.bfloat16:
            precision_policy = None
        else:
            precision_policy = set_fsdp_amp_policy(cfg)
        print(f"fsdp precision policy: {precision_policy}")

        # set device mesh
        if is_giant_model and not cfg.enable_translator:
            num_shard_nodes = 2
        else:
            # the model with less than ~50B can be sharded within a single node
            # e.g., the 70B surrogate is 37B
            num_shard_nodes = 1

        # set fsdp low cpu mode for giant model to save cpu memory
        fsdp_low_cpu_mode = True if is_giant_model else False

        device_mesh_2d = _fsdp_calc_device_mesh(num_shard_nodes=num_shard_nodes)

        # https://pytorch.org/docs/stable/fsdp.html
        model = FSDP(
            model,
            auto_wrap_policy=wrap_policy,
            mixed_precision=precision_policy,
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            use_orig_params=True,
            limit_all_gathers=True,
            device_mesh=device_mesh_2d,
            device_id=device,
            sync_module_states=fsdp_low_cpu_mode,
            cpu_offload=(
                CPUOffload(offload_params=True) if cfg.fsdp_cpu_offload else None
            ),
            param_init_fn=(
                (
                    lambda module: module.to_empty(
                        device=torch.device("cuda"), recurse=False
                    )
                )
                if fsdp_low_cpu_mode and global_rank != 0
                else None
            ),
        )

        def _apply_fsdp_checkpointing(model):
            """
            return None as model is updated in-place
            """

            print("applying fsdp activation checkpointing ...")
            non_reentrant_wrapper = functools.partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            )
            check_fn = lambda submodule: isinstance(submodule, FSDP_DECODER_LAYER)

            apply_activation_checkpointing(
                model,
                checkpoint_wrapper_fn=non_reentrant_wrapper,
                check_fn=check_fn,
            )

        if cfg.fsdp_activation_checkpointing:
            for m in [model.decoder]:
                m.enable_input_require_grads()
                m.gradient_checkpointing_enable()
            _apply_fsdp_checkpointing(model)
            print(": done")
    else:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    print(model)

    # build loader
    loader, sampler = load_loader(
        cfg,
        global_rank=global_rank,
        world_size=world_size,
        is_train=True,
        image_processor=image_processor,
    )
    print(f"rank {global_rank:3d} - dataset length: {len(loader)}")

    # set optimizer
    optim_groups = optim_filter(cfg, model)
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=cfg.lr,
        betas=(cfg.adamw_beta1, cfg.adamw_beta2),
        weight_decay=0.0,
    )
    grad_clip = cfg.grad_clip
    print(optimizer)

    # calc batch size and grad accumulation
    grad_accum_steps = cfg.gradient_accumulation_steps
    batch_size_per_rank = cfg.batch_size
    batch_size = cfg.batch_size * world_size * grad_accum_steps  # total batch size

    # calc training steps
    steps_per_epoch = len(loader) // grad_accum_steps
    total_steps = steps_per_epoch * cfg.epochs
    if cfg.enable_warmup:
        warmup_steps = int(max(cfg.warmup_ratio * total_steps // 100, cfg.warmup_steps))
    else:
        warmup_steps = 0

    init_lrs = []  # for warmup
    for param_group in optimizer.param_groups:
        init_lrs.append(param_group["lr"])
    print("init_lrs:", init_lrs)

    # set lr scheduler
    lr_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=cfg.min_lr,
    )

    # calc ckpt save interval
    ckpt_save_interval = total_steps * cfg.ckpt_save_interval_ratio // 100

    # print training info
    print(
        f"total training steps: {total_steps}\n",
        f"  epochs: {cfg.epochs}\n",
        f"  steps per epoch: {steps_per_epoch}\n",
        f"  dataloader length on each rank: {len(loader)} iters\n",
        f"  num of gpu ranks: {world_size}\n",
        f"  gradient accumulation steps: {grad_accum_steps}\n",
        f"  batch size per rank: {batch_size_per_rank}\n",
        f"  total batch size: {batch_size}\n",
        f"  warmup steps: {warmup_steps} w/ ratio {100 * warmup_steps / total_steps:.2f}%\n",
        f"  ckpt save interval: {ckpt_save_interval}\n",
        f"  grad clip factor: {grad_clip}\n",
        f"  max seq len: {cfg.max_seq_len}",
    )

    # switch mode
    model.train()

    # resume training
    if cfg.resume_ckpt_path and cfg.resume:
        if not cfg.from_scratch:
            start_epoch = int(rse)  # resumed start epoch
            global_step = (start_epoch - 1) * steps_per_epoch  # then count to the rgs

        print(f"resume training at epoch {start_epoch} and global step {global_step}")

    # NOTE: early testing the checkpointing,
    #       if having issues like OOM, the error will be raised here
    # save_checkpoint(
    #     cfg,
    #     model,
    #     None,  # optimizer,
    #     None,  # lr_scheduler,
    #     0,  # local_epoch,
    #     0,  # global_step,
    #     is_master=master_process,
    # )

    # train
    for local_epoch in range(start_epoch, cfg.epochs + 1):
        optimizer.zero_grad(set_to_none=True)

        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(local_epoch)

        running_avg_batch_time = AverageMeter("time", ":6.3f")
        running_avg_loss_value = AverageMeter("loss", ":6.3f")
        grad_norm = 0

        t0 = time.perf_counter()
        for local_step, batch_data in enumerate(loader, start=1):
            data_time = time.perf_counter() - t0

            # skip the first steps if forced to resume from a specific step
            if cfg.resume and not cfg.from_scratch:
                if local_epoch != start_epoch:
                    print(f"skip epoch {local_epoch} to resume at epoch {start_epoch}")
                    break

                if local_step % grad_accum_steps == 0:
                    global_step += 1
                    if global_step > warmup_steps:
                        lr_scheduler.step()
                    if global_step % cfg.log_interval == 0:
                        print(f"resume to skip global step {global_step}")

                if global_step < rgs:
                    continue
                else:
                    global_step += 1  # to avoid saving the same ckpt
                    print(f"resume training at global step {global_step}")

                    cfg.resume = False  # reset the flag
                    t0 = time.perf_counter()  # reset t0

            if (
                local_step == 1
                or local_step % grad_accum_steps == 0
                or not cfg.from_scratch
            ):
                curr_lrs = []
                lr_scheduler.step()

                if global_step > warmup_steps:
                    for group_id, param_group in enumerate(optimizer.param_groups):
                        curr_lr = param_group["lr"]
                        curr_lrs.append(curr_lr)
                else:
                    for group_id, param_group in enumerate(optimizer.param_groups):
                        # linear warmup
                        curr_lr = init_lrs[group_id] * global_step / warmup_steps

                        # # sinusoidal warmup
                        # curr_lr = init_lrs[group_id] * math.sin(
                        #     math.pi * global_step / warmup_steps / 2
                        # )

                        param_group["lr"] = curr_lr
                        curr_lrs.append(curr_lr)

                if not cfg.from_scratch:
                    cfg.from_scratch = True  # reset the flag

            # forward
            loss = model(cfg, batch_data, tokenizer)

            # for grad accumulation
            loss = loss / grad_accum_steps

            if num_shard_nodes > 1:
                # NOTE: mainly for 70B training because it is sharded across 2 nodes minimum (not within a single node),
                # the gradient accumulation logic is different from the model sharded within a single node
                # otherwise, the backward will raise the OOM error
                # https://github.com/meta-llama/llama-cookbook/blob/main/src/llama_cookbook/utils/train_utils.py#L174

                # NOTE: this forward pass can also be used for other model scales, like 8B, 13B,
                # the difference with the other forward pass is that
                # this implementation takes less gpu memory but a little more batch time because of reducing gradients for every step
                # in contrast, the other one takes a little more gpu memory but less batch time
                # if training models on limited gpu memory or long sequences, picking this one is optimal

                # backward
                loss.backward()

                # step optim
                if local_step % grad_accum_steps == 0 or local_step == len(loader):
                    if grad_clip > 0.0:
                        if cfg.fsdp_mode:
                            grad_norm = model.clip_grad_norm_(grad_clip)
                        else:
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                model.parameters(), grad_clip
                            )

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    # next step
                    global_step += 1

            else:
                # backward
                if local_step % grad_accum_steps != 0:
                    with model.no_sync():
                        loss.backward()
                else:
                    # w/ grad sync
                    loss.backward()

                    # clip grad
                    if grad_clip > 0.0:
                        if cfg.fsdp_mode:
                            grad_norm = model.clip_grad_norm_(grad_clip)
                        else:
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                model.parameters(), grad_clip
                            )

                    # step optim
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    # next step
                    global_step += 1

            # recycle mem for the next step
            torch.cuda.empty_cache()

            # log info
            t1 = time.perf_counter()
            batch_time = t1 - t0

            # update running meters
            running_avg_batch_time.update(batch_time)
            curr_loss = loss.item() * grad_accum_steps
            running_avg_loss_value.update(curr_loss)

            if global_step % cfg.log_interval == 0:
                warmup_steps_str_len = len(str(warmup_steps))
                epoch_str_len = len(str(cfg.epochs))
                steps_str_len = len(str(total_steps))

                running_percent = global_step / total_steps * 100
                warmup_tag = (
                    f"warmup step: {str(global_step).zfill(warmup_steps_str_len)} ({warmup_steps})"
                    if global_step <= warmup_steps
                    else ""
                )

                curr_lrs_str = " ' ".join([f"{lr:.10f}" for lr in curr_lrs])
                print(
                    f"epoch: {str(local_epoch).zfill(epoch_str_len)} ",
                    f"step: {str(global_step).zfill(steps_str_len)} ({total_steps}) | {running_percent:5.2f}% ",
                    f"lr: {curr_lrs_str} ",
                    f"loss: {curr_loss:>6.3f} ({running_avg_loss_value.avg:>6.3f}) ",
                    f"grad norm: {grad_norm:>6.3f} " if grad_norm > 0 else "",
                    f"data time: {data_time:>7.5f}s ",
                    f"batch time: {batch_time:>6.3f}s ({running_avg_batch_time.avg:>6.3f}s) ",
                    f"{warmup_tag}",
                )
                if cfg.enable_wandb and master_process:
                    running_step = (
                        local_step // grad_accum_steps
                        + (local_epoch - 1) * steps_per_epoch
                    )
                    wandb.log(
                        {
                            "step": running_step,
                            "loss": curr_loss,
                            "lr": max(curr_lrs),
                            "grad_norm": grad_norm,
                            "percent": running_percent,
                            "epoch": local_epoch,
                        }
                    )

            # save ckpt
            if global_step % ckpt_save_interval == 0:

                # skip the ckpt if the global step is the same as the last saved one
                if global_step == pgs:
                    t0 = time.perf_counter()
                    continue

                if cfg.fsdp_mode:
                    dist.barrier()

                save_checkpoint(
                    cfg,
                    model,
                    None if cfg.ckpt_save_params_only else optimizer,
                    lr_scheduler,
                    local_epoch,
                    global_step,
                    is_master=master_process,
                )
                pgs = global_step

                if cfg.fsdp_mode:
                    dist.barrier()

            # stop training
            if global_step % steps_per_epoch == 0:
                break

            # update time
            t0 = time.perf_counter()

        if cfg.fsdp_mode:
            dist.barrier()

        # save ckpt at the end of each epoch
        save_checkpoint(
            cfg,
            model,
            None if cfg.ckpt_save_params_only else optimizer,
            lr_scheduler,
            local_epoch,
            global_step,
            is_master=master_process,
        )

        if cfg.fsdp_mode:
            dist.barrier()

        # avoid breaking the next epoch
        global_step += 1

    if cfg.enable_wandb and master_process:
        wandb.finish()

    destroy_process_group()


if __name__ == "__main__":
    cfg = load_config(sys.argv)
    main(cfg)
