# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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
    upload_to_hf,
)

from loader import build_hf_dataloader
from models.adapter import Zero

from lmms_eval.evaluator import simple_evaluate
from lmms_eval.tasks import TaskManager
from lmms_eval.api.instance import Instance


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
    load_encoder = cfg.func_load_encoder()
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
    # to enable this, set `cfg.force_to_single_word = True` as a workaround.
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
    if cfg.mute_non_master_ranks and not cfg.eval_enable_vllm:
        setup_for_distributed(master_process, plain_print_mode=True)

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

    # build encoder
    encoder, image_processor, image_input_size = load_encoder(
        cfg=cfg, model_name=cfg.encoder_model_name
    )
    encoder = encoder.to(cfg.ptdtype)
    cfg.image_input_size = image_input_size

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
    model.requires_grad_(False)
    model.eval()
    print_model_numel(model, model_name="zero")
    print(model)

    # compile model
    if cfg.compile_model:
        # torch._dynamo.config.accumulated_cache_size_limit = 128
        model = torch.compile(model)

    # load checkpoint
    if isinstance(cfg.eval_ckpt_path, str):
        cfg.eval_ckpt_path = [cfg.eval_ckpt_path]
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
        os.makedirs(os.path.dirname(cfg.eval_result_file_path).lower(), exist_ok=True)

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

    EvalVlmCls = load_gen_cls(cfg, model, tokenizer, image_processor)

    for benchmark in cfg.benchmark:
        if benchmark in ["cvbench"]:
            continue

        print(f"evaluating on benchmark: {benchmark} ...")

        # reset save path for each benchmark
        save_path = cfg.eval_result_file_path.lower().replace(
            ".json", f"_{benchmark}.json"
        )
        if os.path.exists(save_path):
            os.system(f"rm -fr {save_path}")
        cfg.output_path = save_path

        cfg.eval_save_folder = cfg.eval_save_folder.replace("BENCHMARK_NAME", benchmark)
        if cfg.eval_save_images:
            print(f"images will be saved to {cfg.eval_save_folder}")
            os.makedirs(cfg.eval_save_folder, exist_ok=True)

        # call simple_evaluate

        # https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/lmms_eval/__main__.py#L267
        cfg.process_with_media = False

        if cfg.eval_enable_vllm and cfg.eval_vllm_tensor_parallel_size > 1:
            # vllm with tensor parallelism does not support data parallelism
            EvalVlmCls.world_size = 1
            EvalVlmCls.rank = 0

        # https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/lmms_eval/evaluator.py#L48
        results = simple_evaluate(
            model=EvalVlmCls,
            tasks=[benchmark],
            num_fewshot=cfg.num_fewshot,
            task_manager=task_manager,
            batch_size=cfg.batch_size,
            distributed_executor_backend="torchrun",
            cli_args=cfg,
        )

        # wait for all processes to complete
        if world_size > 1:
            torch.distributed.barrier()

        # save results
        if master_process:
            # remove the existing file or folder created by lmms-eval
            os.system(f"rm -fr {save_path}")

            # dump to single json file
            with open(save_path, "w") as f:
                json.dump(
                    results,
                    f,
                    indent=2,
                    default=_handle_non_serializable,
                    ensure_ascii=False,
                )
            print(f"results saved to {save_path}")

            # print the first 20 lines of the result file
            with open(save_path, "r") as f:
                lines = f.readlines()
                print("".join(lines[:20]))

            if cfg.eval_res_upload_to_hf:
                upload_to_hf(cfg, save_path)

    # eval on cv-bench
    if "cvbench" in cfg.benchmark:
        benchmark = "cvbench"
        print(f"evaluating on {benchmark} ...")

        cfg.data_name = (benchmark,)

        if cfg.eval_enable_vllm and cfg.eval_vllm_tensor_parallel_size > 1:
            # vllm with tensor parallelism does not support data parallelism
            world_size = 1
            global_rank = 0

        loader, sampler = build_hf_dataloader(
            cfg,
            global_rank,
            world_size,
            is_train=True,
        )
        print(f"rank {global_rank:3d} - loader length: {len(loader)}")

        # reset save path for each benchmark
        save_path = cfg.eval_result_file_path.lower().replace(
            ".json", f"_{benchmark}.json"
        )
        if os.path.exists(save_path):
            os.system(f"rm -fr {save_path}")
        cfg.output_path = save_path

        cfg.eval_save_folder = cfg.eval_save_folder.replace("BENCHMARK_NAME", benchmark)
        if cfg.eval_save_images:
            print(f"images will be saved to {cfg.eval_save_folder}")
            os.makedirs(cfg.eval_save_folder, exist_ok=True)

        Instance.__post_init__ = lambda self: None
        requests = []
        answerss = []
        question_typess = []
        for i, sample in enumerate(loader):

            """
            print(sample)

            {
                "idx": 787,
                "type": "2D",
                "task": "Count",
                "image": "<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=640x499 at 0x7F699C6B06D0>",
                "question": "How many wine glasss are in the image?",
                "choices": ["0", "3", "2", "1"],
                "answer": "(D)",
                "prompt": "How many wine glasss are in the image? Select from the following choices.\n(A) 0\n(B) 3\n(C) 2\n(D) 1",
                "filename": "img/2D/count/coco_287.png",
                "source": "COCO",
                "source_dataset": "COCO 2017 Validation Set",
                "source_filename": "/coco/val2017/000000412362.jpg",
                "target_class": "wine glass",
                "target_size": 1732,
                "bbox": None,
            }
            """

            idxs = [d["idx"] for d in sample]
            pil_imgs = [d["image"] for d in sample]
            questions = [d["prompt"] for d in sample]
            question_types = [d["type"].lower() for d in sample]  # 2d or 3d
            answers = [d["answer"] for d in sample]

            prompt = "Answer with the option's letter from the given choices directly."
            reqs = [
                Instance(
                    request_type="generate_until",
                    arguments=(
                        f"{question}\n{prompt}",
                        {
                            "max_new_tokens": cfg.max_gen_len,
                            "temperature": 0.0,
                            "top_p": 1.0,
                            "num_beams": 1,
                            "do_sample": False,
                            "until": ["\n\n"],
                        },
                        pil_img,
                        idx,
                        benchmark,
                        "test",
                    ),
                    metadata=(
                        benchmark,  # task_name
                        idx,  # doc_id
                        1,  # repeats
                    ),
                    idx=idx,
                    resps=[],
                    filtered_resps={},
                    task_name=benchmark,
                    doc_id=0,
                    repeats=1,
                    doc=None,
                )
                for idx, question, pil_img in zip(idxs, questions, pil_imgs)
            ]

            requests.extend(reqs)
            answerss.extend(answers)
            question_typess.extend(question_types)

            if i == 10:
                pass

        ress = EvalVlmCls.generate_until(requests)

        # wait for all processes to complete
        if world_size > 1:
            torch.distributed.barrier()

        # calculate metrics
        acc_2d = 0
        num_2d = 0
        acc_3d = 0
        num_3d = 0

        for res, ans, qtype, req in zip(ress, answerss, question_typess, requests):
            res = res.strip().lower()
            ans = ans.strip().lower()

            if len(res) == 1:
                res = f"({res})"

            if ")" in res:
                res = res.split(")")[0]
                if len(res) == 1:
                    res = "(" + res + ")"
                elif len(res) == 2:
                    res = res + ")"
                else:
                    print(
                        f"unknown response format: {res}"
                        f" - {ans}"
                        f" - {qtype}"
                        f" - {req}"
                    )
                    # treat as wrong answer
                    res = "(X)"

            elif "(" in res:
                res = res.split("(")[-1]
                if len(res) == 1:
                    res = "(" + res + ")"
                elif len(res) == 2:
                    res = "(" + res
                else:
                    print(
                        f"unknown response format: {res}"
                        f" - {ans}"
                        f" - {qtype}"
                        f" - {req}"
                    )
                    # treat as wrong answer
                    res = "(X)"

            # print(f"res: {res}", f"ans: {ans}", f"qtype: {qtype}")

            if ans in res:
                if qtype == "2d":
                    acc_2d += 1
                elif qtype == "3d":
                    acc_3d += 1
                else:
                    raise ValueError(f"unknown question type: {qtype}")

            if qtype == "2d":
                num_2d += 1
            elif qtype == "3d":
                num_3d += 1
            else:
                raise ValueError(f"unknown question type: {qtype}")

        acc_2d = acc_2d / num_2d if num_2d > 0 else 0
        acc_3d = acc_3d / num_3d if num_3d > 0 else 0
        combined_acc = (acc_2d + acc_3d) / 2

        results = {
            "2d_accuracy": acc_2d,
            "3d_accuracy": acc_3d,
            "combined_accuracy": combined_acc,
        }
        print(results)

        # dump results to json
        if master_process:
            with open(save_path, "w") as f:
                json.dump(
                    results,
                    f,
                    indent=2,
                    default=_handle_non_serializable,
                    ensure_ascii=False,
                )
            print(f"results saved to {save_path}")

            # print the first 20 lines of the result file
            with open(save_path, "r") as f:
                lines = f.readlines()
                print("".join(lines[:20]))

            if cfg.eval_res_upload_to_hf:
                upload_to_hf(cfg, save_path)

    # print all results in a table
    if master_process:
        paths = [
            cfg.eval_result_file_path.lower().replace(".json", f"_{benchmark}.json")
            for benchmark in cfg.benchmark
        ]
        print_in_one_row(paths)

    if cfg.enable_wandb and master_process:
        wandb.finish()

    destroy_process_group()


def print_in_one_row(paths: list[str]):
    # init vars
    mme_binary_res_cog = 0
    mme_binary_res_per = 0
    mme_res_cog = 0
    mme_res_per = 0
    pope_adv_binary_acc = 0
    pope_adv_binary_f1 = 0
    pope_pop_binary_acc = 0
    pope_pop_binary_f1 = 0
    pope_random_binary_acc = 0
    pope_random_binary_f1 = 0
    pope_adv_acc = 0
    pope_adv_f1 = 0
    pope_pop_acc = 0
    pope_pop_f1 = 0
    pope_random_acc = 0
    pope_random_f1 = 0
    seedbench_all = 0
    seedbench_img = 0
    seedbench_vid = 0
    mmvet_res = 0
    llave_wild_acc = 0
    mmbench_en_acc = 0
    cvbench_acc_2d = 0
    cvbench_acc_3d = 0
    cvbench_acc_combined = 0
    gqa_acc = 0
    viz_acc = 0
    textvqa_acc = 0
    docvqa_acc = 0
    chartqa_overall_acc = 0
    chartqa_human_split_acc = 0
    chartqa_aug_split_acc = 0
    infovqa_acc = 0
    ai2d_acc = 0

    paths.sort()
    for path in paths:
        print(f"loading results from {path} ...")
        results = json.loads(open(path, "r").read())

        if "cvbench" in path:
            cvbench_acc_2d = results["2d_accuracy"] * 100.0
            cvbench_acc_3d = results["3d_accuracy"] * 100.0
            cvbench_acc_combined = results["combined_accuracy"] * 100.0

        if "llava_in_the_wild" in path:
            llave_wild_acc = results["results"]["llava_in_the_wild"][
                "gpt_eval_llava_all,none"
            ]

        if "mmbench_en_dev" in path:
            mmbench_en_acc = results["results"]["mmbench_en_dev"]["gpt_eval_score,none"]

        if "mme" in path and "binary" not in path:
            mme_res_cog = results["results"]["mme"]["mme_cognition_score,none"]
            mme_res_per = results["results"]["mme"]["mme_perception_score,none"]

        if "mme_binary" in path:
            mme_binary_res_cog = results["results"]["mme_binary"][
                "mme_cognition_score,none"
            ]
            mme_binary_res_per = results["results"]["mme_binary"][
                "mme_perception_score,none"
            ]

        if "mmvet" in path:
            mmvet_res = results["results"]["mmvet"]["gpt_eval_score,none"]

        if "pope_adv" in path and "binary" not in path:
            pope_adv_acc = results["results"]["pope_adv"]["pope_accuracy,none"] * 100.0
            pope_adv_f1 = results["results"]["pope_adv"]["pope_f1_score,none"] * 100.0

        if "pope_pop" in path and "binary" not in path:
            pope_pop_acc = results["results"]["pope_pop"]["pope_accuracy,none"] * 100.0
            pope_pop_f1 = results["results"]["pope_pop"]["pope_f1_score,none"] * 100.0

        if "pope_random" in path and "binary" not in path:
            pope_random_acc = (
                results["results"]["pope_random"]["pope_accuracy,none"] * 100.0
            )
            pope_random_f1 = (
                results["results"]["pope_random"]["pope_f1_score,none"] * 100.0
            )

        if "pope_adv_binary" in path:
            pope_adv_binary_acc = (
                results["results"]["pope_adv_binary"]["pope_accuracy,none"] * 100.0
            )
            pope_adv_binary_f1 = (
                results["results"]["pope_adv_binary"]["pope_f1_score,none"] * 100.0
            )

        if "pope_pop_binary" in path:
            pope_pop_binary_acc = (
                results["results"]["pope_pop_binary"]["pope_accuracy,none"] * 100.0
            )
            pope_pop_binary_f1 = (
                results["results"]["pope_pop_binary"]["pope_f1_score,none"] * 100.0
            )

        if "pope_random_binary" in path:
            pope_random_binary_acc = (
                results["results"]["pope_random_binary"]["pope_accuracy,none"] * 100.0
            )
            pope_random_binary_f1 = (
                results["results"]["pope_random_binary"]["pope_f1_score,none"] * 100.0
            )

        if "seedbench" in path:
            seedbench_all = results["results"]["seedbench"]["seed_all,none"] * 100.0
            seedbench_img = results["results"]["seedbench"]["seed_image,none"] * 100.0
            seedbench_vid = results["results"]["seedbench"]["seed_video,none"] * 100.0

        if "gqa" in path:
            gqa_acc = results["results"]["gqa"]["exact_match,none"] * 100.0

        if "vizwiz_vqa_val" in path:
            viz_acc = results["results"]["vizwiz_vqa_val"]["exact_match,none"] * 100.0

        if "textvqa_val" in path:
            textvqa_acc = results["results"]["textvqa_val"]["exact_match,none"] * 100.0

        if "docvqa_val" in path:
            docvqa_acc = results["results"]["docvqa_val"]["anls,none"] * 100.0

        if "chartqa" in path:
            chartqa_overall_acc = (
                results["results"]["chartqa"]["relaxed_overall,none"] * 100.0
            )
            chartqa_human_split_acc = (
                results["results"]["chartqa"]["relaxed_human_split,none"] * 100.0
            )
            chartqa_aug_split_acc = (
                results["results"]["chartqa"]["relaxed_augmented_split,none"] * 100.0
            )

        if "infovqa_val" in path:
            infovqa_acc = results["results"]["infovqa_val"]["anls,none"] * 100.0

        if "ai2d" in path:
            ai2d_acc = (
                results["results"]["ai2d"]["exact_match,flexible-extract"] * 100.0
            )

    # pope results
    print(
        f"\nnormal POPE:\t"
        f"adv:\t{pope_adv_acc:.2f}\t{pope_adv_f1:.2f}\t"
        f"pop:\t{pope_pop_acc:.2f}\t{pope_pop_f1:.2f}\t"
        f"random:\t{pope_random_acc:.2f}\t{pope_random_f1:.2f}\n"
        f"binary POPE:\t"
        f"adv:\t{pope_adv_binary_acc:.2f}\t{pope_adv_binary_f1:.2f}\t"
        f"pop:\t{pope_pop_binary_acc:.2f}\t{pope_pop_binary_f1:.2f}\t"
        f"random:\t{pope_random_binary_acc:.2f}\t{pope_random_binary_f1:.2f}\t"
    )

    # average pope acc. and f1
    avg_pope_acc = (pope_adv_acc + pope_pop_acc + pope_random_acc) / 3
    avg_pope_f1 = (pope_adv_f1 + pope_pop_f1 + pope_random_f1) / 3
    avg_pope_binary_acc = (
        pope_adv_binary_acc + pope_pop_binary_acc + pope_random_binary_acc
    ) / 3
    avg_pope_binary_f1 = (
        pope_adv_binary_f1 + pope_pop_binary_f1 + pope_random_binary_f1
    ) / 3

    print(
        f"{mme_binary_res_cog:.2f}\t{mme_binary_res_per:.2f}\t"
        f"{mme_res_cog:.2f}\t{mme_res_per:.2f}\t"
        f"{avg_pope_binary_acc:.2f}\t{avg_pope_binary_f1:.2f}\t"
        f"{avg_pope_acc:.2f}\t{avg_pope_f1:.2f}\t"
        f"{seedbench_all:.2f}\t{seedbench_img:.2f}\t{seedbench_vid:.2f}\t"
        f"{mmvet_res:.2f}\t"
        f"{llave_wild_acc:.2f}\t"
        f"{mmbench_en_acc:.2f}\t"
        f"{cvbench_acc_2d:.2f}\t{cvbench_acc_3d:.2f}\t{cvbench_acc_combined:.2f}\t"
        f"{gqa_acc:.2f}\t"
        f"{viz_acc:.2f}\t"
        f"{textvqa_acc:.2f}\t"
        f"{docvqa_acc:.2f}\t"
        f"{chartqa_overall_acc:.2f}\t"
        f"{chartqa_human_split_acc:.2f}\t"
        f"{chartqa_aug_split_acc:.2f}\t"
        f"{infovqa_acc:.2f}\t"
        f"{ai2d_acc:.2f}\t"
    )


if __name__ == "__main__":
    cfg = load_config(sys.argv)
    main(cfg)
