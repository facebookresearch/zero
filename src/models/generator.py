# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple, Any

import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
import PIL
import numpy as np

from copy import deepcopy
from tqdm import tqdm
from torchvision.transforms import v2 as T

from lm_eval.api.model import LM
from lmms_eval.api.model import lmms as LMMS

from loader import expand2square, process_anyres_image
from distributed.torchshard import _gather

"""
toolkit for evaluating vision-language models
"""


class EvalVLM(LMMS):
    """
    base class for evaluating vision-language models
    https://github.com/EvolvingLMMs-Lab/lmms-eval
    """

    def __init__(self, cfg: object, model: Any, tokenizer: Any, image_transform: Any):
        super().__init__()
        self.cfg = cfg
        self.image_transform = image_transform

        self.model = model.module if hasattr(model, "module") else model
        self.tokenizer = tokenizer

        # max input content length
        self.max_ctx_len = self.model.decoder.config.max_position_embeddings

        # enbale vllm
        if self.cfg.eval_enable_vllm:

            from vllm import LLM  # lazy import vllm

            self.patch_vllm(self.cfg.local_rank)
            state_dict = self.get_decoder_state_dict()
            self.llm = LLM(
                model=cfg.decoder_hf_model_name,
                tensor_parallel_size=cfg.eval_vllm_tensor_parallel_size,
                distributed_executor_backend="external_launcher",  # with torchrun
                dtype=cfg.dtype,
                gpu_memory_utilization=cfg.eval_vllm_gpu_memory_utilization,
                seed=cfg.seed,
                enable_prompt_embeds=True,  # vllm >= 0.9.0
                enforce_eager=True,
                disable_custom_all_reduce=(
                    True if cfg.eval_vllm_tensor_parallel_size > 1 else False
                ),
                max_model_len=cfg.eval_vllm_max_model_len,
            )
            if cfg.enable_translator:
                self.build_vllm_surrogate_model()
            self.overwrite_vllm_model_params(state_dict)
            self.llm.set_tokenizer(self.tokenizer)
            print(f"sampling engine: vllm")
        else:
            print(f"sampling engine: transformer")

    """
    submodules for vllm
    """

    def get_decoder_state_dict(self):
        # depends on the model family
        raise NotImplementedError

    def overwrite_vllm_model_params(self, state_dict: dict):
        # load the state_dict to vllm model
        msgs = self.llm.llm_engine.model_executor.driver_worker.worker.model_runner.model.load_state_dict(
            state_dict, strict=bool(1)
        )
        missing_keys, unexpected_keys = msgs.missing_keys, msgs.unexpected_keys
        print(f"vllm model load state_dict from hf model parameters: {msgs}")
        if len(unexpected_keys) > 0:
            raise RuntimeError(
                f"unexpected keys when overriding vllm model parameters: {unexpected_keys}"
            )

        del state_dict

    def build_vllm_surrogate_model(self):
        print("start converting vllm model to surrogate model")
        from build_decoder import rename_layer_to_translator
        from build_decoder import remove_layer_from_model

        model = (
            self.llm.llm_engine.model_executor.driver_worker.worker.model_runner.model
        )

        translators_range = deepcopy(self.cfg.translators_range)
        for range_idx, trans_range in enumerate(translators_range):
            trans_range.sort()

            assert (
                np.sum(np.diff(trans_range) == 1) == len(trans_range) - 1
            ), "translators_range should be consecutive without skipping layers"
            assert np.all(np.array(trans_range) >= 0)
            assert (
                0
                <= min(trans_range)
                < max(trans_range)
                < model.config.num_hidden_layers
            )
            assert self.cfg.num_translators <= len(trans_range)

            index_offset = 0
            if range_idx > 0:
                index_offset += sum(len(translators_range[i]) for i in range(range_idx))

            # rename the layer to translator
            for i in range(self.cfg.num_translators):
                translator_layer_id = trans_range[i]
                print(
                    f"- mv layer {translator_layer_id:>3} to translator "
                    f"-- actual layer {translator_layer_id - index_offset:>3}"
                )
                model = rename_layer_to_translator(
                    model, translator_layer_id - index_offset
                )
                trans_range.pop(i)

            # remove the rest of the layers
            _empty_str = "".join([" "] * 14)
            for layer_id in trans_range:
                print(
                    f"- rm layer {layer_id:>3} {_empty_str}"
                    f"-- actual layer {layer_id - index_offset:>3}"
                )
                model = remove_layer_from_model(model, layer_id - index_offset)
                index_offset += 1

        self.llm.llm_engine.model_executor.driver_worker.worker.model_runner.model = (
            model
        )
        pass

    def patch_vllm(self, local_rank: int):
        from vllm.worker.worker import Worker

        ori_worker_init = Worker.__init__

        def new_worker_init(self, *args, **kwargs):
            # call the original __init__ method
            ori_worker_init(self, *args, **kwargs)

            # replace the local rank
            self.device = torch.device(f"cuda:{local_rank}")
            print(self.parallel_config)

        # replace the original __init__ method with the new one
        Worker.__init__ = new_worker_init

    """
    properties
    """

    @property
    def max_new_tokens(self):
        return self.cfg.max_gen_len

    @property
    def batch_size(self):
        return self.cfg.batch_size

    @property
    def device(self):
        return self.cfg.device

    @property
    def eot_token_id(self):
        # depends on the model family
        raise NotImplementedError

    @property
    def rank(self):
        return self.cfg.global_rank

    @rank.setter
    def rank(self, value):
        self.cfg.global_rank = value

    @property
    def world_size(self):
        return self.cfg.world_size

    @world_size.setter
    def world_size(self, value):
        self.cfg.world_size = value

    """
    main methods for evaluation
    """

    def loglikelihood(self, requests, mute_pbar=False) -> List[Tuple[float, bool]]:
        raise NotImplementedError

    def loglikelihood_rolling(self, requests) -> List[float]:
        raise NotImplementedError

    def generate_until_multi_round(self, requests) -> List[List[str]]:
        raise NotImplementedError

    def generate_until(self, requests, mute_pbar=False) -> List[str]:
        res = []

        reqs = [req.args for req in requests]

        # sort requests by context length
        def _collate_fn(requests) -> List[int]:
            """
            return the indices of the requests in the order of the context lengths
            """
            ctx_lengths = [-len(self._encode_text(req[0])) for req in requests]
            return sorted(range(len(requests)), key=lambda i: ctx_lengths[i])

        reqs, reorder_indices = self._reorder_requests(reqs, _collate_fn)

        # loop over the requests
        iter_reqs = range(0, len(reqs), self.batch_size)
        pbar = (
            iter_reqs
            if mute_pbar
            else tqdm(
                iter_reqs,
                total=len(iter_reqs),
                desc=f"rank {self.rank:2d} / {self.world_size}",
            )
        )

        if self.batch_size != 1:
            raise ValueError("batch size must be 1 for evaluation")

        # logging important information
        for i in pbar:
            batch_reqs = reqs[i : i + self.batch_size]

            # unpack the requests
            contexts = [req[0] for req in batch_reqs]
            gen_kwargs = [req[1] for req in batch_reqs]
            doc_to_visuals = [req[2] for req in batch_reqs]
            doc_ids = [req[3] for req in batch_reqs]
            tasks = [req[4] for req in batch_reqs]
            splits = [req[5] for req in batch_reqs]

            assert len(tasks) == 1, "only one task is supported"
            task_name = tasks[0]

            max_new_tokens = [
                int(kwarg.get("max_new_tokens", -1)) for kwarg in gen_kwargs
            ]
            has_max_new_tokens = all([t != -1 for t in max_new_tokens])
            max_new_tokens = max(max_new_tokens) if has_max_new_tokens else 2048

            # fisrt load sampling parameters from toolkit
            temperature = float(gen_kwargs[0].get("temperature", 0.0))
            top_p = float(gen_kwargs[0].get("top_p", 1.0))
            do_sample = bool(gen_kwargs[0].get("do_sample", False))

            # gather
            sampling_params = self._overwrite_sampling_params(
                {
                    "temperature": temperature,
                    "top_p": top_p,
                    "do_sample": do_sample,
                    "max_new_tokens": max_new_tokens,
                }
            )

            # get visuals
            visuals = []
            for doc_to_visual, doc_id, task, split in zip(
                doc_to_visuals, doc_ids, tasks, splits
            ):
                if isinstance(doc_to_visual, PIL.Image.Image):
                    visuals.append((doc_id, [doc_to_visual]))
                else:
                    visuals.append(
                        (doc_id, doc_to_visual(self.task_dict[task][split][doc_id]))
                    )
            visuals, doc_idss = self._flatten(visuals)

            # encode visuals
            imgs = []
            for i, visual in enumerate(visuals):
                if isinstance(visual, str) and os.path.exists(visual):
                    # assuming visual is a path to a video
                    visual = self._encode_video(visual)
                    for img in visual:
                        imgs.append(img)
                else:
                    img = self._encode_image(
                        self.cfg,
                        visual,
                        doc_id=doc_idss[i],
                    )  # [1, 3, h, w]
                    imgs.append(img)
            assert len(imgs) == len(visuals)

            contexts, imgs = self._pre_processing_requests(
                contexts,
                imgs,
                benchmark=task_name,
                print_info=self.cfg.eval_print_info,
            )

            input_embds, input_ids, attention_mask = self._construct_model_inputs(
                contexts, imgs
            )

            # forward pass
            if self.cfg.eval_enable_vllm:
                # https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/llm.py#L385
                o = self.llm.generate(
                    {"prompt_embeds": input_embds},
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )
                generated_ids = [o[0].outputs[0].token_ids]
            else:
                # https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationConfig
                generated_ids = self.model.decoder.generate(
                    inputs_embeds=input_embds,
                    attention_mask=attention_mask,
                    use_cache=True,
                    **sampling_params,
                )  # [bs, seq_len]

            # decode the generated ids
            orig_strs, out_strs, thinking_traces = self._post_processing_response(
                generated_ids,
                benchmark=task_name,
                print_info=self.cfg.eval_print_info,
            )
            res.extend(out_strs)

            # for debugging
            if self.cfg.eval_print_info:
                batch_id = 0
                t = input_ids[batch_id].tolist()  # np array to list
                print(
                    f"--- rank: {self.rank:2d} / {self.world_size}\n"
                    f"--- doc id: {doc_idss[batch_id]}\n"
                    f"--- sampling params: {sampling_params}\n"
                    f"--- input token ids: {t}\n"
                    f"--- input token txt: {self.tokenizer.decode(t)}\n"
                    f"--- thinking trace: {thinking_traces[batch_id]}\n"
                    f"--- model pred: {out_strs[batch_id]}\n"
                )

            torch.cuda.empty_cache()

            if i == 10:
                pass

        # restore the original order
        original_order_indices = self._restore_original_order(reorder_indices)
        res = [res[i] for i in original_order_indices]
        return res

    """
    helper functions
    """

    def _reorder_requests(self, requests, collate_fn) -> Tuple[List, List[int]]:
        reorder_indices = collate_fn(requests)
        return [requests[i] for i in reorder_indices], reorder_indices

    def _restore_original_order(self, sorted_indices: List[int]) -> List[int]:
        original_order = [None] * len(sorted_indices)
        for i, index in enumerate(sorted_indices):
            original_order[index] = i
        return original_order

    def _construct_model_inputs(self, contexts, images):
        # depends on the model family
        raise NotImplementedError

    def _encode_text(self, text: str) -> List[int]:
        # depends on the model family
        raise NotImplementedError

    def _pre_processing_requests(self, contexts, imgs, benchmark, print_info=False):
        if "seedbench" in benchmark:
            # only use middle frame of the video
            imgs = [imgs[len(imgs) // 2]]
            print(f"only use the middle frame of the video for benchmark: {benchmark}")
        return contexts, imgs

    def _post_processing_response(self, generated_ids, benchmark, print_info=False):
        # depends on the model family
        raise NotImplementedError

    def _encode_image(self, cfg, image, doc_id=None):
        x = image.convert("RGB")

        if cfg.eval_save_images and doc_id is not None:
            save_root = cfg.eval_save_folder
            x.save(f"{save_root}/{doc_id}.jpg")

        bg_value = tuple(int(v * 255) for v in self.image_transform.image_mean)

        if self.cfg.image_ratio_type == "pad":
            x = expand2square(x, bg_value)
            x = self.image_transform(x, return_tensors="pt")[
                "pixel_values"
            ]  # [patches, 3, h, w], where patches is 1
        elif self.image_ratio_type == "anyres":
            # control the number of total image tokens
            if (
                self.max_input_image_size is not None
                or self.min_input_image_size is not None
            ):
                ori_size = image.size
                image = T.functional.resize(
                    image,
                    size=self.min_input_image_size,
                    max_size=self.max_input_image_size,
                )
                # print(f"resized image from {ori_size} to {image.size}")
            x = process_anyres_image(
                x,
                self.image_transform,
                self.anyres_grid_pinpoints,
                self.bg_value,
            )
        else:
            raise ValueError(f"unknown image_ratio_type: {self.cfg.image_ratio_type}")

        return x

    def _encode_video(self, video):
        pass

    def _flatten(self, input):
        """
        input = [(doc_id, [image]), ...]
        """
        img_list = []
        doc_idss = []
        for pair in input:
            doc_id, images = pair
            for img in images:
                img_list.append(img)
                doc_idss.append(doc_id)
        return img_list, doc_idss


"""
toolkit for evaluating language models
"""


class EvalLM(LM):
    """
    base class for evaluating language models
    https://github.com/EleutherAI/lm-evaluation-harness
    """

    def __init__(self, cfg: object, model: Any, tokenizer: Any):
        super().__init__()
        self.cfg = cfg

        self.model = model.module if hasattr(model, "module") else model
        self.model.eval()
        self.model.requires_grad_(False)
        self.tokenizer = tokenizer

        # max input content length
        self.max_ctx_len = self.model.decoder.config.max_position_embeddings

    """
    properties
    """

    @property
    def max_new_tokens(self):
        return self.cfg.max_gen_len

    @property
    def batch_size(self):
        return self.cfg.batch_size

    @property
    def device(self):
        return self.cfg.device

    @property
    def rank(self):
        return self.cfg.global_rank

    @property
    def world_size(self):
        return self.cfg.world_size

    @property
    def eot_token_id(self):
        # depends on the model family
        raise NotImplementedError

    """
    three main funcs for evaluation
    """

    def loglikelihood(self, requests, mute_pbar=False) -> List[Tuple[float, bool]]:
        """
        https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/huggingface.py#L545
        """
        new_reqs = []

        iter_reqs = enumerate([req.args for req in requests])
        pbar = (
            iter_reqs
            if mute_pbar
            else tqdm(
                iter_reqs,
                total=len(requests),
                desc=f"rank {self.rank:2d} / {self.world_size}",
            )
        )
        for i, (context, continuation) in pbar:
            if context == "":
                # end of text as context
                ctx_tokens, cnt_tokens = [self.eot_token_id], self._encode_text(
                    continuation
                )
            else:
                ctx_tokens, cnt_tokens = self._encode_pair(context, continuation)

            new_reqs.append(((context, continuation), ctx_tokens, cnt_tokens))

            if i == 1:
                pass

        return self._loglikelihood_tokens(new_reqs, mute_pbar=mute_pbar)

    def loglikelihood_rolling(self, requests) -> List[float]:
        print("loglikelihood_rolling", requests)
        raise NotImplementedError

    def generate_until(self, requests, mute_pbar=False) -> List[str]:
        res = []

        # max len for inputs = max length, minus room to generate the max new tokens
        max_ctx_len = self.max_ctx_len - self.max_new_tokens

        reqs = [req.args for req in requests]

        # sort requests by context length
        def _collate_fn(requests) -> List[int]:
            """
            return the indices of the requests in the order of the context lengths
            """
            ctx_lengths = [-len(self._encode_text(req[0])) for req in requests]
            return sorted(range(len(requests)), key=lambda i: ctx_lengths[i])

        reqs, reorder_indices = self._reorder_requests(reqs, _collate_fn)

        # loop over the requests
        if self.batch_size != 1:
            raise ValueError("batch size must be 1 for evaluation")

        iter_reqs = range(0, len(reqs), self.batch_size)
        pbar = (
            iter_reqs
            if mute_pbar
            else tqdm(
                iter_reqs,
                total=len(iter_reqs),
                desc=f"rank {self.rank:2d} / {self.world_size}",
            )
        )
        for i in pbar:
            batch_reqs = reqs[i : i + self.batch_size]
            contexts = [req[0] for req in batch_reqs]  # (context, gen_kwargs)

            if self.cfg.eval_enable_chat_template:
                contexts = [self._apply_chat_template(ctx, "") for ctx in contexts]
            ctx_tokens = [self._encode_text(ctx) for ctx in contexts]
            ctx_tokens = [t[-max_ctx_len:] for t in ctx_tokens]  # [bs, seq_len]

            ctx_tokens = torch.tensor(ctx_tokens, device=self.device)  # [bs, seq_len]
            attn_mask = torch.ones_like(ctx_tokens)

            output = self.model.decoder.generate(
                input_ids=ctx_tokens,
                attention_mask=attn_mask,
                max_length=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

            cont_toks = output.cpu().numpy().tolist()
            cont_toks = [o[:-1] for o in cont_toks]  # remove the last stop token
            cont_strs = [self.tokenizer.decode(t) for t in cont_toks]
            res.extend(cont_strs)

            if i == 1:
                pass

        # restore the original order
        original_order_indices = self._restore_original_order(reorder_indices)
        res = [res[i] for i in original_order_indices]
        return res

    """
    helper functions
    """

    def _apply_chat_template(self, context, continuation):
        print("_apply_chat_template:", "context:", context, "cont:", continuation)
        raise NotImplementedError

    def _reorder_requests(self, requests, collate_fn) -> Tuple[List, List[int]]:
        reorder_indices = collate_fn(requests)
        return [requests[i] for i in reorder_indices], reorder_indices

    def _restore_original_order(self, sorted_indices: List[int]) -> List[int]:
        original_order = [None] * len(sorted_indices)
        for i, index in enumerate(sorted_indices):
            original_order[index] = i
        return original_order

    def _encode_text(self, text: str) -> List[int]:
        # depends on the model family
        raise NotImplementedError

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        if self.cfg.eval_enable_chat_template:
            context, continuation = self._apply_chat_template(context, continuation)

        all_tokens = self._encode_text(context + continuation)
        ctx_tokens = self._encode_text(context)
        cnt_tokens = all_tokens[len(ctx_tokens) :]
        return ctx_tokens, cnt_tokens

    def _loglikelihood_tokens(self, requests, mute_pbar=False):
        res = []

        # sort requests by context + continuation length
        def _collate_fn(requests) -> List[int]:
            """
            return the indices of the requests in the order of the context lengths
            """
            lengths = [-len(req[1] + req[2]) for req in requests]
            return sorted(range(len(requests)), key=lambda i: lengths[i])

        requests, reorder_indices = self._reorder_requests(requests, _collate_fn)

        iter_reqs = enumerate(requests)
        pbar = iter_reqs if mute_pbar else tqdm(iter_reqs, total=len(requests))
        for i, ((context, continuation), ctx_tokens, cnt_tokens) in pbar:
            # prepare input tokens
            tokens = [ctx_tokens + cnt_tokens]
            tokens = torch.tensor(tokens, device=self.device)  # [bs, seq_len]

            # how this all works (illustrated on a causal decoder-only setup):
            #          CTX      CONT
            # inp    0 1 2 3 | 4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
            # model   \                 \
            # logits   1 2 3 | 4 5 6 7 8 9   <- the ctx half gets tossed out by the
            # cont_toks        4 5 6 7 8 9      [:, -len(cnt_tokens):, :self.vocab_size] slice

            # truncate the context by left if it's too long
            tokens = tokens[:, -(self.max_ctx_len + 1) :][:, :-1]

            # lengths
            bs, seq_len = tokens.shape  # [bs, seq_len]
            ctx_len, cnt_len = len(ctx_tokens), len(cnt_tokens)

            # forward pass
            output = self.model.decoder(input_ids=tokens, use_cache=False)
            logits = output["logits"].float()
            logits = F.log_softmax(logits, dim=-1)

            # ignore the context part, only consider the continuation part
            logits = logits[:, -cnt_len:, :]

            # check if per-token argmax is exactly the same as the continuation
            top1_toks = torch.argmax(logits, dim=-1)
            cont_toks = torch.tensor(cnt_tokens, device=self.device).unsqueeze(0)
            max_equal = (top1_toks == cont_toks).all()

            # compute log likelihood
            logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(
                -1
            )  # [1, seq_len]

            # answer: (log prob, is-exact-match)
            answer = (float(logits.sum()), bool(max_equal))

            # append to results
            res.append(answer)

            if i == 1:
                pass

        # restore the original order
        original_order_indices = self._restore_original_order(reorder_indices)
        res = [res[i] for i in original_order_indices]
        return res
