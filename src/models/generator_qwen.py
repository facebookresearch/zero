# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Any

import torch

from torch.nn.utils.rnn import pad_sequence
from copy import deepcopy

from models.generator import EvalLM, EvalVLM
from distributed.torchshard import _gather, _split


class Qwen3EvalLM(EvalLM):
    @property
    def eot_token_id(self):
        return self.tokenizer.convert_tokens_to_ids("<|im_end|>")

    def _encode_text(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)


class Qwen3EvalVLM(EvalVLM):
    def __init__(self, cfg: object, model: Any, tokenizer: Any, image_transform: Any):
        super().__init__(cfg, model, tokenizer, image_transform)

        chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% if not enable_thinking is defined %}{% set enable_thinking = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}{% if enable_thinking %}{{ '<|think|>\n' }}{% endif %}"
        self.tokenizer = deepcopy(self.tokenizer)
        self.tokenizer.chat_template = chat_template

        # enbale vllm
        if self.cfg.eval_enable_vllm:
            # reset the tokenizer
            self.llm.set_tokenizer(self.tokenizer)

        self.wte = self._get_wte()

    """
    submodules for vllm
    """

    def get_decoder_state_dict(self):
        """
        supported:
            tensor parallel size > 1
        """

        tp_size = self.cfg.eval_vllm_tensor_parallel_size

        # step 1. offload the state_dict to cpu to save gpu memory
        # and filter out the encoder parameters
        state_dict = {}
        for n, p in self.model.decoder.named_parameters():

            if tp_size > 1:

                """
                row parallel layers
                """

                if "embed_tokens" in n:
                    state_dict[n] = _split(p, dim=0).cpu()
                    continue

                if "lm_head.weight" in n:
                    state_dict[n] = _split(p, dim=0).cpu()
                    continue

                """
                col parallel layers
                """

                if "self_attn.o_proj" in n:
                    state_dict[n] = _split(p, dim=1).cpu()
                    continue

                if "mlp.down_proj" in n:
                    state_dict[n] = _split(p, dim=1).cpu()
                    continue

                pass

            # merge model.layers.{layer_id}.mlp.gate_proj.weight
            #       model.layers.{layer_id}.mlp.up_proj.weight
            # into  model.layers.{layer_id}.mlp.gate_up_proj.weight
            if "mlp" in n and (
                n.endswith(".gate_proj.weight") or n.endswith(".up_proj.weight")
            ):
                ori_n = n

                ns = n.split(".")
                prefix = ".".join(ns[: -ns.index("mlp")])
                n = prefix + ".mlp.gate_up_proj.weight"

                if "gate_proj" in ori_n:
                    if tp_size > 1:
                        state_dict[n] = _split(p, dim=0).cpu()
                    else:
                        state_dict[n] = p.cpu()
                elif "up_proj" in ori_n:
                    if tp_size > 1:
                        p = _split(p, dim=0).cpu()
                    else:
                        p = p.cpu()

                    state_dict[n] = torch.cat(
                        [
                            state_dict[n],  # gate_proj
                            p,  # up_proj
                        ],
                        dim=0,
                    )
                else:
                    raise RuntimeError(
                        f"unexpected parameter name for mlp: {n}, {p.shape}, {ori_n}"
                    )
            # merge model.layers.{layer_id}.self_attn.q_proj.weight
            #       model.layers.{layer_id}.self_attn.k_proj.weight
            #       model.layers.{layer_id}.self_attn.v_proj.weight
            # into  model.layers.{layer_id}.self_attn.qkv_proj.weight
            elif "self_attn" in n and (
                n.endswith(".q_proj.weight")
                or n.endswith(".k_proj.weight")
                or n.endswith(".v_proj.weight")
            ):
                ori_n = n

                ns = n.split(".")
                prefix = ".".join(ns[: -ns.index("self_attn")])
                n = prefix + ".self_attn.qkv_proj.weight"

                if "q_proj" in ori_n:
                    if tp_size > 1:
                        state_dict[n] = _split(p, dim=0).cpu()
                    else:
                        state_dict[n] = p.cpu()
                elif "k_proj" in ori_n:

                    if tp_size > 1:
                        p = _split(p, dim=0).cpu()
                    else:
                        p = p.cpu()

                    state_dict[n] = torch.cat(
                        [
                            state_dict[n],  # q_proj
                            p,  # k_proj
                        ],
                        dim=0,
                    )
                elif "v_proj" in ori_n:
                    if tp_size > 1:
                        p = _split(p, dim=0).cpu()
                    else:
                        p = p.cpu()

                    state_dict[n] = torch.cat(
                        [
                            state_dict[n],  # qk_proj
                            p,  # v_proj
                        ],
                        dim=0,
                    )
                else:
                    raise RuntimeError(
                        f"unexpected parameter name for self_attn: {n}, {p.shape}, {ori_n}"
                    )
            else:
                state_dict[n] = p.cpu()

        # if model is tied, copy embed_tokens.weight to lm_head.weight
        if (
            self.model.decoder.config.tie_word_embeddings
            and "lm_head.weight" not in state_dict
        ):
            state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]

            # TODO: currently, we only use cfg.qwen_model_scale to determine
            # wether to split the lm_head.weight for vllm
            is_sharded_lm_head = float(self.cfg.qwen_model_scale) > 4

            if tp_size > 1 and is_sharded_lm_head:
                t = torch.tensor(state_dict["lm_head.weight"], device=self.device)
                state_dict["lm_head.weight"] = _split(t, dim=0).cpu()

        # we need to release the memory of the model
        # when the model is too large
        # need to make sure vllm has enough gpu memory to load the model
        del self.model.decoder

        return state_dict

    """
    submodules for generation
    """

    @property
    def eot_token_id(self):
        return self.tokenizer.convert_tokens_to_ids("<|eot_id|>")

    def _encode_text(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def _decode_text(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def _overwrite_sampling_params(self, sampling_params: dict):
        t = sampling_params["temperature"]
        if t > 0.0 and not sampling_params["do_sample"]:
            print(
                f"warning: temperature is {t}, "
                f"set to 0.0 instead due to do_sample=False"
            )
            sampling_params["temperature"] = 0.0

        # default values
        sampling_params["top_k"] = 0
        sampling_params["min_p"] = 0.0
        sampling_params["num_beams"] = 1

        if self.cfg.eval_enable_thinking_mode:
            # see https://huggingface.co/Qwen/Qwen3-4B
            sampling_params["do_sample"] = True
            sampling_params["temperature"] = 0.6
            sampling_params["top_p"] = 0.95
            sampling_params["top_k"] = 20
            sampling_params["min_p"] = 0.0
            del sampling_params["num_beams"]

        if self.cfg.eval_force_to_set_max_gen_len:
            sampling_params["max_new_tokens"] = self.max_new_tokens

        if self.cfg.eval_enable_vllm:
            # https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py#L117
            from vllm import SamplingParams

            sampling_params = SamplingParams(
                temperature=sampling_params["temperature"],
                top_p=sampling_params["top_p"],
                top_k=sampling_params["top_k"],
                min_p=sampling_params["min_p"],
                max_tokens=sampling_params["max_new_tokens"],
            )

        return sampling_params

    def _get_wte(self):
        if self.cfg.eval_enable_vllm:
            wte = (
                self.llm.llm_engine.model_executor.driver_worker.worker.model_runner.model.model.embed_tokens
            )
        else:
            wte = self.model.decoder.get_input_embeddings()
        print(f"wte: {wte}")
        return wte

    @torch.no_grad()
    def _construct_model_inputs(self, contexts: List[str], images: List[torch.Tensor]):
        img_token_str = "<|image_pad|>"
        img_id = self.tokenizer.convert_tokens_to_ids(img_token_str)
        pad_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")

        # encode imgs
        imgs: List[torch.Tensor] = [img.to(self.device) for img in images]
        embds_imgs: List[torch.Tensor] = self.model._forward(imgs)

        # encode questions
        input_ids = []
        for i, conv in enumerate(contexts):
            input_id = []

            if self.cfg.qwen_model_version == "3" and not self.cfg.enable_translator:
                # we disable reasoning ability for Qwen 3 in default
                # because we have no reasoning benchmark in this paper
                if not self.cfg.eval_enable_thinking_mode:
                    conv = conv.replace(
                        "First please perform reasoning, and think step by step to provide best answer to the following question:",
                        "",
                    ).strip()
                    conv += " /no_think"
                else:
                    conv += " /think"

            usr_text = "<image>\n" + conv
            usr_role = "user"
            has_img = bool(1)

            if has_img:
                spans = usr_text.split("<image>")
                new_spans = []
                for i in range(len(spans) - 1):
                    new_spans.append(spans[i])
                    image_placeholder = img_token_str
                    new_spans.append(image_placeholder)
                new_spans.append(spans[-1])
                usr_text = "".join(new_spans)

            # NOTE: for debugging
            # usr_text = "Hello, hello!? What is the meaning of life? /no_think"

            usr_turn = [{"role": usr_role, "content": usr_text}]
            encode_id = self.tokenizer.apply_chat_template(
                usr_turn,
                add_generation_prompt=True,
                enable_thinking=self.cfg.eval_enable_thinking_mode,
            )
            input_id += encode_id

            # batch token ids
            input_ids.append(input_id)

        # construct input embeddings
        input_embds = []
        targets_ids = []

        for batch_idx, input_id in enumerate(input_ids):
            if img_id in input_id:
                img_idx = input_id.index(img_id)
                n_img_tokens = embds_imgs[batch_idx].shape[0]

                inp1_ids = input_id[:img_idx]
                inp2_ids = input_id[img_idx + 1 :]

                inp1_ids = torch.tensor(inp1_ids, dtype=torch.long, device=self.device)
                inp2_ids = torch.tensor(inp2_ids, dtype=torch.long, device=self.device)

                embds_inp1 = self.wte(inp1_ids)  # [seq_len, dim]
                embds_inp2 = self.wte(inp2_ids)  # [seq_len, dim]

                input_embds.append(
                    torch.cat(
                        [
                            embds_inp1,
                            embds_imgs[batch_idx],
                            embds_inp2,
                        ],
                        dim=0,
                    )
                )
                # no need to pad all image tokens here because later we only
                # forward the embeddings
                targets_ids.append(inp1_ids.tolist() + [img_id] * 1 + inp2_ids.tolist())
            else:
                input_id = torch.tensor(input_id, dtype=torch.long, device=self.device)
                embds_inp = self.wte(input_id)
                input_embds.append(embds_inp)
                targets_ids.append(input_id)

        # stack token embds and ids
        input_embds = pad_sequence(input_embds, batch_first=True, padding_value=0.0)
        targets_ids = pad_sequence(
            [
                torch.tensor(t, dtype=torch.long, device=self.device)
                for t in targets_ids
            ],
            batch_first=True,
            padding_value=pad_id,
        )

        attention_mask = torch.ones(
            input_embds.shape[:2], dtype=torch.long, device=self.device
        )
        return input_embds, targets_ids, attention_mask

    def _post_processing_response(self, generated_ids, benchmark, print_info=False):
        # post-processing for output tokens
        out_strs = []
        thinking_traces = []

        for ids in generated_ids:

            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()

            # parsing thinking content
            try:
                # rindex finding 151668 (</think>)
                index = len(ids) - ids[::-1].index(151668)
            except ValueError:
                index = 0

            thinking_content = self._decode_text(ids[:index]).strip()
            content = self._decode_text(ids[index:]).strip()

            # in case of indexing failure
            content = content.replace("<think>", "").replace("</think>", "").strip()
            out_strs.append(content)
            thinking_traces.append(thinking_content)

        # post-processing for benchmarks
        orig_strs = out_strs

        if "pope" in benchmark:
            # https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/eval_pope.py
            if print_info:
                print(f"apply special post-processing for benchmark: {benchmark}")

            ress = []
            for s in out_strs:
                # keep the first sentence
                if s.find(".") != -1:
                    s = s.split(".")[0]
                s = s.replace(",", "")
                s = s.split(" ")
                if "no" in s or "not" in s or "No" in s:
                    res = "no"
                else:
                    res = "yes"
                ress.append(res)
            out_strs = ress

        return orig_strs, out_strs, thinking_traces
