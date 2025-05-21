# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple

import torch
import torch.nn.functional as F
import torch.distributed as dist

from copy import deepcopy
from torch.nn.utils.rnn import pad_sequence
from utils import save_tensor_images
from distributed.torchshard import reduce_sum

IGNORE_INDEX = -100


"""
functions for forwarding vision language models
"""


def forward_hf(
    self,
    cfg: object,
    data: Tuple[torch.Tensor, List[str], List[str]],
    tokenizer: object,
) -> dict:
    imgs, conversations, keys = data

    max_seq_len = cfg.max_seq_len
    device = cfg.device

    img_token_str = "<|image_pad|>"
    img_id = tokenizer.convert_tokens_to_ids(img_token_str)
    pad_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")

    # encode imgs
    imgs: List[torch.Tensor] = [img.to(device) for img in imgs]  # bs * [3, h, w]
    embds_imgs: List[torch.Tensor] = self._forward(imgs)  # bs * [num_tokens, dim]

    # # NOTE: for debugging
    # if dist.get_rank() == 0:
    #     for i, x in enumerate(embds_imgs):
    #         print(x.shape)
    #     for i, x in enumerate(imgs):
    #         print(x.shape)
    #         save_tensor_images(x, count=i, save_folder="saved_images")

    # clean chat template
    chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer = deepcopy(tokenizer)
    tokenizer.chat_template = chat_template

    # NOTE: for debugging
    # IGNORE_INDEX = pad_id  # !!! remember to comment this line !!!

    input_ids, targets = [], []
    for i, conv in enumerate(conversations):
        input_id, target = [], []  # per sample

        # input_id += tokenizer.apply_chat_template(
        #     [{"role": "system", "content": "You are a helpful assistant."}]
        # )
        # if cfg.ignore_instruction_token_ids:
        #     target += [IGNORE_INDEX] * len(input_id)
        # else:
        #     target += input_id

        for i, turn in enumerate(zip(conv[::2], conv[1::2])):
            usr_role = turn[0]["role"]
            ast_role = turn[1]["role"]

            assert usr_role in ["human", "user"]
            assert ast_role in ["model", "assistant"]

            # encode text from user turn
            usr_text = turn[0]["content"]
            usr_role = "user"
            has_img = "<image>" in usr_text

            if has_img:
                spans = usr_text.split("<image>")
                new_spans = []
                for i in range(len(spans) - 1):
                    new_spans.append(spans[i])
                    new_spans.append(img_token_str)
                new_spans.append(spans[-1])
                usr_text = "".join(new_spans)
            usr_turn = [{"role": usr_role, "content": usr_text}]
            encode_id = tokenizer.apply_chat_template(usr_turn)
            input_id += encode_id

            if cfg.ignore_instruction_token_ids:
                if cfg.include_special_tokens_in_masking_instruction_tokens:
                    # NOTE: for training encoder with surrogate models,
                    # we leave the special tokens unmasked,
                    # see Section A.5. Training Recipes in the paper
                    target += (
                        [encode_id[0]]  # <|im_start|>
                        + [IGNORE_INDEX] * (len(encode_id) - 3)
                        + [encode_id[-2]]  # <|im_end|>
                        + [IGNORE_INDEX]  # \n
                    )
                else:
                    target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id

            # encode text from assistant or model turn
            ast_text = turn[1]["content"]
            ast_role = "assistant"
            ast_turn = [{"role": ast_role, "content": ast_text}]
            encode_id = tokenizer.apply_chat_template(ast_turn)
            input_id += encode_id

            if cfg.ignore_instruction_token_ids:
                if cfg.include_special_tokens_in_masking_instruction_tokens:
                    # NOTE: for training encoder with surrogate models,
                    # we leave the special tokens unmasked,
                    # see Section A.5. Training Recipes in the paper
                    target += (
                        [encode_id[0]]  # <|im_start|>
                        + [IGNORE_INDEX] * 2  # assistant + \n
                        + encode_id[3:]  # all other tokens
                    )
                else:
                    # ignore the first 3 tokens: <|im_start|> + assistant + \n
                    target += [IGNORE_INDEX] * 3 + encode_id[3:]
            else:
                target += encode_id

        # batch token ids
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        input_ids.append(input_id)
        targets.append(target)

    # construct input embeddings and target ids with image tokens
    wte = self.decoder.model.get_input_embeddings()

    input_embds = []
    targets_ids = []
    for batch_idx, (input_id, target) in enumerate(zip(input_ids, targets)):
        if img_id in input_id:
            img_pad_idx = input_id.index(img_id)
            n_img_tokens = embds_imgs[batch_idx].shape[0]

            inp1_ids = input_id[:img_pad_idx]
            inp2_ids = input_id[img_pad_idx + 1 :]
            inp2_ids = inp2_ids[: max_seq_len - len(inp1_ids) - n_img_tokens]

            inp1_ids = torch.tensor(inp1_ids, dtype=torch.long, device=device)
            inp2_ids = torch.tensor(inp2_ids, dtype=torch.long, device=device)

            embds_inp1 = wte(inp1_ids)  # [seq_len, dim]
            embds_inp2 = wte(inp2_ids)  # [seq_len, dim]

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

            targets_ids.append(
                target[:img_pad_idx]
                + [IGNORE_INDEX] * n_img_tokens
                + target[img_pad_idx + 1 :]
            )
        else:
            input_id = torch.tensor(input_id, dtype=torch.long, device=device)

            input_embds.append(wte(input_id))
            targets_ids.append(target)

    # ----------------------------------------------------------------
    # NOTE: uncomment for checking the inputs
    # for i, (k, t, tgt) in enumerate(zip(keys, input_ids, targets_ids)):
    #     print(
    #         f"... QWEN HF FORWARD FUNCTION:\n"
    #         f"--- rank: {torch.distributed.get_rank()}\n"
    #         f"--- batch id: {i}\n"
    #         f"--- key: {k}\n"
    #         f"--- input token len: {len(tgt)} - input str len: {len(tokenizer.decode(tgt).split())}\n"
    #         f"--- token ids - p1: {t}\n"
    #         f"--- input txt - p1: {tokenizer.decode(t)}\n"
    #         f"--- target ids: {tgt}\n"
    #         f"--- target txt: {tokenizer.decode(tgt)}\n"
    #     )
    # # save_tensor_images(imgs, count=i, save_folder="saved_images")
    # ----------------------------------------------------------------

    # stack and pad token embds and ids
    input_embds = pad_sequence(input_embds, batch_first=True, padding_value=0.0)
    targets_ids = pad_sequence(
        [torch.tensor(t, dtype=torch.long, device=device) for t in targets_ids],
        batch_first=True,
        padding_value=IGNORE_INDEX,
    )

    # truncate to max_seq_len
    inputs_embeds = input_embds[:, :max_seq_len, :]
    target = targets_ids[:, :max_seq_len]

    # decode
    output = self.decoder(
        inputs_embeds=inputs_embeds,
        use_cache=False,
    )
    logits = output["logits"].float()

    # compute loss
    logits = logits[:, :-1, :].contiguous()
    target = target[:, 1:].contiguous()
    target[target == pad_id] = IGNORE_INDEX

    # same as reduction="mean" in F.cross_entropy over N samples in a mini-batch
    # batch mean = (l_1 + l_2 + ... + l_N) / (t_1 + t_2 + ... + t_N),
    # where l = loss, t = number of tokens

    loss = F.cross_entropy(
        logits.view(-1, logits.shape[-1]),
        target.view(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    )  # [bs * seq_len]
    loss = loss.view(logits.shape[0], -1)  # [bs, seq_len]

    # sum over seq_len per sample
    loss = loss.sum()  # [1]
    num_tokens = target.ne(IGNORE_INDEX).sum()  # [1]

    # reduce loss and tokens across all ranks
    if cfg.reduce_batch_loss:
        loss = reduce_sum(loss)  # [1]
        num_tokens = reduce_sum(num_tokens)  # [1]

    # compute loss in the global batch
    loss = loss / num_tokens  # [1]

    if loss.sum().isnan():
        raise ValueError(f"loss is NaN on rank {dist.get_rank()}")

    return loss


"""
function for forwarding language models only
"""


def forward_hf_lang(
    self,
    cfg: object,
    data: List[str],
    tokenizer: object,
) -> dict:
    max_seq_len = cfg.max_seq_len
    device = cfg.device

    pad_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")

    # rebatch data according to the dataset
    conversations = [d["messages"] for d in data]

    # clean chat template
    chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|start_header_id|>' + message['role'] + '<|end_header_id|>' + '\n' + message['content'] + '<|eot_id|>'}}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n' }}{% endif %}"
    tokenizer = deepcopy(tokenizer)
    tokenizer.chat_template = chat_template

    input_ids, targets_ids = [], []
    for i, conv in enumerate(conversations):
        input_id, target = [], []  # per sample

        # input_id += tokenizer.apply_chat_template(
        #     [{"role": "system", "content": "You are a helpful assistant."}]
        # )
        # if cfg.ignore_instruction_token_ids:
        #     target += [IGNORE_INDEX] * len(input_id)
        # else:
        #     target += input_id

        for i, turn in enumerate(zip(conv[::2], conv[1::2])):
            usr_role = turn[0]["role"]
            ast_role = turn[1]["role"]

            assert usr_role in ["human", "user"]
            assert ast_role in ["model", "assistant"]

            # encode text from user turn
            usr_text = turn[0]["content"]
            usr_role = "user"
            usr_turn = [{"role": usr_role, "content": usr_text}]
            encode_id = tokenizer.apply_chat_template(usr_turn)
            input_id += encode_id

            if cfg.ignore_instruction_token_ids:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id

            # encode text from assistant or model turn
            ast_text = turn[1]["content"]
            ast_role = "assistant"
            ast_turn = [{"role": ast_role, "content": ast_text}]
            encode_id = tokenizer.apply_chat_template(ast_turn)
            input_id += encode_id

            if cfg.ignore_instruction_token_ids:
                # ignore the first 3 tokens: <|im_start|> + assistant + \n
                target += [IGNORE_INDEX] * 3 + encode_id[3:]
            else:
                target += encode_id

        # batch token ids
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        input_ids.append(input_id)
        targets_ids.append(target)

    # stack and pad token ids
    input_ids = pad_sequence(
        [torch.tensor(t, dtype=torch.long, device=device) for t in input_ids],
        batch_first=True,
        padding_value=pad_id,
    )
    targets_ids = pad_sequence(
        [torch.tensor(t, dtype=torch.long, device=device) for t in targets_ids],
        batch_first=True,
        padding_value=IGNORE_INDEX,
    )

    # truncate to max_seq_len
    input_ids = input_ids[:, :max_seq_len]
    target = targets_ids[:, :max_seq_len]

    # decoder from hf
    output = self.decoder(
        input_ids=input_ids,
        use_cache=False,
    )
    logits = output["logits"].float()

    # compute loss
    logits = logits[:, :-1, :].contiguous()
    target = target[:, 1:].contiguous()
    target[target == pad_id] = IGNORE_INDEX

    loss = F.cross_entropy(
        logits.view(-1, logits.shape[-1]),
        target.view(-1),
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )
    num_tokens = target.ne(IGNORE_INDEX).sum()  # [1]

    # reduce loss and tokens across all ranks
    if cfg.reduce_batch_loss:
        loss = reduce_sum(loss)  # [1]
        num_tokens = reduce_sum(num_tokens)  # [1]

    # compute loss in the global batch
    loss = loss / num_tokens  # [1]

    if loss.sum().isnan():
        raise ValueError(f"loss is NaN on rank {dist.get_rank()}")

    return loss
