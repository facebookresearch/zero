from typing import List, Tuple

import random
import torch
import torch.nn.functional as F
import torch.distributed as dist

from copy import deepcopy
from torch.nn.utils.rnn import pad_sequence

from methods._questions_prompt_for_caption import template_for_image_tokens
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

    img_token_str = "<|reserved_special_token_0|>"
    img_id = tokenizer.convert_tokens_to_ids(img_token_str)
    bot_id = tokenizer.convert_tokens_to_ids("<|begin_of_text|>")
    pad_id = tokenizer.convert_tokens_to_ids("<|end_of_text|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    # encode imgs
    imgs: List[torch.Tensor] = [img.to(device) for img in imgs]  # bs * [3, h, w]
    embds_imgs: List[torch.Tensor] = self._forward(imgs)  # bs * [num_tokens, dim]

    # NOTE: for debugging
    # if dist.get_rank() == 0:
    #     for i, x in enumerate(embds_imgs):
    #         print(x.shape)
    #     for i, x in enumerate(imgs):
    #         print(x.shape)
    #         save_tensor_images(x, count=i, save_folder="saved_images")

    # clean chat template
    chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|start_header_id|>' + message['role'] + '<|end_header_id|>' + '\n' + message['content'] + '<|eot_id|>'}}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n' }}{% endif %}"
    tokenizer = deepcopy(tokenizer)
    tokenizer.chat_template = chat_template

    # NOTE: for debugging
    # IGNORE_INDEX = pad_id  # !!! remember to comment this line !!!

    input_ids, targets = [], []
    for i, conv in enumerate(conversations):
        input_id, target = [], []  # per sample

        # put bot_id at the beginning
        # sine we use `apply_chat_template` to directly encode the conversation
        # it won't automatically add bot_id like directly calling `tokenizer`
        input_id += [bot_id]
        if cfg.ignore_instruction_token_ids:
            target += [IGNORE_INDEX] * len(input_id)
        else:
            target += input_id

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

            # NOTE: to apply the prompt wrapper for image tokens
            # it only works for the case where just one image is in the text
            # here we stick to the original format in the paper
            # but the prompt wrapper is NOT necessary for zero-shot grafting ability
            apply_img_prompt = False
            if has_img and cfg.enable_image_prompt:
                apply_img_prompt = True

                usr_text = usr_text.replace("<image>\n", "")
                usr_text = usr_text.replace("\n<image>", "")
                usr_text = usr_text.strip()

                if cfg.enable_random_image_prompt:
                    image_prompt = random.choice(template_for_image_tokens)
                else:
                    # first one is the default and basic prompt
                    image_prompt = template_for_image_tokens[0]

                # lowercase the beginning of the text
                usr_text = usr_text[0].lower() + usr_text[1:]

                if cfg.split_image_tokens_in_prompt_with_newline:
                    usr_text = f"{image_prompt}\n{img_token_str},\n{usr_text}"
                else:
                    usr_text = f"{image_prompt}{img_token_str}, {usr_text}"

            if has_img and not apply_img_prompt:
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
                        [encode_id[0]]  # <|start_header_id|>
                        + [IGNORE_INDEX] * 1  # user
                        + [encode_id[2]]  # <|end_header_id|>
                        + [IGNORE_INDEX] * (len(encode_id) - 4)
                        + [encode_id[-1]]  # <|eot_id|>
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
                        [encode_id[0]]  # <|start_header_id|>
                        + [IGNORE_INDEX] * 1  # assistant
                        + [encode_id[2]]  # <|end_header_id|>
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

    modality_tags = []
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
            modality_tags.append("img")
        else:
            input_id = torch.tensor(input_id, dtype=torch.long, device=device)

            input_embds.append(wte(input_id))
            targets_ids.append(target)
            modality_tags.append("txt")

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

    # calc loss weights for each token in a batch
    if cfg.enable_balance_loss_weights:

        # NOTE: dynamic weights for balancing loss
        # see Section A.7. Surrogate Training for Smaller Models
        # and Figure A.1. Dynamic Loss Weights in the paper
        w = _calc_loss_weights_with_batchmean(
            cfg,
            target,
            pad_id=pad_id,
            eot_id=eot_id,
            device=device,
            modality_tags=modality_tags,
        )

    # ----------------------------------------------------------------
    # NOTE: uncomment for checking the inputs
    # for i, (k, t, tgt) in enumerate(zip(keys, input_ids, targets_ids)):
    #     print(
    #         f"... LLAMA HF FORWARD FUNCTION:\n"
    #         f"--- rank: {torch.distributed.get_rank()}\n"
    #         f"--- batch id: {i}\n"
    #         f"--- key: {k}\n"
    #         f"--- input token len: {len(tgt)} - input str len: {len(tokenizer.decode(tgt).split())}\n"
    #         f"--- token ids - p1: {t}\n"
    #         f"--- input txt - p1: {tokenizer.decode(t)}\n"
    #         f"--- target ids: {tgt}\n"
    #         f"--- target txt: {tokenizer.decode(tgt)}\n"
    #         f"--- w: {w[i].cpu().numpy().tolist() if cfg.enable_balance_loss_weights else 'N/A'}\n"
    #     )
    # # save_tensor_images(imgs, count=i, save_folder="saved_images")
    # ----------------------------------------------------------------

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

    if cfg.enable_balance_loss_weights:
        # weights are the same for all tokens in each response
        loss = w[:, 1:].detach() * loss  # [bs, seq_len]

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

    bot_id = tokenizer.convert_tokens_to_ids("<|begin_of_text|>")
    pad_id = tokenizer.convert_tokens_to_ids("<|end_of_text|>")

    # rebatch data according to the dataset
    conversations = [d["messages"] for d in data]

    # clean chat template
    chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|start_header_id|>' + message['role'] + '<|end_header_id|>' + '\n' + message['content'] + '<|eot_id|>'}}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n' }}{% endif %}"
    tokenizer = deepcopy(tokenizer)
    tokenizer.chat_template = chat_template

    input_ids, targets_ids = [], []
    for i, conv in enumerate(conversations):
        input_id, target = [], []  # per sample

        # put bot_id at the beginning
        # sine we use `apply_chat_template` to directly encode the conversation
        # it won't automatically add bot_id like directly calling `tokenizer`
        input_id += [bot_id]
        if cfg.ignore_instruction_token_ids:
            target += [IGNORE_INDEX] * len(input_id)
        else:
            target += input_id

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


"""
functions for calculating loss balance weights
see Section A.11. in the paper
"""


@torch.no_grad()
def _calc_loss_weights_with_batchmean(
    cfg: object,
    target: torch.Tensor,
    pad_id: int,
    eot_id: int,
    device: torch.device,
    modality_tags: List[str] = None,
):
    if modality_tags is None:
        raise ValueError("modality_tags must be provided")
    else:
        assert len(modality_tags) == len(target)

    eot_idxss = []  # List[List[int]]: indices of eot_id in each sequence
    tok_cntss = []  # List[List[int]]: counts of tokens invloved in loss
    actual_tok_cntss = []  # List[List[int]]: counts of tokens in each segment

    for t, tag in zip(target, modality_tags):
        # add eot_id to the end of the sequence, which is shaved by the max_seq_len
        t[-1] = eot_id if t[-1] != pad_id else pad_id

        # count tokens per segment
        eot_mask = t.eq(eot_id).nonzero().squeeze(1) + 1
        segs = torch.tensor_split(t, eot_mask.tolist())
        actual_cnts = torch.tensor(
            [(seg != pad_id).sum() for seg in segs[:-1]],
            dtype=torch.long,
            device=device,
        )
        if tag == "txt":
            cnts = torch.ones_like(actual_cnts)
        else:
            cnts = actual_cnts

        eot_idxss.append(eot_mask.tolist())
        tok_cntss.append(cnts)
        actual_tok_cntss.append(actual_cnts)

    tok_cntss = pad_sequence(
        tok_cntss,
        batch_first=True,
        padding_value=0,
    )  # on the same device as target

    # calc weights
    ws_mask = (tok_cntss != 0).long()

    # log smooth
    ws = torch.log(tok_cntss.float() + 1)  # [bs, seq_len]

    # max norm
    ws_max = ws.max()
    if cfg.reduce_batch_loss:
        dist.all_reduce(ws_max, op=dist.ReduceOp.MAX)
    ws = ws / ws_max  # [bs, seq_len]

    # gradual inverse weighting
    ord = cfg.balance_loss_exp_ord
    ws = 1 / torch.pow(ws, ord)  # [bs, seq_len]
    ws[ws_mask == 0] = 0.0

    # avg norm
    if cfg.balance_loss_exp_avg_norm:
        ws_sum = ws.sum()
        ws_mask_sum = ws_mask.sum()
        if cfg.reduce_batch_loss:
            dist.all_reduce(ws_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(ws_mask_sum, op=dist.ReduceOp.SUM)
        ws_mean = ws_sum / ws_mask_sum
        ws = ws / ws_mean  # [bs, seq_len]

    # don't penalize long responses
    ws = torch.clamp(ws, min=1.0)  # [bs, seq_len]
    ws[ws_mask == 0] = 0.0

    # fill values
    w = torch.zeros_like(target, dtype=torch.float)
    for i, (eot_idxs, tok_cnts, val_ws, tag) in enumerate(
        zip(eot_idxss, actual_tok_cntss, ws, modality_tags)
    ):
        for j, eot_idx in enumerate(eot_idxs):
            tok_cnt = tok_cnts[j]
            if tag == "txt":
                w[i, eot_idx - tok_cnt : eot_idx] = 1.0
            else:
                w[i, eot_idx - tok_cnt : eot_idx] = val_ws[j]

    # check
    assert ((w > 0) == (target != pad_id)).all()
    return w
