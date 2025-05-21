# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, List

import torch
import torch.nn as nn


class Zero(nn.Module):

    def __init__(
        self,
        cfg: object,
        encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
    ):
        super(Zero, self).__init__()
        mlp_depth = cfg.adapter_mlp_depth  # multiple layer perceptron depth
        num_embd = cfg.num_embd

        # ease saving/loading models in one place
        self.encoder = encoder

        self.encoder_shave_last_n_layers = cfg.encoder_shave_last_n_layers
        if self.encoder_shave_last_n_layers > 0:
            print(f"shave last {self.encoder_shave_last_n_layers} layers of encoder")
        self.ignore_encoder_cls_token = cfg.ignore_encoder_cls_token

        # input dim is the output dim of the encoder, e.g., clip
        modules = [nn.Linear(1024, num_embd)]  # TODO: rm hard-coded input dim
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(num_embd, num_embd))
        self.adapter = nn.Sequential(*modules)

        if self.encoder is not None:
            # <|start_of_image|> and <|end_of_image|>
            self.embd_soi = nn.Parameter(torch.randn(1, num_embd, device=cfg.device))
            self.embd_eoi = nn.Parameter(torch.randn(1, num_embd, device=cfg.device))
            self._init_soi_eoi(device=cfg.device)

        self.decoder = decoder

        # for language-only training
        if self.encoder is None:
            self.adapter = None  # :)

        # move to device
        if self.adapter is not None:
            self.adapter = self.adapter.to(cfg.device)

    def _init_soi_eoi(self, device=None):
        d = self.embd_soi.shape[1]  # [n, dim]
        std = 1 / d**0.5
        self.embd_soi = torch.nn.Parameter(torch.randn(1, d, device=device) * std)
        self.embd_eoi = torch.nn.Parameter(torch.randn(1, d, device=device) * std)

    # real forward function is registered from methods
    def forward(self):
        raise NotImplementedError

    def _forward(self, xs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            xs (List[torch.Tensor]): list of tensors, each tensor is [patches, c, h, w]
                e.g., List[torch.Size([5, 3, 336, 336]), torch.Size([7, 3, 336, 336])]
                batch size is len(xs)
        """
        # get number of patches
        split_sizes = [x.shape[0] for x in xs]

        # get shapes
        c, h, w = xs[0].shape[1:]  # [c, h, w]
        bs = len(xs)  # batch size

        # cat images
        x = torch.cat(xs, dim=0)  # [bs * [num_patches], c, h, w]

        # get image tokens
        x = self.encoder(x, output_hidden_states=True)
        if self.encoder_shave_last_n_layers > 0:
            x = x.hidden_states[-self.encoder_shave_last_n_layers]
        else:
            x = x.last_hidden_state  # [bs * [num_patches], num_tokens, dim]

        # ignore cls token
        if self.ignore_encoder_cls_token:
            x = x[:, 1:, :]

        x = self.encoder.vision_model.post_layernorm(x)
        x = self.adapter(x)

        # NOTE: to organize the image tokens of multiple image patches
        # we simply flatten and concat those patch tokens
        # for more comprehensive ways for any-resolution (anyres) image input
        # see https://github.com/LLaVA-VL/LLaVA-NeXT/blob/main/llava/model/llava_arch.py#L350
        # and https://arxiv.org/abs/2408.03326

        # or like Gemma-3 to perform pooling operation
        # see https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/modeling_gemma3.py

        # reshape back
        xs = torch.split(x, split_sizes, dim=0)  # bs * [num_patches, num_tokens, dim]

        # flatten to [num_patches * num_tokens, dim]
        xs = [x.reshape(-1, x.shape[-1]) for x in xs]

        # add soi and eoi embeddings
        xs = [torch.cat([self.embd_soi, x, self.embd_eoi], dim=0) for x in xs]

        # bs * [1 + num_patches * num_tokens + 1, dim]
        return xs
