"""
functions and classes for reducing/gathering tensors across all ranks
with the registered autograd function

source: https://github.com/kaiyuyue/torchshard/blob/main/torchshard/distributed/comm.py
"""

import torch
import torch.distributed as dist


class Reduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx: "Reduce", input: torch.Tensor) -> torch.Tensor:
        return _reduce(input)

    @staticmethod
    def backward(ctx: "Reduce", grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


def _reduce(tensor: torch.Tensor) -> torch.Tensor:
    if dist.get_world_size() == 1:
        return tensor

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def reduce_sum(input: torch.Tensor) -> torch.Tensor:
    if torch.is_grad_enabled() and input.requires_grad:
        input = Reduce.apply(input)
    else:
        input = _reduce(input)
    return input


def _gather(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if dist.get_world_size() == 1:
        return tensor
    tensor_list = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list, tensor, async_op=False)

    output = torch.cat(tensor_list, dim=dim)
    return output


def check_divisibility(numerator, denominator):
    assert numerator % denominator == 0, "{} is not divisible by {}".format(
        numerator, denominator
    )


def divide(numerator, denominator):
    check_divisibility(numerator, denominator)
    return numerator // denominator


def _split(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if dist.get_world_size() == 1:
        return tensor

    split_size = divide(tensor.shape[dim], dist.get_world_size())
    tensor_list = torch.split(tensor, split_size, dim=dim)

    output = tensor_list[dist.get_rank()].contiguous()

    return output
