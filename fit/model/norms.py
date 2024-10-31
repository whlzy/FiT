# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

from functools import partial

import torch
import torch.nn as nn

import triton
import triton.language as tl



def create_norm(norm_type: str, dim: int, eps: float = 1e-6):
    """
    Creates the specified normalization layer based on the norm_type.

    Args:
        norm_type (str): The type of normalization layer to create.
            Supported types: 1. rmsnorm 2. fused_rmsnorm 3. layernorm 4. np_layernorm
        dim (int): The dimension of the normalization layer.
        eps (float, optional): The epsilon value for numerical stability. Defaults to 1e-6.

    Returns:
        The created normalization layer.

    Raises:
        NotImplementedError: If an unknown norm_type is provided.
    """
    if norm_type == None or norm_type == "":
        return nn.Identity()
    norm_type = norm_type.lower()  # Normalize to lowercase
    
    if norm_type == "w_layernorm":
        return nn.LayerNorm(dim, eps=eps, bias=False)
    elif norm_type == "layernorm": 
        return nn.LayerNorm(dim, eps=eps, elementwise_affine=False, bias=False)
    elif norm_type == "w_rmsnorm":
        return RMSNorm(dim, eps=eps)
    elif norm_type == "rmsnorm":
        return RMSNorm(dim, include_weight=False, eps=eps)
    elif norm_type == 'none':
        return nn.Identity()
    else:
        raise NotImplementedError(f"Unknown norm_type: '{norm_type}'")


class RMSNorm(nn.Module):
    """
    Initialize the RMSNorm normalization layer.

    Args:
        dim (int): The dimension of the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        weight (nn.Parameter): Learnable scaling parameter.

    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)  # type: ignore

