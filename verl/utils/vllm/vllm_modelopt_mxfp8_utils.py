# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Quantize rollout weights to the ModelOpt MXFP8 checkpoint format.

The actor stays BF16; each weight becomes an FP8-E4M3 tensor plus a uint8 E8M0
scale shared by 32 elements along the input dim. vLLM's ``modelopt_mxfp8``
backend owns everything after that.
"""

import copy
from collections.abc import Iterable, Iterator
from typing import Any

import torch

# Imported at module scope on purpose: the rollout server imports this module while
# building the engine, so a missing or too-old nvidia-modelopt fails there instead
# of mid-transfer, where the receiver would die without ACKing and hang the sender.
from modelopt.torch.quantization.qtensor import MXFP8QTensor

from verl.utils.vllm.vllm_quant_utils import get_module_from_param_name

MODELOPT_MXFP8_QUANT_KWARGS: dict[str, Any] = {
    "quant_method": "modelopt",
    "quant_algo": "MXFP8",
    "ignore": ["lm_head", "*mlp.gate", "*mlp.shared_expert_gate"],
}


def get_modelopt_mxfp8_quant_config() -> dict[str, Any]:
    return copy.deepcopy(MODELOPT_MXFP8_QUANT_KWARGS)


def should_quantize_param(param_name: str, model: torch.nn.Module) -> bool:
    """Whether vLLM allocated MXFP8 weights for this parameter's destination.

    Read off the target module rather than a name list, like
    ``vllm_quant_utils.is_fp8_weight`` reads the target dtype, so excluded
    layers (router, ``lm_head``), norms and embeddings pass through in BF16.
    """
    if not param_name.endswith(".weight"):
        return False

    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptMxFp8FusedMoE,
        ModelOptMxFp8LinearMethod,
    )

    module = get_module_from_param_name(model, param_name)
    return isinstance(
        getattr(module, "quant_method", None),
        (ModelOptMxFp8LinearMethod, ModelOptMxFp8FusedMoE),
    )


def quant_weights_by_name(
    weights: Iterable[tuple[str, torch.Tensor]],
    model: torch.nn.Module,
    dtype: torch.dtype = torch.bfloat16,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield MXFP8 weight/scale pairs, passing everything else through."""
    for name, tensor in weights:
        if not should_quantize_param(name, model):
            # vLLM's layerwise reload can hold this past the callback, and the
            # receiver reuses one IPC bucket buffer across buckets.
            yield (name, tensor.clone())
            continue

        weight = tensor.to(dtype).contiguous()
        scale = MXFP8QTensor.get_weights_scaling_factor(weight)
        yield (name, MXFP8QTensor.quantize_with_scale(weight, scale))
        yield (name + "_scale", scale)


def load_modelopt_mxfp8_weights(weights, model_runner):
    """Quantize a weight bucket and load it into the MXFP8 vLLM model."""
    model = model_runner.model
    dtype = model_runner.vllm_config.model_config.dtype
    return model.load_weights(quant_weights_by_name(weights, model, dtype=dtype))
