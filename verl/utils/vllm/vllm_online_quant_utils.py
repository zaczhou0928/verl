# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Helpers for vLLM's native online-quantization reload path."""

from typing import Any


def is_online_mxfp8_model(vllm_config: Any) -> bool:
    """Return whether ``vllm_config`` describes vLLM online MXFP8.

    The exact quantization key is checked in addition to the config class so
    future online schemes are not silently routed through the MXFP8 path.
    Imports are intentionally lazy: verl's CPU-only tests and non-vLLM
    backends must remain importable without the vLLM online modules present.
    """
    quant_config = getattr(vllm_config, "quant_config", None)
    if quant_config is None:
        return False

    try:
        from vllm.model_executor.layers.quantization.online.base import OnlineQuantizationConfig
        from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
    except ImportError:
        return False

    if not isinstance(quant_config, OnlineQuantizationConfig):
        return False

    args = getattr(quant_config, "args", None)
    specs = (getattr(args, "linear", None), getattr(args, "moe", None))
    configured_weights = [getattr(spec, "weight", None) for spec in specs if spec is not None]
    return bool(configured_weights) and all(weight == kMxfp8Dynamic for weight in configured_weights)


def requires_native_vllm_reload(vllm_config: Any) -> bool:
    """Return whether weight sync must use vLLM's native reload entry point."""
    return is_online_mxfp8_model(vllm_config)
