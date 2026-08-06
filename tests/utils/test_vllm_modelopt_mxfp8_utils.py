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

import torch
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMxFp8FusedMoE,
    ModelOptMxFp8LinearMethod,
)

from verl.utils.vllm import vllm_modelopt_mxfp8_utils as mxfp8


def _build_model():
    """model.layers.0 with an MXFP8 attention linear, MXFP8 experts and a BF16 router."""

    def _module(quant_method=None):
        module = torch.nn.Module()
        if quant_method is not None:
            module.quant_method = quant_method
        return module

    # object.__new__ skips the constructors, which need a full vLLM quant config.
    attn = _module()
    attn.qkv_proj = _module(object.__new__(ModelOptMxFp8LinearMethod))

    mlp = _module()
    # w13_weight/w2_weight is how get_module_from_param_name recognises the fused
    # MoE block and stops before the per-expert name parts.
    mlp.experts = _module(object.__new__(ModelOptMxFp8FusedMoE))
    mlp.experts.w13_weight = torch.empty(0)
    mlp.experts.w2_weight = torch.empty(0)
    mlp.gate = _module(object.__new__(UnquantizedLinearMethod))

    layer0 = _module()
    layer0.self_attn = attn
    layer0.mlp = mlp
    layer0.input_layernorm = _module()

    inner = _module()
    inner.layers = torch.nn.ModuleList([layer0])

    model = _module()
    model.model = inner
    model.packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    return model


def test_get_modelopt_mxfp8_quant_config_returns_expected_fields():
    config = mxfp8.get_modelopt_mxfp8_quant_config()

    assert config["quant_method"] == "modelopt"
    assert config["quant_algo"] == "MXFP8"
    assert "lm_head" in config["ignore"]
    assert set(config) == {"quant_method", "quant_algo", "ignore"}


def test_should_quantize_param_follows_target_module():
    model = _build_model()

    assert mxfp8.should_quantize_param("model.layers.0.self_attn.q_proj.weight", model)
    assert mxfp8.should_quantize_param("model.layers.0.mlp.experts.0.gate_proj.weight", model)
    # Router is excluded in the quant config, so vLLM gave it UnquantizedLinearMethod.
    assert not mxfp8.should_quantize_param("model.layers.0.mlp.gate.weight", model)
    assert not mxfp8.should_quantize_param("model.layers.0.input_layernorm.weight", model)
    assert not mxfp8.should_quantize_param("model.layers.0.self_attn.q_proj.bias", model)


def test_quant_weights_by_name_emits_weight_scale(monkeypatch):
    monkeypatch.setattr(mxfp8, "should_quantize_param", lambda name, model: "q_proj" in name)

    passthrough = torch.randn(64, dtype=torch.bfloat16)
    weights = [
        ("model.layers.0.self_attn.q_proj.weight", torch.randn(4, 64, dtype=torch.bfloat16)),
        ("model.layers.0.input_layernorm.weight", passthrough),
    ]

    result = list(mxfp8.quant_weights_by_name(weights, model=None))

    assert [name for name, _ in result] == [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_proj.weight_scale",
        "model.layers.0.input_layernorm.weight",
    ]
    assert result[0][1].dtype == torch.float8_e4m3fn
    assert result[1][1].dtype == torch.uint8
    assert result[1][1].shape == (4, 2)  # one E8M0 scale per 32 elements along K
    # Passthrough tensors are cloned out of the reused IPC bucket buffer.
    assert result[2][1] is not passthrough
