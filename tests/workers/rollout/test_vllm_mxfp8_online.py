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

from types import SimpleNamespace

from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer


def _server(quantization="mxfp8", load_format="dummy"):
    server = object.__new__(vLLMHttpServer)
    server.config = SimpleNamespace(
        quantization=quantization,
        qat=None,
        load_format=load_format,
        quantization_config_file=None,
    )
    server.model_config = SimpleNamespace(hf_config=SimpleNamespace(num_hidden_layers=2))
    return server


def test_mxfp8_selects_vllm_online_quantization():
    quantization, hf_overrides, quantization_config = _server()._apply_quantization()

    assert quantization == "mxfp8"
    # A checkpoint quantization_config in hf_overrides would make vLLM build a
    # checkpoint-format backend instead of OnlineQuantizationConfig.
    assert hf_overrides == {}
    assert quantization_config == {"ignore": [r"re:.*mlp\.gate$", r"re:.*mlp\.shared_expert_gate$"]}


def test_no_quantization_is_unaffected():
    assert _server(quantization=None)._apply_quantization() == (None, {}, None)
