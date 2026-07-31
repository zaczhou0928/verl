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

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace


def _load_online_quant_utils():
    module_path = Path(__file__).resolve().parents[2] / "verl/utils/vllm/vllm_online_quant_utils.py"
    spec = importlib.util.spec_from_file_location("vllm_online_quant_utils", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


_online_quant_utils = _load_online_quant_utils()
is_online_mxfp8_model = _online_quant_utils.is_online_mxfp8_model
requires_native_vllm_reload = _online_quant_utils.requires_native_vllm_reload


class _FakeOnlineQuantizationConfig:
    def __init__(self, *, linear=None, moe=None):
        self.args = SimpleNamespace(linear=linear, moe=moe)


def _install_fake_online_quant_modules(monkeypatch):
    mxfp8_key = object()
    fake_online_base = types.ModuleType("vllm.model_executor.layers.quantization.online.base")
    fake_online_base.OnlineQuantizationConfig = _FakeOnlineQuantizationConfig
    fake_quant_utils = types.ModuleType("vllm.model_executor.layers.quantization.utils.quant_utils")
    fake_quant_utils.kMxfp8Dynamic = mxfp8_key
    monkeypatch.setitem(sys.modules, fake_online_base.__name__, fake_online_base)
    monkeypatch.setitem(sys.modules, fake_quant_utils.__name__, fake_quant_utils)
    return mxfp8_key


def test_detects_online_mxfp8_by_config_class_and_exact_scheme(monkeypatch):
    mxfp8_key = _install_fake_online_quant_modules(monkeypatch)
    spec = SimpleNamespace(weight=mxfp8_key)
    config = SimpleNamespace(quant_config=_FakeOnlineQuantizationConfig(linear=spec, moe=spec))

    assert is_online_mxfp8_model(config)
    assert requires_native_vllm_reload(config)


def test_does_not_route_another_online_scheme_to_mxfp8_reload(monkeypatch):
    _install_fake_online_quant_modules(monkeypatch)
    other_spec = SimpleNamespace(weight=object())
    config = SimpleNamespace(quant_config=_FakeOnlineQuantizationConfig(linear=other_spec))

    assert not is_online_mxfp8_model(config)


def test_does_not_treat_legacy_quant_config_as_online_mxfp8(monkeypatch):
    mxfp8_key = _install_fake_online_quant_modules(monkeypatch)
    config = SimpleNamespace(
        quant_config=SimpleNamespace(args=SimpleNamespace(linear=SimpleNamespace(weight=mxfp8_key), moe=None))
    )

    assert not is_online_mxfp8_model(config)
