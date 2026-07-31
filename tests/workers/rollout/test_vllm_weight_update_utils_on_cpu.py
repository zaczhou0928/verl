# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_weight_update_utils():
    module_path = _REPO_ROOT / "verl/workers/rollout/vllm_rollout/weight_update_utils.py"
    spec = importlib.util.spec_from_file_location("weight_update_utils", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


_weight_update_utils = _load_weight_update_utils()
apply_buffer_updates = _weight_update_utils.apply_buffer_updates
split_buffer_updates = _weight_update_utils.split_buffer_updates


def _load_vllm_rollout_utils():
    """Load vllm_rollout/utils.py with heavyweight deps stubbed.

    Injected ``sys.modules`` entries are restored afterwards so the fakes do not
    leak into other tests; the loaded module keeps working since it binds the
    names it needs at import time.
    """
    module_name = "verl.workers.rollout.vllm_rollout.utils"
    module_path = _REPO_ROOT / "verl/workers/rollout/vllm_rollout/utils.py"

    fake_outputs = types.ModuleType("vllm.outputs")

    class _FakeRequestOutput:
        pass

    fake_outputs.RequestOutput = _FakeRequestOutput
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.outputs = fake_outputs

    fake_vllm_third_party = types.ModuleType("verl.third_party.vllm")
    fake_vllm_third_party.VLLM_SLEEP_LEVEL = 1
    fake_vllm_third_party.get_version = lambda pkg: "0.8.0"

    fake_vllm_utils = types.ModuleType("verl.utils.vllm")

    class _FakeTensorLoRARequest:
        pass

    class _FakeVLLMHijack:
        @staticmethod
        def hijack():
            return None

    fake_vllm_utils.TensorLoRARequest = _FakeTensorLoRARequest
    fake_vllm_utils.VLLMHijack = _FakeVLLMHijack

    fake_vllm_patch = types.ModuleType("verl.utils.vllm.patch")
    fake_vllm_patch.patch_vllm_moe_model_weight_loader = lambda model: None

    fake_vllm_fp8 = types.ModuleType("verl.utils.vllm.vllm_fp8_utils")
    fake_vllm_fp8.apply_vllm_fp8_patches = lambda: None
    fake_vllm_fp8.is_fp8_model = lambda config: False
    fake_vllm_fp8.load_quanted_weights = lambda weights, runner, is_drafter=False: weights

    fake_vllm_online_quant = types.ModuleType("verl.utils.vllm.vllm_online_quant_utils")
    fake_vllm_online_quant.requires_native_vllm_reload = lambda config: False

    fake_platform = types.ModuleType("verl.plugin.platform")
    fake_platform.get_platform = lambda: None

    fakes = {
        "vllm": fake_vllm,
        "vllm.outputs": fake_outputs,
        "verl.third_party.vllm": fake_vllm_third_party,
        "verl.utils.vllm": fake_vllm_utils,
        "verl.utils.vllm.patch": fake_vllm_patch,
        "verl.utils.vllm.vllm_fp8_utils": fake_vllm_fp8,
        "verl.utils.vllm.vllm_online_quant_utils": fake_vllm_online_quant,
        "verl.plugin.platform": fake_platform,
        "verl.workers.rollout.vllm_rollout.weight_update_utils": _weight_update_utils,
    }

    saved = {name: sys.modules.get(name) for name in fakes}
    try:
        sys.modules.update(fakes)
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        for name, prev in saved.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev
    return module


_vllm_rollout_utils = _load_vllm_rollout_utils()
vLLMColocateWorkerExtension = _vllm_rollout_utils.vLLMColocateWorkerExtension


class _ToyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4, bias=False)
        self.register_buffer("e_score_correction_bias", torch.zeros(4, dtype=torch.float32))


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_ToyBlock()])


class _FakeVllmConfig:
    def __init__(self, speculative_config=None):
        self.speculative_config = speculative_config


class _FakeModelRunner:
    def __init__(self, inner_model, speculative_config=None):
        self.model = inner_model
        self.vllm_config = _FakeVllmConfig(speculative_config=speculative_config)


def test_split_buffer_updates_routes_registered_buffers():
    model = _ToyModel()
    weights = [
        ("model.layers.0.linear.weight", torch.ones(4, 4, dtype=torch.float32)),
        ("model.layers.0.e_score_correction_bias", torch.arange(4, dtype=torch.float32)),
    ]

    param_updates, buffer_updates, named_buffers = split_buffer_updates(model, weights)

    assert [name for name, _ in param_updates] == ["model.layers.0.linear.weight"]
    assert [name for name, _ in buffer_updates] == ["model.layers.0.e_score_correction_bias"]
    assert "model.layers.0.e_score_correction_bias" in named_buffers


def test_apply_buffer_updates_copies_buffer_values():
    model = _ToyModel()
    updates = [("model.layers.0.e_score_correction_bias", torch.arange(4, dtype=torch.float32) + 1)]

    loaded = apply_buffer_updates(model, updates)

    assert loaded == 1
    torch.testing.assert_close(
        model.model.layers[0].e_score_correction_bias, torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    )


def test_apply_buffer_updates_ignores_non_buffer_weights():
    model = _ToyModel()
    weights = [("model.layers.0.linear.weight", torch.ones(4, 4, dtype=torch.float32))]

    loaded = apply_buffer_updates(model, weights)

    assert loaded == 0
    assert torch.count_nonzero(model.model.layers[0].e_score_correction_bias) == 0


def test_vllm_update_weights_loads_params_and_buffers():
    model = _ToyModel()
    loaded_param_names = []
    apply_named_buffers = []

    def _fake_load_weights(weights):
        loaded_param_names.extend(name for name, _ in weights)

    model.load_weights = _fake_load_weights

    original_apply_buffer_updates = _vllm_rollout_utils.apply_buffer_updates

    def _spy_apply_buffer_updates(inner_model, buffer_updates, named_buffers=None):
        apply_named_buffers.append(named_buffers)
        return original_apply_buffer_updates(inner_model, buffer_updates, named_buffers=named_buffers)

    _vllm_rollout_utils.apply_buffer_updates = _spy_apply_buffer_updates

    worker = object.__new__(vLLMColocateWorkerExtension)
    worker.model_runner = _FakeModelRunner(model)

    weights = [
        ("model.layers.0.linear.weight", torch.ones(4, 4, dtype=torch.float32)),
        ("model.layers.0.e_score_correction_bias", torch.arange(4, dtype=torch.float32) + 5),
    ]

    try:
        worker._update_weights(weights, peft_config=None, base_sync_done=False)
    finally:
        _vllm_rollout_utils.apply_buffer_updates = original_apply_buffer_updates

    assert loaded_param_names == ["model.layers.0.linear.weight"]
    assert apply_named_buffers and apply_named_buffers[0] is not None
    torch.testing.assert_close(
        model.model.layers[0].e_score_correction_bias, torch.tensor([5, 6, 7, 8], dtype=torch.float32)
    )


def test_vllm_update_weights_syncs_buffers_to_mtp_drafter():
    """When an MTP drafter is synced, its registered buffers must be updated too."""
    main_model = _ToyModel()
    drafter_model = _ToyModel()
    main_model.load_weights = lambda weights: None
    drafter_model.load_weights = lambda weights: None

    class _SpecConfig:
        method = "mtp"
        draft_model_config = object()

    class _Drafter:
        def __init__(self, m):
            self.model = m

    worker = object.__new__(vLLMColocateWorkerExtension)
    worker.model_runner = _FakeModelRunner(main_model, speculative_config=_SpecConfig())
    worker.model_runner.drafter = _Drafter(drafter_model)

    weights = [
        ("model.layers.0.linear.weight", torch.ones(4, 4, dtype=torch.float32)),
        ("model.layers.0.e_score_correction_bias", torch.arange(4, dtype=torch.float32) + 5),
    ]

    worker._update_weights(weights, peft_config=None, base_sync_done=False)

    expected = torch.tensor([5, 6, 7, 8], dtype=torch.float32)
    torch.testing.assert_close(main_model.model.layers[0].e_score_correction_bias, expected)
    torch.testing.assert_close(drafter_model.model.layers[0].e_score_correction_bias, expected)


class _FakeNativeReceiver:
    instances = []
    weights = [
        ("model.layers.0.self_attn.q_proj.weight", torch.ones(2, 2)),
        ("lm_head.weight", torch.full((2, 2), 2.0)),
    ]

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.finished = False
        self.aborted_with = None
        self.iter_options = None
        self.__class__.instances.append(self)

    def iter_weights(self, **kwargs):
        self.iter_options = kwargs
        return iter(self.weights)

    def finish(self):
        self.finished = True

    def abort(self, error):
        self.aborted_with = error


def _make_native_reload_worker(reload_weights):
    worker = object.__new__(vLLMColocateWorkerExtension)
    worker.model_runner = _FakeModelRunner(_ToyModel())
    worker.device = torch.device("cpu")
    worker.local_rank = 0
    worker._native_reload_poisoned = False
    worker._get_zmq_handle = lambda: "ipc:///tmp/test-native-mxfp8.sock"
    worker.reload_weights = reload_weights
    return worker


def test_online_mxfp8_uses_one_native_reload_for_complete_stream():
    _FakeNativeReceiver.instances.clear()
    received = []
    calls = []

    def _reload_weights(*, weights_iterator, is_checkpoint_format):
        calls.append(is_checkpoint_format)
        received.extend(weights_iterator)

    worker = _make_native_reload_worker(_reload_weights)
    worker._reload_online_mxfp8_from_ipc(_FakeNativeReceiver, peft_config=None, use_shm=False)

    receiver = _FakeNativeReceiver.instances[-1]
    assert calls == [True]
    assert [name for name, _ in received] == [name for name, _ in _FakeNativeReceiver.weights]
    assert receiver.iter_options == {"clone": True, "defer_last_ack": True}
    assert receiver.finished
    assert receiver.aborted_with is None
    assert not worker._native_reload_poisoned


def test_online_mxfp8_reload_failure_poison_worker_and_rejects_next_sync():
    _FakeNativeReceiver.instances.clear()

    def _reload_weights(*, weights_iterator, is_checkpoint_format):
        next(iter(weights_iterator))
        raise ValueError("quantization failed")

    worker = _make_native_reload_worker(_reload_weights)

    with pytest.raises(ValueError, match="quantization failed"):
        worker._reload_online_mxfp8_from_ipc(_FakeNativeReceiver, peft_config=None, use_shm=False)

    receiver = _FakeNativeReceiver.instances[-1]
    assert isinstance(receiver.aborted_with, ValueError)
    assert worker._native_reload_poisoned

    with pytest.raises(RuntimeError, match="poisoned"):
        worker._reload_online_mxfp8_from_ipc(_FakeNativeReceiver, peft_config=None, use_shm=False)


def test_online_mxfp8_rejects_training_side_scale_weights():
    weights = [
        ("model.layers.0.mlp.down_proj.weight", torch.ones(2, 2)),
        ("model.layers.0.mlp.down_proj.weight_scale", torch.ones(2)),
    ]

    with pytest.raises(ValueError, match="quantization scale"):
        list(vLLMColocateWorkerExtension._reject_online_mxfp8_scale_weights(weights))


def test_online_mxfp8_does_not_apply_legacy_fp8_patch(monkeypatch):
    patch_calls = []
    monkeypatch.setenv("VERL_VLLM_FP8_QUANT_ENABLED", "1")
    monkeypatch.setattr(_vllm_rollout_utils, "requires_native_vllm_reload", lambda config: True)
    monkeypatch.setattr(_vllm_rollout_utils, "apply_vllm_fp8_patches", lambda: patch_calls.append(True))

    worker = vLLMColocateWorkerExtension.__new__(
        vLLMColocateWorkerExtension,
        vllm_config=types.SimpleNamespace(quant_config=None),
    )

    assert patch_calls == []
    assert not worker._native_reload_poisoned
