# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
from vllm.config.speculative import SpeculativeConfig

import vllm_ascend.patch.platform.patch_speculative_config as speculative_patch


class _FakeModelConfig:
    def __init__(self, **kwargs):
        self.model = kwargs["model"]
        self.max_model_len = kwargs["spec_target_max_model_len"]
        self.architectures = kwargs["hf_overrides"]["architectures"]
        self.hf_config = SimpleNamespace(model_type="deepseek_v4")
        self._verified_parallel_config = None

    def get_vocab_size(self):
        return 129280

    def verify_with_parallel_config(self, parallel_config):
        self._verified_parallel_config = parallel_config


class _FakeTargetConfig:
    tokenizer = "/models/dsv4"
    tokenizer_mode = "deepseek_v4"
    trust_remote_code = True
    allowed_local_media_path = None
    allowed_media_domains = None
    dtype = "bfloat16"
    seed = 0
    tokenizer_revision = None
    max_model_len = 1048576
    max_logprobs = 20
    config_format = "auto"

    def get_vocab_size(self):
        return 129280


def _make_config(monkeypatch, **overrides):
    monkeypatch.setattr(speculative_patch, "ModelConfig", _FakeModelConfig)
    monkeypatch.setattr(
        SpeculativeConfig,
        "create_draft_parallel_config",
        staticmethod(lambda target, size: (target, size)),
    )
    values = {
        "method": "orthrus",
        "model": "/models/orthrus-head",
        "num_speculative_tokens": 31,
        "target_model_config": _FakeTargetConfig(),
        "target_parallel_config": SimpleNamespace(tensor_parallel_size=8),
    }
    values.update(overrides)
    return SpeculativeConfig(**values)


def test_orthrus_config_uses_separate_head_and_target_tp(monkeypatch):
    config = _make_config(monkeypatch)

    assert config.method == "orthrus"
    assert config.parallel_drafting
    assert config.enforce_eager
    assert config.uses_draft_model()
    assert config.draft_tensor_parallel_size == 8
    assert config.draft_model_config.architectures == ["OrthrusDSV4DraftModel"]
    assert config.draft_model_config._verified_parallel_config == (
        config.target_parallel_config,
        8,
    )


def test_orthrus_config_rejects_non_k32_head(monkeypatch):
    with pytest.raises(ValueError, match="requires 31 speculative tokens"):
        _make_config(monkeypatch, num_speculative_tokens=7)


def test_orthrus_config_rejects_independent_draft_tp(monkeypatch):
    with pytest.raises(ValueError, match="draft TP must equal target model TP"):
        _make_config(monkeypatch, draft_tensor_parallel_size=1)
