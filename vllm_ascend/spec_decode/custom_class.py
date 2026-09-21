# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from inspect import getattr_static
from typing import Any, Final

import torch
from vllm.config import VllmConfig

MODEL_AWARE_CUSTOM_PROPOSER_API_VERSION: Final = 1


@dataclass(frozen=True, slots=True)
class ModelAwareCustomProposerRuntime:
    """Runtime resources for a model-backed custom proposer.

    The upstream ``custom_class`` contract remains token-only. A proposer opts
    into the Ascend model-runner lifecycle by declaring the matching API
    version and implementing ``bind_runtime``. The runner is intentionally
    explicit here: built-in Ascend model proposers receive the same object and
    plugins can reuse that implementation instead of duplicating scheduling,
    cache, and distributed logic.
    """

    api_version: int
    vllm_config: VllmConfig
    device: torch.device
    runner: Any


def is_model_aware_custom_proposer(proposer: object | None) -> bool:
    if proposer is None:
        return False
    # Read the marker without invoking a dynamic ``__getattr__``. Opting into
    # the model lifecycle must be explicit on the instance or its class.
    version = getattr_static(proposer, "npu_model_runner_api_version", None)
    if version is None:
        return False
    if version != MODEL_AWARE_CUSTOM_PROPOSER_API_VERSION:
        raise ValueError(
            "unsupported NPU model-aware custom proposer API version: "
            f"expected {MODEL_AWARE_CUSTOM_PROPOSER_API_VERSION}, got {version}"
        )
    if not callable(getattr(proposer, "bind_runtime", None)):
        raise TypeError("a model-aware custom proposer must implement bind_runtime(runtime)")
    return True
