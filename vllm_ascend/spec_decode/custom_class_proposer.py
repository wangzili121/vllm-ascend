#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
# This file is a part of the vllm-ascend project.
#

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

CUSTOM_CLASS_METHOD = "custom_class"


def is_custom_class_method(speculative_config: Any) -> bool:
    return getattr(speculative_config, "method", None) == CUSTOM_CLASS_METHOD


def create_custom_class_proposer(vllm_config: Any) -> Any:
    speculative_config = getattr(vllm_config, "speculative_config", None)
    class_path = getattr(speculative_config, "model", None)
    if not class_path:
        raise ValueError("custom_class speculative config requires model class path")

    module_name, _, class_name = str(class_path).rpartition(".")
    if not module_name or not class_name:
        raise ValueError(f"invalid custom proposer class path: {class_path!r}")

    module = importlib.import_module(module_name)
    proposer_class = getattr(module, class_name)
    proposer = proposer_class(vllm_config)
    ensure_custom_class_lifecycle(proposer)
    return proposer


def ensure_custom_class_lifecycle(proposer: Any) -> None:
    if not hasattr(proposer, "load_model"):
        setattr(proposer, "load_model", lambda *args, **kwargs: None)
    if not hasattr(proposer, "dummy_run"):
        setattr(proposer, "dummy_run", lambda *args, **kwargs: None)


def set_custom_class_request_ids(
    proposer: Any,
    request_ids: Any,
    sampled_token_ids: Any,
) -> None:
    setter: Callable[[tuple[str, ...]], Any] | None = getattr(
        proposer, "set_request_ids", None
    )
    if setter is None:
        return

    ids = tuple(str(request_id) for request_id in request_ids)
    batch_size = _batch_size(sampled_token_ids)
    if batch_size is not None:
        ids = ids[:batch_size]
    setter(ids)


def _batch_size(sampled_token_ids: Any) -> int | None:
    if isinstance(sampled_token_ids, list):
        return len(sampled_token_ids)
    shape = getattr(sampled_token_ids, "shape", None)
    if shape:
        return int(shape[0])
    try:
        return len(sampled_token_ids)
    except TypeError:
        return None
