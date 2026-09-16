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
from collections.abc import Callable, Mapping
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


def register_custom_class_draft_bundle(
    worker: Any,
    request_id: str,
    draft_token_ids: list[list[int]],
    boundary_token_ids: list[list[int]],
    prompt_token_count: int | None = None,
    candidate_metadata: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    proposer = _worker_custom_proposer(worker)
    register = getattr(proposer, "register_draft_bundle", None)
    if proposer is None or register is None:
        return {"status": "skipped", "reason": "no_custom_proposer"}

    try:
        internal_request_id = _worker_request_id(worker, request_id)
        if prompt_token_count is None:
            prompt_token_count = _worker_prompt_token_count(
                worker, internal_request_id
            )
        result = register(
            internal_request_id,
            draft_token_ids,
            boundary_token_ids,
            prompt_token_count,
            candidate_metadata=candidate_metadata,
            external_request_id=request_id,
        )
        if isinstance(result, Mapping):
            return {
                "request_id": request_id,
                "internal_request_id": internal_request_id,
                **dict(result),
            }
        return {
            "status": "ok",
            "request_id": request_id,
            "internal_request_id": internal_request_id,
            "result": result,
        }
    except KeyError:
        return {"status": "pending", "reason": "request_not_active"}
    except (TypeError, ValueError, RuntimeError) as error:
        return {"status": "error", "error": str(error)}


def clear_custom_class_draft(worker: Any, request_id: str) -> dict[str, Any]:
    proposer = _worker_custom_proposer(worker)
    clear = getattr(proposer, "clear_request", None)
    if proposer is None or clear is None:
        return {"status": "skipped", "reason": "no_custom_proposer"}
    result = clear(request_id)
    return dict(result) if isinstance(result, Mapping) else {"status": "ok"}


def custom_class_draft_status(worker: Any) -> dict[str, Any]:
    proposer = _worker_custom_proposer(worker)
    status = getattr(proposer, "status", None)
    if proposer is None or status is None:
        return {"status": "skipped", "reason": "no_custom_proposer"}
    result = status()
    return {"status": "ok", **(dict(result) if isinstance(result, Mapping) else {})}


def _worker_custom_proposer(worker: Any) -> Any | None:
    model_runner = getattr(worker, "model_runner", None)
    return getattr(model_runner, "drafter", None)


def _worker_request_id(worker: Any, external_request_id: str) -> str:
    model_runner = getattr(worker, "model_runner", None)
    requests = getattr(model_runner, "requests", None)
    if not isinstance(requests, Mapping):
        raise RuntimeError("active vLLM request mapping is unavailable")
    if external_request_id in requests:
        return external_request_id

    bases = (
        external_request_id,
        f"cmpl-{external_request_id}-0",
        f"chatcmpl-{external_request_id}",
    )
    matches = {
        str(internal_request_id)
        for internal_request_id in requests
        if any(
            str(internal_request_id) == base
            or str(internal_request_id).startswith(base + "-")
            for base in bases
        )
    }
    if len(matches) == 1:
        return matches.pop()
    if not matches:
        raise KeyError(external_request_id)
    raise RuntimeError(f"external request id {external_request_id!r} is ambiguous")


def _worker_prompt_token_count(worker: Any, request_id: str) -> int:
    model_runner = getattr(worker, "model_runner", None)
    requests = getattr(model_runner, "requests", None)
    request_state = requests.get(request_id) if isinstance(requests, Mapping) else None
    if request_state is None:
        raise RuntimeError(f"active request state not found for {request_id!r}")

    count = getattr(request_state, "num_prompt_tokens", None)
    if count is not None:
        return int(count)
    prompt_token_ids = getattr(request_state, "prompt_token_ids", None)
    if prompt_token_ids is not None:
        return len(prompt_token_ids)
    prompt_embeds = getattr(request_state, "prompt_embeds", None)
    shape = getattr(prompt_embeds, "shape", None)
    if shape:
        return int(shape[0])
    raise RuntimeError(f"prompt length is unavailable for {request_id!r}")
