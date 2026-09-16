import sys
from types import ModuleType, SimpleNamespace

from vllm_ascend.spec_decode.custom_class_proposer import (
    create_custom_class_proposer,
    set_custom_class_request_ids,
)


class _CustomProposer:
    def __init__(self, vllm_config):
        self.vllm_config = vllm_config
        self.request_ids = None

    def propose(self, *args, **kwargs):
        return [[1, 2]]

    def set_request_ids(self, request_ids):
        self.request_ids = request_ids


def test_create_custom_class_proposer_loads_class_and_adds_lifecycle(monkeypatch):
    module = ModuleType("test_custom_proposer_module")
    module.CustomProposer = _CustomProposer
    monkeypatch.setitem(sys.modules, module.__name__, module)
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            model=f"{module.__name__}.CustomProposer",
        )
    )

    proposer = create_custom_class_proposer(vllm_config)

    assert isinstance(proposer, _CustomProposer)
    assert proposer.vllm_config is vllm_config
    assert proposer.load_model("target") is None
    assert proposer.dummy_run(num_tokens=1) is None


def test_set_custom_class_request_ids_trims_to_batch_size():
    proposer = _CustomProposer(SimpleNamespace())

    set_custom_class_request_ids(
        proposer,
        ["req-0", "req-1", "req-2"],
        [[1], [2]],
    )

    assert proposer.request_ids == ("req-0", "req-1")
