# Model-aware custom proposers

## Motivation

Upstream vLLM's `custom_class` proposer is intentionally token-only. It is a
good fit for retrieval and heuristic proposers, but a neural proposer also
needs model loading, distributed state, attention metadata, and KV cache
initialization. Reimplementing those facilities in each plugin would duplicate
the NPU model runner and make cache ownership unsafe.

vLLM-Ascend therefore provides a small, optional extension to `custom_class`.
It does not register model-specific methods or architectures in the framework.

## Contract

A model-aware proposer declares:

```python
class MyProposer:
    npu_model_runner_api_version = 1

    def __init__(self, vllm_config): ...
    def bind_runtime(self, runtime): ...
```

`runtime` is a `ModelAwareCustomProposerRuntime` containing the API version,
`VllmConfig`, device, and owning `NPUModelRunner`. The raw runner is exposed
because built-in Ascend model proposers already use it for input buffers,
attention metadata, cache groups, and distributed execution. A plugin can
delegate to that implementation instead of maintaining a second runner.

Plugin options can be passed under
`additional_config.custom_proposer_config`. vLLM-Ascend admits this opaque
namespace while the custom proposer owns validation of its nested schema.

After binding, the proposer participates in the existing model-proposer
lifecycle:

1. `load_model(target_model)` loads the draft and may share target modules.
2. Draft attention layers register cache aliases through
   `kv_sharing_target_layer_name` before cache planning.
3. `initialize_attn_backend(...)` receives the finalized target cache plan.
4. Input preparation and `_propose(...)` use the built-in padded model path.
5. `dummy_run(...)` participates in memory profiling.

The model can implement `bind_shared_backbone(target_language_model)` to share
target modules after normal embedding and LM-head sharing. Cache aliases are
recognized on any `AttentionLayerBase`, so plugin-defined wrappers do not
allocate duplicate history caches.

## Compatibility

The extension is capability-based and versioned. Proposers without the marker
keep the upstream token-only behavior. An unsupported version or missing
`bind_runtime` fails during runner setup. Model-aware custom proposers are
eager-only until a plugin explicitly implements and validates graph support.

Model architecture registration, checkpoint validation, quantization, and
algorithm semantics belong to the plugin, not vLLM-Ascend.
