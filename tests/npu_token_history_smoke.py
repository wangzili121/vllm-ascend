import os

import torch

from vllm_ascend.ops.triton.spec_decode.token_history import append_sampled_tokens


def main() -> None:
    device = os.environ.get("TOKEN_HISTORY_SMOKE_DEVICE", "npu:0")
    history = torch.zeros((3, 36), dtype=torch.int32, device=device)
    history[0, :2] = torch.tensor([10, 11], dtype=torch.int32, device=device)
    history[1, :31] = torch.arange(31, dtype=torch.int32, device=device)
    history[2, :2] = torch.tensor([20, 21], dtype=torch.int32, device=device)
    lengths = torch.tensor([2, 31, 2], dtype=torch.int32, device=device)
    sampled = torch.tensor(
        [[12, -1, 13], [40, 41, 42], [22, 23, 24]],
        dtype=torch.int32,
        device=device,
    )
    discarded = torch.tensor([False, False, True], device=device)

    next_ids, counts, updated = append_sampled_tokens(
        history,
        lengths,
        sampled,
        discarded,
        vocab_size=100,
        max_sequence_length=32,
    )

    assert next_ids.cpu().tolist() == [12, 40, 21]
    assert counts.cpu().tolist() == [1, 1, 0]
    assert updated.cpu().tolist() == [3, 32, 2]
    assert history[0, :3].cpu().tolist() == [10, 11, 12]
    assert int(history[1, 31].cpu()) == 40
    assert history[2, :3].cpu().tolist() == [20, 21, 0]

    # Async sampling can return a narrow view whose row stride is larger than
    # its visible width. Every request must read from its own physical row.
    strided_source = torch.full(
        (4, 8), -1, dtype=torch.int32, device=device
    )
    strided_source[:, 0] = torch.tensor(
        [70, 71, 72, 73], dtype=torch.int32, device=device
    )
    strided_sampled = strided_source[:, :1]
    assert not strided_sampled.is_contiguous()

    strided_history = torch.zeros((4, 8), dtype=torch.int32, device=device)
    strided_lengths = torch.zeros(4, dtype=torch.int32, device=device)
    strided_discarded = torch.zeros(4, dtype=torch.bool, device=device)
    strided_next, strided_counts, strided_updated = append_sampled_tokens(
        strided_history,
        strided_lengths,
        strided_sampled,
        strided_discarded,
        vocab_size=100,
        max_sequence_length=4,
    )

    assert strided_next.cpu().tolist() == [70, 71, 72, 73]
    assert strided_counts.cpu().tolist() == [1, 1, 1, 1]
    assert strided_updated.cpu().tolist() == [1, 1, 1, 1]
    assert strided_history[:, 0].cpu().tolist() == [70, 71, 72, 73]
    print("token history smoke passed", {"device": device})


if __name__ == "__main__":
    main()
