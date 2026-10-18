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
    print("token history smoke passed", {"device": device})


if __name__ == "__main__":
    main()
