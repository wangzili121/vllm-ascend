import torch
from vllm.triton_utils import tl, triton


@triton.jit
def append_sampled_tokens_kernel(
    token_ids_ptr,
    sequence_lengths_ptr,
    sampled_ptr,
    discard_ptr,
    next_token_ids_ptr,
    valid_counts_ptr,
    updated_lengths_ptr,
    history_stride: tl.constexpr,
    sampled_row_stride: tl.constexpr,
    sampled_column_stride: tl.constexpr,
    max_sequence_length: tl.constexpr,
    max_new_tokens: tl.constexpr,
    block_size: tl.constexpr,
    vocab_size: tl.constexpr,
):
    row = tl.program_id(0)
    sequence_length = tl.load(sequence_lengths_ptr + row)
    offsets = tl.arange(0, block_size)
    in_sample = offsets < max_new_tokens
    sampled = tl.load(
        sampled_ptr
        + row * sampled_row_stride
        + offsets * sampled_column_stride,
        mask=in_sample,
        other=-1,
    )
    valid = in_sample & (sampled >= 0) & (sampled < vocab_size)
    first_invalid = tl.min(
        tl.where(in_sample & (valid == 0), offsets, max_new_tokens), axis=0
    )
    discarded = tl.load(discard_ptr + row) != 0
    valid_count = tl.where(discarded, 0, first_invalid)
    valid_count = tl.minimum(
        valid_count,
        tl.maximum(max_sequence_length - sequence_length, 0),
    )

    row_start = row * history_stride
    tl.store(
        token_ids_ptr + row_start + sequence_length + offsets,
        sampled,
        mask=offsets < valid_count,
    )
    updated_length = sequence_length + valid_count
    backup_position = tl.maximum(sequence_length - 1, 0)
    backup_token = tl.load(token_ids_ptr + row_start + backup_position)
    next_token = tl.sum(
        tl.where(offsets == valid_count - 1, sampled, 0), axis=0
    )
    next_token = tl.where(valid_count > 0, next_token, backup_token)

    tl.store(next_token_ids_ptr + row, next_token)
    tl.store(valid_counts_ptr + row, valid_count)
    tl.store(updated_lengths_ptr + row, updated_length)


def append_sampled_tokens(
    token_ids: torch.Tensor,
    sequence_lengths: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    discard_request_mask: torch.Tensor,
    *,
    vocab_size: int,
    max_sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Append sampled prefixes and return next IDs, counts, and new lengths."""
    batch_size = sequence_lengths.shape[0]
    sampled_token_ids = sampled_token_ids[:batch_size]
    max_new_tokens = sampled_token_ids.shape[1]
    if max_sequence_length + max_new_tokens > token_ids.shape[1]:
        raise ValueError("Device token history has insufficient append capacity.")
    if batch_size == 0:
        empty = torch.empty(0, dtype=torch.int32, device=token_ids.device)
        return empty, empty, empty

    outputs = torch.empty(batch_size * 3, dtype=torch.int32, device=token_ids.device)
    next_token_ids, valid_counts, updated_lengths = outputs.chunk(3)
    block_size = triton.next_power_of_2(max_new_tokens)
    append_sampled_tokens_kernel[(batch_size,)](
        token_ids,
        sequence_lengths,
        sampled_token_ids,
        discard_request_mask[:batch_size],
        next_token_ids,
        valid_counts,
        updated_lengths,
        history_stride=token_ids.shape[1],
        sampled_row_stride=sampled_token_ids.stride(0),
        sampled_column_stride=sampled_token_ids.stride(1),
        max_sequence_length=max_sequence_length,
        max_new_tokens=max_new_tokens,
        block_size=block_size,
        vocab_size=vocab_size,
    )
    return next_token_ids, valid_counts, updated_lengths
