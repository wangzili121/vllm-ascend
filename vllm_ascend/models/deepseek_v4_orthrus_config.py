# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Constants shared by the DeepSeek-V4 Orthrus model and proposer."""

ORTHRUS_HEAD_FORMAT = "orthrus-dsv4-head-v2"
ORTHRUS_BLOCK_SIZE = 32
ORTHRUS_NUM_SPECULATIVE_TOKENS = ORTHRUS_BLOCK_SIZE - 1
