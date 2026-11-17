# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Triton kernels behind the portable short convolution.

One program advances one channel block of one sequence's history, so a decode step is a single
launch with no host work, as CUDA Graph capture requires. Masked loads never touch the history
of null or fresh slots, which may hold NaNs.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _causal_conv1d_decode_kernel(
    x,
    x_row_stride,
    weight,
    output,
    state,
    state_slot_stride,
    state_indices,
    has_initial_state,
    channels,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    ROUTED: tl.constexpr,
    MASKED: tl.constexpr,
    SILU: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    channel = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    in_range = channel < channels
    if ROUTED:
        slot = tl.load(state_indices + row).to(tl.int64)
        live = slot > 0
    else:
        slot = row
        live = row >= 0
    resumed = live
    if MASKED:
        resumed = resumed & tl.load(has_initial_state + row)
    history = state + slot * state_slot_stride + channel
    taps = weight + channel * WIDTH
    newest = tl.load(x + row * x_row_stride + channel, mask=in_range & live, other=0.0)
    last_tap = tl.load(taps + WIDTH - 1, mask=in_range, other=0.0).to(tl.float32)
    value = newest.to(tl.float32) * last_tap
    for step in tl.static_range(WIDTH - 1):
        tap = WIDTH - 2 - step
        previous = tl.load(history + tap * channels, mask=in_range & resumed, other=0.0)
        weight_tap = tl.load(taps + tap, mask=in_range, other=0.0).to(tl.float32)
        value += previous.to(tl.float32) * weight_tap
    if SILU:
        value = value * tl.sigmoid(value)
    value = tl.where(live, value, 0.0)
    tl.store(output + row * channels + channel, value.to(output.dtype.element_ty), mask=in_range)
    for position in tl.static_range(WIDTH - 2):
        shifted = tl.load(history + (position + 1) * channels, mask=in_range & resumed, other=0.0)
        tl.store(history + position * channels, shifted, mask=in_range & live)
    tl.store(history + (WIDTH - 2) * channels, newest, mask=in_range & live)


@triton.jit
def _causal_conv1d_forward_kernel(
    x,
    x_batch_stride,
    x_token_stride,
    weight,
    output,
    history,
    live,
    sequence,
    offset,
    active,
    tokens,
    channels,
    WIDTH: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
    PACKED: tl.constexpr,
    SILU: tl.constexpr,
):
    batch = tl.program_id(2).to(tl.int64)
    token = tl.program_id(0).to(tl.int64) * BLOCK_T + tl.arange(0, BLOCK_T)
    channel = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    token_in = token < tokens
    channel_in = channel < channels
    if PACKED:
        owner = tl.load(sequence + token, mask=token_in, other=0)
        position = tl.load(offset + token, mask=token_in, other=0)
        in_use = token_in & tl.load(active + token, mask=token_in, other=0)
    else:
        owner = tl.zeros([BLOCK_T], dtype=tl.int64) + batch
        position = token
        in_use = token_in
    routed = in_use & tl.load(live + owner, mask=token_in, other=0)
    rows = x + batch * x_batch_stride + channel[None, :]
    value = tl.zeros([BLOCK_T, BLOCK_C], dtype=tl.float32)
    for shift in tl.static_range(WIDTH):
        tap = WIDTH - 1 - shift
        inside = in_use & (position >= shift)
        current = tl.load(
            rows + (token - shift)[:, None] * x_token_stride,
            mask=inside[:, None] & channel_in[None, :],
            other=0.0,
        ).to(tl.float32)
        if shift > 0:
            row = owner * (WIDTH - 1) + position + tap
            past = tl.load(
                history + row[:, None] * channels + channel[None, :],
                mask=(in_use & (position < shift))[:, None] & channel_in[None, :],
                other=0.0,
            ).to(tl.float32)
            current = tl.where(inside[:, None], current, past)
        weight_tap = tl.load(weight + channel * WIDTH + tap, mask=channel_in, other=0.0)
        value += current * weight_tap.to(tl.float32)[None, :]
    if SILU:
        value = value * tl.sigmoid(value)
    value = tl.where(routed[:, None], value, 0.0)
    tl.store(
        output + (batch * tokens + token)[:, None] * channels + channel[None, :],
        value.to(output.dtype.element_ty),
        mask=token_in[:, None] & channel_in[None, :],
    )


def causal_conv1d_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    history: torch.Tensor,
    live: torch.Tensor,
    positions: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    activation: str | None,
) -> torch.Tensor:
    """Convolve ``x`` ``[B, T, C]`` after each sequence's ``history`` ``[N, W - 1, C]``.

    ``positions`` is ``portable._packed_positions`` for packed input and ``None`` for dense
    rows. Outputs of sequences whose ``live`` entry is false, and of tokens past the packed
    active endpoint, are zero.
    """
    batches, tokens, channels = x.shape
    output = x.new_empty(batches, tokens, channels)
    block_t, block_c = 32, min(128, triton.next_power_of_2(channels))
    sequence, offset, active = (output, output, output) if positions is None else positions
    grid = (triton.cdiv(tokens, block_t), triton.cdiv(channels, block_c), batches)
    _causal_conv1d_forward_kernel[grid](
        x,
        x.stride(0),
        x.stride(1),
        weight,
        output,
        history,
        live,
        sequence,
        offset,
        active,
        tokens,
        channels,
        WIDTH=weight.shape[1],
        BLOCK_T=block_t,
        BLOCK_C=block_c,
        PACKED=positions is not None,
        SILU=activation == "silu",
        num_warps=4,
    )
    return output


def causal_conv1d_decode(
    x: torch.Tensor,
    weight: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor | None,
    has_initial_state: torch.Tensor | None,
    activation: str | None,
) -> torch.Tensor:
    """Launch the validated one-token step; see ``portable.causal_conv1d_decode``."""
    rows, channels = x.shape
    output = x.new_empty(rows, channels)
    block = min(1024, triton.next_power_of_2(channels))
    _causal_conv1d_decode_kernel[(rows, triton.cdiv(channels, block))](
        x,
        x.stride(0),
        weight,
        output,
        state,
        state.stride(0),
        output if state_indices is None else state_indices,
        output if has_initial_state is None else has_initial_state,
        channels,
        WIDTH=weight.shape[1],
        BLOCK=block,
        ROUTED=state_indices is not None,
        MASKED=has_initial_state is not None,
        SILU=activation == "silu",
        num_warps=4,
    )
    return output
