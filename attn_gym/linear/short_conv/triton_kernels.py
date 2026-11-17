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

from attn_gym.linear.short_conv.common import packed_positions


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
        resumed = resumed & (tl.load(has_initial_state + row) != 0)
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
def _token_tile(
    sequence, offset, active, batch, token, tokens, BLOCK_T: tl.constexpr, PACKED: tl.constexpr
):
    """Return each token's owning sequence, offset in it, and whether it is active."""
    token_in = token < tokens
    if PACKED:
        owner = tl.load(sequence + token, mask=token_in, other=0)
        position = tl.load(offset + token, mask=token_in, other=0)
        in_use = token_in & (tl.load(active + token, mask=token_in, other=0) != 0)
    else:
        owner = tl.zeros([BLOCK_T], dtype=tl.int64) + batch
        position = token
        in_use = token_in
    return owner, position, in_use


@triton.jit
def _tap_values(
    rows,
    x_token_stride,
    history,
    owner,
    position,
    in_use,
    token,
    channel,
    channel_in,
    channels,
    shift: tl.constexpr,
    WIDTH: tl.constexpr,
    HISTORY: tl.constexpr,
):
    """Return the FP32 input each token reads ``shift`` positions back, from x or the history."""
    inside = in_use & (position >= shift)
    current = tl.load(
        rows + (token - shift)[:, None] * x_token_stride,
        mask=inside[:, None] & channel_in[None, :],
        other=0.0,
    ).to(tl.float32)
    if HISTORY and shift > 0:
        row = owner * (WIDTH - 1) + position + WIDTH - 1 - shift
        past = tl.load(
            history + row[:, None] * channels + channel[None, :],
            mask=(in_use & (position < shift))[:, None] & channel_in[None, :],
            other=0.0,
        ).to(tl.float32)
        current = tl.where(inside[:, None], current, past)
    return current


@triton.jit
def _causal_conv1d_forward_kernel(
    x,
    x_batch_stride,
    x_token_stride,
    weight,
    output,
    upstream,
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
    HISTORY: tl.constexpr,
    ROUTED: tl.constexpr,
    SILU: tl.constexpr,
    GRADIENT: tl.constexpr,
):
    """Write the activated output, or with ``GRADIENT`` the FP32 preactivation gradient."""
    batch = tl.program_id(2).to(tl.int64)
    token = tl.program_id(0).to(tl.int64) * BLOCK_T + tl.arange(0, BLOCK_T)
    channel = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    token_in = token < tokens
    channel_in = channel < channels
    owner, position, in_use = _token_tile(
        sequence, offset, active, batch, token, tokens, BLOCK_T, PACKED
    )
    routed = in_use
    if ROUTED:
        routed = in_use & (tl.load(live + owner, mask=token_in, other=0) != 0)
    rows = x + batch * x_batch_stride + channel[None, :]
    value = tl.zeros([BLOCK_T, BLOCK_C], dtype=tl.float32)
    for shift in tl.static_range(WIDTH):
        current = _tap_values(
            rows, x_token_stride, history, owner, position, in_use, token, channel, channel_in,
            channels, shift, WIDTH, HISTORY,
        )
        weight_tap = tl.load(weight + channel * WIDTH + WIDTH - 1 - shift, mask=channel_in)
        value += current * weight_tap.to(tl.float32)[None, :]
    tile = (batch * tokens + token)[:, None] * channels + channel[None, :]
    tile_in = token_in[:, None] & channel_in[None, :]
    if GRADIENT:
        value_grad = tl.load(upstream + tile, mask=tile_in, other=0.0).to(tl.float32)
        if SILU:
            gate = tl.sigmoid(value)
            value_grad = value_grad * gate * (1.0 + value * (1.0 - gate))
        value = value_grad
    elif SILU:
        value = value * tl.sigmoid(value)
    value = tl.where(routed[:, None], value, 0.0)
    tl.store(output + tile, value.to(output.dtype.element_ty), mask=tile_in)


@triton.jit
def _causal_conv1d_backward_kernel(
    x,
    x_batch_stride,
    x_token_stride,
    weight,
    preactivation_grad,
    grad_x,
    partials,
    history,
    sequence,
    offset,
    active,
    tokens,
    channels,
    WIDTH: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
    PACKED: tl.constexpr,
    HISTORY: tl.constexpr,
):
    """Write the input gradient and this token tile's FP32 weight-gradient partials."""
    batch = tl.program_id(2).to(tl.int64)
    token = tl.program_id(0).to(tl.int64) * BLOCK_T + tl.arange(0, BLOCK_T)
    channel = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    token_in = token < tokens
    channel_in = channel < channels
    owner, position, in_use = _token_tile(
        sequence, offset, active, batch, token, tokens, BLOCK_T, PACKED
    )
    # x[t] reaches output t + shift through tap W - 1 - shift when both lie in one sequence.
    value = tl.zeros([BLOCK_T, BLOCK_C], dtype=tl.float32)
    for shift in tl.static_range(WIDTH):
        later = token + shift
        _, later_position, later_in_use = _token_tile(
            sequence, offset, active, batch, later, tokens, BLOCK_T, PACKED
        )
        reached = later_in_use & (later_position >= shift)
        later_grad = tl.load(
            preactivation_grad + (batch * tokens + later)[:, None] * channels + channel[None, :],
            mask=reached[:, None] & channel_in[None, :],
            other=0.0,
        )
        weight_tap = tl.load(weight + channel * WIDTH + WIDTH - 1 - shift, mask=channel_in)
        value += later_grad * weight_tap.to(tl.float32)[None, :]
    tile = (batch * tokens + token)[:, None] * channels + channel[None, :]
    tile_in = token_in[:, None] & channel_in[None, :]
    tl.store(grad_x + tile, value.to(grad_x.dtype.element_ty), mask=tile_in)

    here = tl.load(preactivation_grad + tile, mask=tile_in, other=0.0)
    rows = x + batch * x_batch_stride + channel[None, :]
    partial_row = (batch * tl.num_programs(0) + tl.program_id(0)) * WIDTH
    for shift in tl.static_range(WIDTH):
        current = _tap_values(
            rows, x_token_stride, history, owner, position, in_use, token, channel, channel_in,
            channels, shift, WIDTH, HISTORY,
        )
        tl.store(
            partials + (partial_row + WIDTH - 1 - shift) * channels + channel,
            tl.sum(here * current, axis=0),
            mask=channel_in,
        )


def _forward_launch(x, weight, output, upstream, history, live, positions, activation):
    batches, tokens, channels = x.shape
    block_t, block_c = 32, min(128, triton.next_power_of_2(channels))
    sequence, offset, active = (None, None, None) if positions is None else positions
    grid = (triton.cdiv(tokens, block_t), triton.cdiv(channels, block_c), batches)
    _causal_conv1d_forward_kernel[grid](
        x,
        x.stride(0),
        x.stride(1),
        weight,
        output,
        upstream,
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
        HISTORY=history is not None,
        ROUTED=live is not None,
        SILU=activation == "silu",
        GRADIENT=upstream is not None,
        num_warps=4,
    )
    return output


def causal_conv1d_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    history: torch.Tensor,
    live: torch.Tensor,
    positions: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    activation: str | None,
) -> torch.Tensor:
    """Convolve ``x`` ``[B, T, C]`` after each sequence's ``history`` ``[N, W - 1, C]``.

    ``positions`` is ``packed_positions`` for packed input and ``None`` for dense rows.
    Outputs of sequences whose ``live`` entry is false, and of tokens past the packed active
    endpoint, are zero.
    """
    output = x.new_empty(x.shape)
    return _forward_launch(x, weight, output, None, history, live, positions, activation)


def _state_gradient(preactivation_grad, weight, cu_seqlens, state_shape, tokens):
    """Gradient of the history: token ``o`` of a sequence reads row ``o + tap`` while ``o + tap
    < W - 1``."""
    sequences, rows, channels = state_shape
    flat = preactivation_grad.view(-1, channels)
    if cu_seqlens is None:
        starts = torch.arange(sequences, device=flat.device) * tokens
        lengths = torch.full((sequences,), tokens, device=flat.device)
    else:
        starts = cu_seqlens[:-1].long()
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).long()
    taps = weight.float()
    grad = flat.new_zeros(state_shape)
    for position in range(rows):
        read = torch.where(
            (lengths > position)[:, None],
            flat[(starts + position).clamp(max=flat.shape[0] - 1)],
            0.0,
        )
        for tap in range(rows - position):
            grad[:, position + tap] += read * taps[:, tap]
    return grad


@torch.library.custom_op(
    "attn_gym::_portable_short_conv_fwd",
    mutates_args=(),
    device_types="cuda",
    schema="(Tensor x, Tensor weight, Tensor? cu_seqlens, Tensor? initial_state,"
    " str? activation) -> Tensor",
)
def short_conv_forward(x, weight, cu_seqlens, initial_state, activation):
    """Training forward of the portable backend; differentiable through ``register_autograd``."""
    positions = None if cu_seqlens is None else packed_positions(cu_seqlens, x.shape[1])
    output = x.new_empty(x.shape)
    return _forward_launch(x, weight, output, None, initial_state, None, positions, activation)


@short_conv_forward.register_fake
def _short_conv_forward_fake(x, weight, cu_seqlens, initial_state, activation):
    return x.new_empty(x.shape)


@torch.library.custom_op(
    "attn_gym::_portable_short_conv_bwd",
    mutates_args=(),
    device_types="cuda",
    schema="(Tensor x, Tensor weight, Tensor grad_output, Tensor? cu_seqlens,"
    " Tensor? initial_state, str? activation, bool state_gradient) -> (Tensor, Tensor, Tensor)",
)
def short_conv_backward(
    x, weight, grad_output, cu_seqlens, initial_state, activation, state_gradient
):
    """Return the input, weight and (when requested) history gradients."""
    batches, tokens, channels = x.shape
    width = weight.shape[1]
    positions = None if cu_seqlens is None else packed_positions(cu_seqlens, tokens)
    preactivation_grad = _forward_launch(
        x,
        weight,
        torch.empty(x.shape, device=x.device, dtype=torch.float32),
        grad_output.contiguous(),
        initial_state,
        None,
        positions,
        activation,
    )
    block_t, block_c = 32, min(128, triton.next_power_of_2(channels))
    blocks = triton.cdiv(tokens, block_t)
    grad_x = x.new_empty(x.shape)
    partials = torch.empty(
        batches * blocks, width, channels, device=x.device, dtype=torch.float32
    )
    sequence, offset, active = (None, None, None) if positions is None else positions
    _causal_conv1d_backward_kernel[(blocks, triton.cdiv(channels, block_c), batches)](
        x,
        x.stride(0),
        x.stride(1),
        weight,
        preactivation_grad,
        grad_x,
        partials,
        initial_state,
        sequence,
        offset,
        active,
        tokens,
        channels,
        WIDTH=width,
        BLOCK_T=block_t,
        BLOCK_C=block_c,
        PACKED=positions is not None,
        HISTORY=initial_state is not None,
        num_warps=4,
    )
    grad_weight = partials.sum(0).t().contiguous().to(weight.dtype)
    if not state_gradient:
        return grad_x, grad_weight, x.new_empty(0)
    grad_state = _state_gradient(
        preactivation_grad, weight, cu_seqlens, initial_state.shape, tokens
    )
    return grad_x, grad_weight, grad_state.to(initial_state.dtype)


@short_conv_backward.register_fake
def _short_conv_backward_fake(
    x, weight, grad_output, cu_seqlens, initial_state, activation, state_gradient
):
    state = initial_state.new_empty(initial_state.shape) if state_gradient else x.new_empty(0)
    return x.new_empty(x.shape), weight.new_empty(weight.shape), state


def _setup_context(ctx, inputs, output):
    x, weight, cu_seqlens, initial_state, activation = inputs
    ctx.save_for_backward(x, weight, cu_seqlens, initial_state)
    ctx.activation = activation


def _backward(ctx, grad_output):
    x, weight, cu_seqlens, initial_state = ctx.saved_tensors
    state_gradient = initial_state is not None and ctx.needs_input_grad[3]
    grad_x, grad_weight, grad_state = short_conv_backward(
        x, weight, grad_output, cu_seqlens, initial_state, ctx.activation, state_gradient
    )
    return grad_x, grad_weight, None, grad_state if state_gradient else None, None


short_conv_forward.register_autograd(_backward, setup_context=_setup_context)


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
        state_indices,
        has_initial_state,
        channels,
        WIDTH=weight.shape[1],
        BLOCK=block,
        ROUTED=state_indices is not None,
        MASKED=has_initial_state is not None,
        SILU=activation == "silu",
        num_warps=4,
    )
    return output
