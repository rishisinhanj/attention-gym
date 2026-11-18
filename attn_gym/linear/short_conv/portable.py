# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""PyTorch short convolution for platforms without the CuTeDSL backend.

``attn_gym.linear.short_conv`` resolves these operations when
:mod:`attn_gym.linear.short_conv.cute` does not import, for example on ROCm. They take the
same arguments and follow the same contracts on any device PyTorch supports. Every tap is
accumulated in FP32, newest first, and the activation is applied to the FP32 sum. Paged prefill
and decode read routing and freshness on the device, so CUDA Graphs capture them.

Only the built-in activations are available, and ``kernel_options``, which select CuTeDSL
schedules, are ignored.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_ACTIVATIONS = {None: lambda value: value, "silu": F.silu}


def _activation(name: str | None):
    if name not in _ACTIVATIONS:
        raise ValueError(f"unknown activation {name!r}: the PyTorch backend has None and 'silu'")
    return _ACTIVATIONS[name]


def _check_inputs(x, weight, cu_seqlens, rank: int) -> int:
    """Reject shapes that would broadcast silently; return the number of sequences."""
    if x.ndim != rank or weight.ndim != 2 or weight.shape[0] != x.shape[-1]:
        raise ValueError(
            f"x must have {rank} dims and weight shape [C, W] with C = x.shape[-1], "
            f"got {tuple(x.shape)} and {tuple(weight.shape)}"
        )
    if cu_seqlens is None:
        return x.shape[0]
    if x.shape[0] != 1:
        raise ValueError(f"packed input must have shape [1, T, C], got {tuple(x.shape)}")
    return cu_seqlens.shape[0] - 1


def _check_state(name, x, weight, state, state_indices, has_initial_state, sequences) -> None:
    """Validate a history pool and its per-sequence routing for the in-place operations."""
    if torch.is_grad_enabled() and any(t.requires_grad for t in (x, weight, state)):
        raise RuntimeError(
            f"{name} is inference-only and has no backward; use causal_conv1d for autograd"
        )
    width, channels = weight.shape[1], weight.shape[0]
    slots = state.shape[0] if state_indices is not None else sequences
    if width < 2 or state.shape != (slots, width - 1, channels):
        raise ValueError(
            f"{name} needs W >= 2 and state of shape [num_slots, W - 1, C] (one slot per "
            f"sequence without state_indices), got weight {tuple(weight.shape)} and state "
            f"{tuple(state.shape)}"
        )
    vectors = {"state_indices": state_indices, "has_initial_state": has_initial_state}
    for label, vector in vectors.items():
        if vector is not None and vector.shape != (sequences,):
            raise ValueError(f"{label} must have shape ({sequences},), got {tuple(vector.shape)}")


def _packed_positions(cu_seqlens, tokens):
    """Each token's sequence, its offset in that sequence, and whether it is before the end."""
    positions = torch.arange(tokens, device=cu_seqlens.device, dtype=cu_seqlens.dtype)
    sequence = torch.searchsorted(cu_seqlens[1:], positions, right=True)
    sequence = sequence.clamp_(max=cu_seqlens.shape[0] - 2)
    offset = positions.long() - cu_seqlens[sequence].long()
    return sequence, offset, positions < cu_seqlens[-1]


def _convolve(x, weight, cu_seqlens, history, positions=None):
    """FP32 causal convolution of ``x`` ``[B, T, C]`` by ``weight`` ``[C, W]``.

    ``history`` ``[N, W - 1, C]``, zeros when None, precedes each batch row, or each
    ``cu_seqlens`` interval of the single packed row.
    """
    tokens, width = x.shape[1], weight.shape[1]
    taps = weight.float()
    if cu_seqlens is None:
        if history is None:
            history = x.new_zeros(x.shape[0], width - 1, x.shape[2])
        window = torch.cat((history, x), dim=1).float()
        output = window[:, width - 1 :] * taps[:, width - 1]
        for tap in range(width - 2, -1, -1):
            output = output + window[:, tap : tap + tokens] * taps[:, tap]
        return output
    sequence, offset, active = positions or _packed_positions(cu_seqlens, tokens)
    # Positions past the active endpoint read zeros, so a NaN suffix reaches neither the outputs
    # nor the gradients.
    active = active[:, None]
    padded = F.pad(x[0].float(), (0, 0, width - 1, 0))
    # Gather from an FP32 copy so the history gradient also accumulates in FP32.
    history = None if history is None else history.float()
    output = torch.where(active, padded[width - 1 :], 0.0) * taps[:, width - 1]
    for tap in range(width - 2, -1, -1):
        earlier = 0.0
        if history is not None:
            row = (offset + tap).clamp(max=width - 2)
            earlier = torch.where(active, history[sequence, row], 0.0)
        inside = (offset >= width - 1 - tap)[:, None] & active
        output = output + torch.where(inside, padded[tap : tap + tokens], earlier) * taps[:, tap]
    return output.unsqueeze(0)


def _final_history(x, history, cu_seqlens):
    """Each sequence's newest ``W - 1`` inputs, completed from ``history`` when it is shorter.

    Selection only, so the bits of the inputs, including signed zeros, are kept.
    """
    if cu_seqlens is None:
        return torch.cat((history, x), dim=1)[:, x.shape[1] :]
    length = history.shape[1]
    starts, ends = cu_seqlens[:-1].long(), cu_seqlens[1:].long()
    rows = torch.arange(length, device=x.device)
    positions = ends[:, None] - length + rows
    from_input = (positions >= starts[:, None])[:, :, None]
    carried = (rows + (ends - starts)[:, None]).clamp(max=length - 1)
    earlier = history.gather(1, carried[:, :, None].expand(-1, -1, x.shape[2]))
    return torch.where(from_input, x[0, positions.clamp(0, x.shape[1] - 1)], earlier)


def causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    activation: str | None = None,
    cu_seqlens: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    return_final_state: bool = False,
    kernel_options: dict | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Apply causal depthwise convolution with an optional activation.

    Same contract as :func:`attn_gym.linear.short_conv.cute.causal_conv1d`, through PyTorch
    autograd to ``x``, ``weight``, and ``initial_state``.
    """
    del kernel_options
    epilogue = _activation(activation)
    sequences = _check_inputs(x, weight, cu_seqlens, rank=3)
    width = weight.shape[1]
    if initial_state is not None and initial_state.shape != (sequences, width - 1, x.shape[2]):
        raise ValueError(
            f"initial_state must have shape {(sequences, width - 1, x.shape[2])}, "
            f"got {tuple(initial_state.shape)}"
        )
    history = None if width == 1 else initial_state
    output = epilogue(_convolve(x, weight, cu_seqlens, history)).to(x.dtype)
    if not return_final_state:
        return output
    if history is None:
        history = x.new_zeros(sequences, width - 1, x.shape[2])
    return output, _final_history(x, history, cu_seqlens)


def paged_causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    state_cache: torch.Tensor,
    state_indices: torch.Tensor,
    *,
    activation: str | None = None,
    cu_seqlens: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply multi-token convolution while advancing a paged history in place.

    Same contract as :func:`attn_gym.linear.short_conv.cute.paged_causal_conv1d`.
    Inference-only.
    """
    epilogue = _activation(activation)
    sequences = _check_inputs(x, weight, cu_seqlens, rank=3)
    _check_state(
        "paged_causal_conv1d", x, weight, state_cache, state_indices, has_initial_state, sequences
    )
    live = state_indices > 0
    # Null routes read slot zero and write back what they read, so the pool keeps its bits.
    slots = torch.where(live, state_indices, 0).long()
    stored = state_cache[slots]
    resumed = live if has_initial_state is None else live & has_initial_state
    history = torch.where(resumed[:, None, None], stored, 0)
    if cu_seqlens is None:
        positions, owned = None, live[:, None, None]
    else:
        positions = _packed_positions(cu_seqlens, x.shape[1])
        owned = live[positions[0]][None, :, None]
    output = epilogue(_convolve(x, weight, cu_seqlens, history, positions))
    final = _final_history(x, history, cu_seqlens)
    state_cache[slots] = torch.where(live[:, None, None], final, stored)
    return torch.where(owned, output, 0.0).to(x.dtype)


def causal_conv1d_decode(
    x: torch.Tensor,
    weight: torch.Tensor,
    state: torch.Tensor,
    *,
    activation: str | None = None,
    state_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    kernel_options: dict | None = None,
) -> torch.Tensor:
    """Advance the convolution by one token per sequence over a paged history.

    Same contract as :func:`attn_gym.linear.short_conv.cute.causal_conv1d_decode`.
    Inference-only.
    """
    del kernel_options
    epilogue = _activation(activation)
    sequences = _check_inputs(x, weight, None, rank=2)
    _check_state(
        "causal_conv1d_decode", x, weight, state, state_indices, has_initial_state, sequences
    )
    if state_indices is None:
        live, stored, resumed = None, state, has_initial_state
    else:
        live = state_indices > 0
        slots = torch.where(live, state_indices, 0).long()
        stored = state[slots]
        resumed = live if has_initial_state is None else live & has_initial_state
    history = stored if resumed is None else torch.where(resumed[:, None, None], stored, 0)
    window = torch.cat((history, x[:, None]), dim=1)
    taps = weight.float()
    output = window[:, -1].float() * taps[:, -1]
    for tap in range(weight.shape[1] - 2, -1, -1):
        output = output + window[:, tap].float() * taps[:, tap]
    output = epilogue(output)
    if live is None:
        state.copy_(window[:, 1:])
        return output.to(x.dtype)
    state[slots] = torch.where(live[:, None, None], window[:, 1:], stored)
    return torch.where(live[:, None], output, 0.0).to(x.dtype)


__all__ = ["causal_conv1d", "causal_conv1d_decode", "paged_causal_conv1d"]
