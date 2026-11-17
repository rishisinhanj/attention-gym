# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Portable PyTorch causal depthwise convolution.

``attn_gym.linear.short_conv`` resolves these operations when the CuTeDSL backend does not
import. They take the same arguments as :mod:`attn_gym.linear.short_conv.cute` and return the
same results, histories, and in-place state updates, on any device PyTorch supports:

* every tap is multiplied and accumulated in FP32, newest tap first, and the activation runs on
  the FP32 sum before rounding to the input dtype;
* :func:`causal_conv1d` differentiates through autograd, including the weight and initial-state
  gradients;
* paged prefill and decode read slot routing, freshness, and packed boundaries on the device
  and never synchronize with the host, so CUDA Graphs capture them and replay changed metadata;
* on GPUs, where Triton imports, decode and the paged prefill convolution are Triton kernels
  (``triton_kernels``); training keeps the differentiable PyTorch path.

Only the built-in activations exist here: :func:`register_activation` takes CuTeDSL
expressions. ``kernel_options`` schedules are validated as CuTeDSL validates them and otherwise
ignored, since they choose CuTeDSL register tilings.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

from attn_gym.linear.short_conv.common import (
    DECODE_KERNEL_OPTION_SPEC,
    KERNEL_OPTION_SPEC,
    ShortConvConfig,
    ShortConvDecodeKernelOptions,
    ShortConvKernelOptions,
    ShortConvTunedConfig,
    final_conv_state,
    packed_positions,
)
from attn_gym.types import validate_kernel_options

try:
    from attn_gym.linear.short_conv import triton_kernels as _triton_kernels
except ImportError:
    _triton_kernels = None

SHORT_CONV_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

_ACTIVATIONS: dict[str | None, Callable[[torch.Tensor], torch.Tensor]] = {
    None: lambda value: value,
    "silu": F.silu,
}


def _resolve_activation(activation: str | None) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return the FP32 epilogue for a built-in activation name."""
    if activation not in _ACTIVATIONS:
        names = ", ".join(repr(name) for name in _ACTIVATIONS if name is not None)
        raise ValueError(
            f"unknown activation {activation!r}; expected None or one of {names} "
            "(registered activations need the CuTeDSL backend)"
        )
    return _ACTIVATIONS[activation]


def _validate_channel_contiguous_input(x: torch.Tensor) -> None:
    """Validate the read-only input without constraining outer strides."""
    if x.dtype not in SHORT_CONV_DTYPES or x.stride(-1) != 1:
        raise ValueError("x must be an FP16, BF16, or FP32 tensor with contiguous channels")


def _validate_inputs(
    x: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
) -> None:
    """Validate the shared training and paged-prefill tensor contract."""
    if x.ndim != 3:
        raise ValueError(f"x must have shape [B, T, C], got {tuple(x.shape)}")
    if x.shape[0] < 1 or x.shape[1] < 1 or x.shape[2] < 1:
        raise ValueError(f"x must have positive B, T, and C dimensions, got {tuple(x.shape)}")
    _validate_channel_contiguous_input(x)
    if weight.ndim != 2 or weight.shape[0] != x.shape[2] or weight.shape[1] < 1:
        raise ValueError(
            f"weight must have shape [C, W] with W positive, got {tuple(weight.shape)}"
        )
    if weight.dtype != x.dtype or weight.device != x.device or not weight.is_contiguous():
        raise ValueError("weight must match x dtype and be contiguous on x.device")
    if cu_seqlens is not None:
        if x.shape[0] != 1:
            raise ValueError("packed cu_seqlens require x to have batch size 1")
        if cu_seqlens.ndim != 1 or cu_seqlens.shape[0] < 2:
            raise ValueError("cu_seqlens must have shape [num_sequences + 1]")
        if (
            cu_seqlens.dtype != torch.int32
            or cu_seqlens.device != x.device
            or not cu_seqlens.is_contiguous()
        ):
            raise ValueError("cu_seqlens must be contiguous int32 on x.device")
    if initial_state is not None:
        state_sequences = x.shape[0] if cu_seqlens is None else cu_seqlens.shape[0] - 1
        expected_state = (state_sequences, weight.shape[1] - 1, x.shape[2])
        if initial_state.shape != expected_state:
            raise ValueError(
                f"initial_state must have shape {expected_state}, got {tuple(initial_state.shape)}"
            )
        if (
            initial_state.dtype != x.dtype
            or initial_state.device != x.device
            or not initial_state.is_contiguous()
        ):
            raise ValueError("initial_state must match x dtype and be contiguous on x.device")


def _validate_config(config: ShortConvConfig, channels: int, name: str) -> None:
    """Reject schedules the CuTeDSL backend would reject, so options stay portable."""
    if config.threads < 32 or config.threads > 1024 or config.threads % 32 != 0:
        raise ValueError(f"{name}.threads must be a warp multiple in [32, 1024]")
    if config.channels_per_thread < 1 or channels % config.channels_per_thread != 0:
        raise ValueError(f"C must be divisible by positive {name}.channels_per_thread")
    if config.times_per_block < 1:
        raise ValueError(f"{name}.times_per_block must be positive")


def _validate_state_pool(x: torch.Tensor, weight: torch.Tensor, state: torch.Tensor) -> None:
    """Validate a mutable history pool of compact, non-overlapping ``[W - 1, C]`` slots."""
    channels = x.shape[-1]
    if state.dtype != x.dtype or state.device != x.device:
        raise ValueError("state must match x dtype and be on x.device")
    if state.stride()[1:] != (channels, 1):
        raise ValueError("state must be contiguous within each [W - 1, C] slot")
    if state.stride(0) < (weight.shape[1] - 1) * channels:
        raise ValueError("state slots must not overlap")


def _validate_sequence_vector(
    x: torch.Tensor,
    tensor: torch.Tensor | None,
    sequences: int,
    dtype: torch.dtype,
    name: str,
) -> None:
    """Validate an optional contiguous per-sequence vector on ``x.device``."""
    if tensor is not None and (
        tensor.shape != (sequences,)
        or tensor.dtype != dtype
        or tensor.device != x.device
        or not tensor.is_contiguous()
    ):
        dtype_name = str(dtype).removeprefix("torch.")
        raise ValueError(
            f"{name} must be contiguous {dtype_name} with shape ({sequences},) on x.device"
        )


def _validate_decode_inputs(
    x: torch.Tensor,
    weight: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor | None,
    has_initial_state: torch.Tensor | None = None,
) -> None:
    """Validate the one-token in-place decode tensor contract."""
    if x.ndim != 2:
        raise ValueError(f"x must have shape [num_sequences, C], got {tuple(x.shape)}")
    if x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError(
            f"x must have positive sequence and channel dimensions, got {tuple(x.shape)}"
        )
    _validate_channel_contiguous_input(x)
    if weight.ndim != 2 or weight.shape[0] != x.shape[1] or weight.shape[1] < 2:
        raise ValueError(
            f"weight must have shape [{x.shape[1]}, W] with W >= 2, got {tuple(weight.shape)}"
        )
    if weight.dtype != x.dtype or weight.device != x.device or not weight.is_contiguous():
        raise ValueError("weight must match x dtype and be contiguous on x.device")

    sequences, channels = x.shape
    slots = sequences if state_indices is None else (state.shape[0] if state.ndim == 3 else -1)
    expected_state = (slots, weight.shape[1] - 1, channels)
    if slots < 1 or state.ndim != 3 or state.shape != expected_state:
        raise ValueError(f"state must have shape {expected_state}, got {tuple(state.shape)}")
    _validate_state_pool(x, weight, state)
    _validate_sequence_vector(x, state_indices, sequences, torch.int32, "state_indices")
    _validate_sequence_vector(x, has_initial_state, sequences, torch.bool, "has_initial_state")


def _validate_paged_inputs(
    x: torch.Tensor,
    weight: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
) -> None:
    """Validate multi-token convolution over a mutable paged history pool."""
    _validate_inputs(x, weight, cu_seqlens)
    if weight.shape[1] < 2:
        raise ValueError("paged causal convolution requires W >= 2; width one carries no history")

    sequences = x.shape[0] if cu_seqlens is None else cu_seqlens.shape[0] - 1
    channels = x.shape[2]
    expected_tail = (weight.shape[1] - 1, channels)
    if state.ndim != 3 or state.shape[1:] != expected_tail or state.shape[0] < 1:
        raise ValueError(
            f"state must have shape [num_slots, {expected_tail[0]}, {channels}], "
            f"got {tuple(state.shape)}"
        )
    _validate_state_pool(x, weight, state)
    _validate_sequence_vector(x, state_indices, sequences, torch.int32, "state_indices")
    _validate_sequence_vector(x, has_initial_state, sequences, torch.bool, "has_initial_state")


def _reject_autograd(name: str, *tensors: torch.Tensor) -> None:
    """Refuse gradient-tracking inputs to the inference-only stateful operations."""
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in tensors):
        raise RuntimeError(
            f"{name} is inference-only and has no backward; use "
            "causal_conv1d for training or call under torch.no_grad()"
        )


def _preactivation(
    x: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    positions: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Return the FP32 causal convolution of ``x`` ``[B, T, C]`` with ``weight`` ``[C, W]``.

    ``initial_state`` ``[N, W - 1, C]`` (zeros when ``None``) holds the history before each
    sequence: every batch row, or every ``cu_seqlens`` interval of the single packed row.
    ``positions`` is ``packed_positions(cu_seqlens, T)`` when the caller already has it.
    """
    tokens, width = x.shape[1], weight.shape[1]
    taps = weight.float()
    if cu_seqlens is None:
        history = (
            x.new_zeros(x.shape[0], width - 1, x.shape[2])
            if initial_state is None
            else initial_state
        )
        window = torch.cat((history, x), dim=1).float()
        output = window[:, width - 1 :] * taps[:, width - 1]
        for tap in range(width - 2, -1, -1):
            output = output + window[:, tap : tap + tokens] * taps[:, tap]
        return output

    sequence, offset, active = positions or packed_positions(cu_seqlens, tokens)
    # Inactive tokens read zeros, so a NaN suffix reaches neither outputs nor gradients.
    active = active[:, None]
    padded = F.pad(x[0].float(), (0, 0, width - 1, 0))
    history = None if initial_state is None else initial_state.float()
    output = torch.where(active, padded[width - 1 :], 0.0) * taps[:, width - 1]
    for tap in range(width - 2, -1, -1):
        # Tap ``tap`` reads ``shift`` tokens back; earlier positions come from the history.
        shift = width - 1 - tap
        inside = (offset >= shift)[:, None] & active
        if history is None:
            outside = 0.0
        else:
            row = (offset + tap).clamp(max=width - 2)
            outside = torch.where(active, history[sequence, row], 0.0)
        value = torch.where(inside, padded[tap : tap + tokens], outside)
        output = output + value * taps[:, tap]
    return output.unsqueeze(0)


def causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    activation: str | None = None,
    cu_seqlens: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    return_final_state: bool = False,
    kernel_options: ShortConvKernelOptions | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Apply causal depthwise convolution with an optional activation.

    Same contract as :func:`attn_gym.linear.short_conv.cute.causal_conv1d`: ``x`` is
    ``[B, T, C]`` with contiguous channels and any batch and token strides, ``weight`` is
    ``[C, W]``, ``cu_seqlens`` optionally packs independent sequences into ``[1, T, C]`` up to
    the active endpoint ``cu_seqlens[-1]``, and ``initial_state`` ``[N, W - 1, C]`` replaces
    the zero history. Gradients flow to ``x``, ``weight``, and ``initial_state``.

    Returns:
        A contiguous tensor shaped like ``x``; for packed input only positions before the
        active endpoint are defined. When ``return_final_state`` is true, also returns the
        ``[N, W - 1, C]`` final history.
    """
    options = validate_kernel_options(kernel_options, KERNEL_OPTION_SPEC, op_name="causal_conv1d")
    epilogue = _resolve_activation(activation)
    _validate_inputs(x, weight, cu_seqlens, initial_state)
    for name in ("forward_config", "input_grad_config", "weight_grad_config"):
        if options[name] is not None:
            _validate_config(options[name], x.shape[2], name)
    if options["persistent_tma_input_gradient"] and cu_seqlens is None:
        raise ValueError("persistent TMA input gradients require packed cu_seqlens")
    width = weight.shape[1]
    history = None if width == 1 else initial_state
    # The Triton grid puts dense rows on its z axis, which holds at most 65535 blocks.
    if _triton_kernels is not None and x.is_cuda and x.shape[0] <= 65535:
        output = _triton_kernels.short_conv_forward(x, weight, cu_seqlens, history, activation)
    else:
        output = epilogue(_preactivation(x, weight, cu_seqlens, history)).to(x.dtype)
    if initial_state is not None and width == 1:
        output = output + initial_state.sum()
    if not return_final_state:
        return output
    return output, final_conv_state(x, initial_state, width - 1, cu_seqlens)


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

    Same contract as :func:`attn_gym.linear.short_conv.cute.paged_causal_conv1d`. Positive
    ``state_indices`` select distinct ``state_cache`` slots, which are read (unless
    ``has_initial_state`` is false) and then overwritten with each sequence's final history;
    empty fresh sequences clear their slot and empty resumed sequences keep it. Non-positive
    indices produce zero output and leave the pool untouched. Inference-only.
    """
    epilogue = _resolve_activation(activation)
    _validate_paged_inputs(x, weight, state_cache, state_indices, has_initial_state, cu_seqlens)
    _reject_autograd("paged_causal_conv1d", x, weight, state_cache)
    tokens, width = x.shape[1], weight.shape[1]
    live = state_indices > 0
    # Null routes read slot zero and write back what they read: the pool keeps its bits.
    slots = torch.where(live, state_indices, 0).long()
    history = state_cache[slots]
    resumed = live if has_initial_state is None else live & has_initial_state
    initial_state = torch.where(resumed[:, None, None], history, 0)
    positions = None if cu_seqlens is None else packed_positions(cu_seqlens, tokens)
    # The Triton grid puts dense rows on its z axis, which holds at most 65535 blocks.
    if _triton_kernels is not None and x.is_cuda and x.shape[0] <= 65535:
        output = _triton_kernels.causal_conv1d_forward(
            x, weight, initial_state, live, positions, activation
        )
    else:
        owned = live[:, None, None] if positions is None else live[positions[0]][None, :, None]
        output = epilogue(_preactivation(x, weight, cu_seqlens, initial_state, positions))
        output = torch.where(owned, output, 0.0).to(x.dtype)
    if cu_seqlens is None and tokens >= width - 1:
        # Copy the newest inputs: final_conv_state's autograd zero would turn -0.0 into +0.0.
        final_state = x[:, tokens - width + 1 :]
    else:
        final_state = final_conv_state(x, initial_state, width - 1, cu_seqlens)
    state_cache[slots] = torch.where(live[:, None, None], final_state, history)
    return output


def causal_conv1d_decode(
    x: torch.Tensor,
    weight: torch.Tensor,
    state: torch.Tensor,
    *,
    activation: str | None = None,
    state_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    kernel_options: ShortConvDecodeKernelOptions | None = None,
) -> torch.Tensor:
    """Advance the convolution by one token per sequence over a paged history.

    Same contract as :func:`attn_gym.linear.short_conv.cute.causal_conv1d_decode`. ``state``
    ``[num_slots, W - 1, C]`` is advanced in place: the selected slot's rows shift left and
    ``x`` becomes the newest. Without ``state_indices`` sequence ``i`` uses slot ``i``; with
    them, non-positive entries produce zero output and leave the pool untouched. False
    ``has_initial_state`` entries start from zeros without using the slot's old values.
    Inference-only.
    """
    options = validate_kernel_options(
        kernel_options, DECODE_KERNEL_OPTION_SPEC, op_name="causal_conv1d_decode"
    )
    epilogue = _resolve_activation(activation)
    _validate_decode_inputs(x, weight, state, state_indices, has_initial_state)
    _reject_autograd("causal_conv1d_decode", x, weight, state)
    if options["forward_config"] is not None:
        _validate_config(options["forward_config"], x.shape[1], "forward_config")
    if _triton_kernels is not None and x.is_cuda:
        return _triton_kernels.causal_conv1d_decode(
            x, weight, state, state_indices, has_initial_state, activation
        )
    width = weight.shape[1]
    if state_indices is None:
        live = slots = None
        history = state
        resumed = has_initial_state
    else:
        live = state_indices > 0
        slots = torch.where(live, state_indices, 0).long()
        history = state[slots]
        resumed = live if has_initial_state is None else live & has_initial_state
    initial_state = history if resumed is None else torch.where(resumed[:, None, None], history, 0)
    window = torch.cat((initial_state, x[:, None]), dim=1)
    taps = weight.float()
    output = window[:, width - 1].float() * taps[:, width - 1]
    for tap in range(width - 2, -1, -1):
        output = output + window[:, tap].float() * taps[:, tap]
    output = epilogue(output)
    if live is None:
        state.copy_(window[:, 1:])
        return output.to(x.dtype)
    state[slots] = torch.where(live[:, None, None], window[:, 1:], history)
    return torch.where(live[:, None], output, 0.0).to(x.dtype)


__all__ = [
    "ShortConvConfig",
    "ShortConvDecodeKernelOptions",
    "ShortConvKernelOptions",
    "ShortConvTunedConfig",
    "causal_conv1d",
    "causal_conv1d_decode",
    "paged_causal_conv1d",
]
