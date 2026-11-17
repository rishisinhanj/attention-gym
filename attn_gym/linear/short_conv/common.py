# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Backend-neutral short-convolution schedule types and final-history helpers.

Every backend of ``attn_gym.linear.short_conv`` accepts the same ``kernel_options`` and
returns the same final histories, so these live outside any one backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import torch

from attn_gym.types import KernelOptionSpec


@dataclass(frozen=True)
class ShortConvConfig:
    """Compile-time launch and register-tiling specialization."""

    threads: int
    channels_per_thread: int
    times_per_block: int


@dataclass(frozen=True)
class ShortConvTunedConfig:
    """Selected specializations for forward and both first-order gradients."""

    forward: ShortConvConfig
    input_gradient: ShortConvConfig
    weight_gradient: ShortConvConfig

    @classmethod
    def default(
        cls,
        dtype: torch.dtype = torch.bfloat16,
        *,
        packed: bool = False,
    ) -> ShortConvTunedConfig:
        """Return measured GB300 defaults for one storage dtype and layout mode."""
        match dtype:
            case torch.float16 | torch.bfloat16:
                forward = ShortConvConfig(128, 4, 16)
                gradients = (
                    (ShortConvConfig(128, 4, 16), ShortConvConfig(128, 2, 32))
                    if packed
                    else (ShortConvConfig(128, 2, 32), ShortConvConfig(128, 2, 64))
                )
            case torch.float32:
                forward = ShortConvConfig(128, 4, 4)
                gradients = (
                    (ShortConvConfig(128, 2, 16), ShortConvConfig(128, 4, 32))
                    if packed
                    else (ShortConvConfig(128, 2, 12), ShortConvConfig(128, 2, 160))
                )
            case _:
                raise ValueError(f"unsupported short-convolution dtype {dtype}")
        return cls(forward, *gradients)

    def kernel_options(self) -> ShortConvKernelOptions:
        """Return these specializations as ``kernel_options`` for :func:`causal_conv1d`."""
        return {
            "forward_config": self.forward,
            "input_grad_config": self.input_gradient,
            "weight_grad_config": self.weight_gradient,
        }


class ShortConvDecodeKernelOptions(TypedDict, total=False):
    """Kernel options of :func:`causal_conv1d_decode`."""

    forward_config: ShortConvConfig
    """Forward schedule specialization. Defaults to a measured schedule for the inputs and device."""


class ShortConvKernelOptions(ShortConvDecodeKernelOptions, total=False):
    """Kernel options of :func:`causal_conv1d`.

    An absent schedule uses the measured default for the dtype and layout. The operation never
    benchmarks on its own, so there is no ``autotune`` key; :func:`tune_causal_conv1d` selects
    schedules explicitly.
    """

    input_grad_config: ShortConvConfig
    """Input-gradient schedule specialization."""

    weight_grad_config: ShortConvConfig
    """Weight-gradient schedule specialization."""

    persistent_tma_input_gradient: bool
    """Use an explicit machine-bounded persistent worker grid for packed TMA input gradients.
    Defaults to ``False`` because it favors overcaptured replays and can regress full-capacity
    backward. Requires ``x``'s base and outer strides to be 16-byte aligned; backward raises
    otherwise."""


DECODE_KERNEL_OPTION_SPEC: KernelOptionSpec = {"forward_config": (ShortConvConfig, None)}
KERNEL_OPTION_SPEC: KernelOptionSpec = {
    **DECODE_KERNEL_OPTION_SPEC,
    "input_grad_config": (ShortConvConfig, None),
    "weight_grad_config": (ShortConvConfig, None),
    "persistent_tma_input_gradient": (bool, False),
}


def dense_final_conv_state(
    x: torch.Tensor,
    initial_state: torch.Tensor | None,
    state_length: int,
) -> torch.Tensor:
    """Return the final dense history for a positive state length."""
    if x.shape[1] >= state_length:
        if initial_state is None:
            return x[:, -state_length:].clone()
        # Preserve the shaped zero gradient of replaced caller history.
        return x[:, -state_length:] + initial_state[:, :0].sum()
    prefix = (
        x.new_zeros(x.shape[0], state_length - x.shape[1], x.shape[2])
        if initial_state is None
        else initial_state[:, x.shape[1] :]
    )
    return torch.cat((prefix, x), dim=1)


def packed_final_conv_state(
    x: torch.Tensor,
    initial_state: torch.Tensor | None,
    state_length: int,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Gather final histories from packed sequences without host reads."""
    starts = cu_seqlens[:-1].to(torch.int64)
    ends = cu_seqlens[1:].to(torch.int64)
    offsets = torch.arange(state_length, device=x.device)
    positions = ends[:, None] - state_length + offsets
    from_input = positions >= starts[:, None]
    input_values = x[0, positions.clamp(0, x.shape[1] - 1)]
    if initial_state is None:
        state_values = torch.zeros_like(input_values)
    else:
        state_positions = (offsets + (ends - starts)[:, None]).clamp(0, state_length - 1)
        state_values = initial_state.gather(
            1,
            state_positions[:, :, None].expand(-1, -1, x.shape[2]),
        )
    return torch.where(from_input[:, :, None], input_values, state_values)


def packed_positions(
    cu_seqlens: torch.Tensor, tokens: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return each physical token's packed sequence, offset in it, and whether it is active.

    Repeated offsets (empty sequences) own no token. Tokens at or past the active endpoint
    ``cu_seqlens[-1]`` are inactive and assigned to the last sequence.
    """
    positions = torch.arange(tokens, device=cu_seqlens.device, dtype=cu_seqlens.dtype)
    sequence = torch.searchsorted(cu_seqlens[1:], positions, right=True)
    sequence = sequence.clamp_(max=cu_seqlens.shape[0] - 2)
    offset = positions.long() - cu_seqlens[sequence].long()
    return sequence, offset, positions < cu_seqlens[-1]


def final_conv_state(
    x: torch.Tensor,
    initial_state: torch.Tensor | None,
    state_length: int,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return each sequence's final short history without copying the complete input."""
    if state_length == 0:
        num_sequences = x.shape[0] if cu_seqlens is None else cu_seqlens.shape[0] - 1
        final_state = x[:, :0].expand(num_sequences, -1, -1).clone()
        return final_state if initial_state is None else final_state + initial_state
    if cu_seqlens is None:
        return dense_final_conv_state(x, initial_state, state_length)
    return packed_final_conv_state(x, initial_state, state_length, cu_seqlens)


__all__ = [
    "DECODE_KERNEL_OPTION_SPEC",
    "KERNEL_OPTION_SPEC",
    "ShortConvConfig",
    "ShortConvDecodeKernelOptions",
    "ShortConvKernelOptions",
    "ShortConvTunedConfig",
    "dense_final_conv_state",
    "final_conv_state",
    "packed_final_conv_state",
    "packed_positions",
]
