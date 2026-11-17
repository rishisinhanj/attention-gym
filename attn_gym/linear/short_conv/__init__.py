# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Optimized short-convolution operations shared by the linear-attention models.

The convolutions resolve to the CuTeDSL backend when it imports and otherwise to the portable
PyTorch backend (``attn_gym.linear.short_conv.portable``), which has the same contract.
Tuning and custom activations are CuTeDSL-only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from attn_gym.linear._lazy import lazy_exports

# Registers the torch-only operator schemas; the CuTeDSL backend stays unimported
# until an exported name is first resolved.
from attn_gym.linear.short_conv import ops as _ops

if TYPE_CHECKING:
    from attn_gym.linear.short_conv.activations import register_activation
    from attn_gym.linear.short_conv.common import (
        ShortConvConfig,
        ShortConvDecodeKernelOptions,
        ShortConvKernelOptions,
        ShortConvTunedConfig,
    )
    from attn_gym.linear.short_conv.cute import (
        causal_conv1d,
        causal_conv1d_decode,
        paged_causal_conv1d,
        tune_causal_conv1d,
    )

_CONVOLUTION_BACKENDS = (
    "attn_gym.linear.short_conv.cute",
    "attn_gym.linear.short_conv.portable",
)

_BACKEND_EXPORTS = {
    "ShortConvConfig": "attn_gym.linear.short_conv.common",
    "ShortConvDecodeKernelOptions": "attn_gym.linear.short_conv.common",
    "ShortConvKernelOptions": "attn_gym.linear.short_conv.common",
    "ShortConvTunedConfig": "attn_gym.linear.short_conv.common",
    "causal_conv1d": _CONVOLUTION_BACKENDS,
    "causal_conv1d_decode": _CONVOLUTION_BACKENDS,
    "paged_causal_conv1d": _CONVOLUTION_BACKENDS,
    "register_activation": "attn_gym.linear.short_conv.activations",
    "tune_causal_conv1d": "attn_gym.linear.short_conv.cute",
}

__all__ = sorted(_BACKEND_EXPORTS)  # noqa: PLE0605 -- backend exports resolve lazily


__getattr__ = lazy_exports(__name__, _BACKEND_EXPORTS, requirement="CuTeDSL backend")
