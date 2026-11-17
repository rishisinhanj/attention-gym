# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Lazy loaders for backend-backed exports and operator kernels (see Note: Lazy Imports)."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from types import ModuleType

import torch


def lazy_exports(
    module_name: str, exports: Mapping[str, str | Sequence[str]], *, requirement: str
) -> Callable[[str], object]:
    """Return a module ``__getattr__`` that imports each export's owning module on first use.

    ``exports`` maps a public name to the module that defines it, or to modules in order of
    preference: the first one that imports supplies the name, so a portable implementation can
    stand behind an optional backend. The choice is made once per name. A missing optional
    dependency surfaces as ``ImportError("<name> requires the optional <requirement>: pip install
    ...")``, chained from the first module's error.
    """
    chosen: dict[str, ModuleType] = {}

    def __getattr__(name: str) -> object:
        owners = exports.get(name)
        if owners is None:
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
        module = chosen.get(name)
        if module is None:
            errors: list[ImportError] = []
            for owner in (owners,) if isinstance(owners, str) else owners:
                try:
                    module = importlib.import_module(owner)
                    break
                except ImportError as error:
                    errors.append(error)
            else:
                raise ImportError(
                    f"{name} requires the optional {requirement}: pip install attn-gym[linear]"
                ) from errors[0]
            chosen[name] = module
        return getattr(module, name)

    return __getattr__


def register_lazy_cuda_impls(
    backend: Callable[[], ModuleType], kernels: Mapping[str, str]
) -> None:
    """Register ``attn_gym::<op>`` CUDA kernels that call ``backend().<attr>`` on dispatch.

    ``kernels`` maps an operator name to the backend attribute implementing it, so the optional
    backend is imported when an operator first executes rather than when it is registered.
    Every operator's first argument is a tensor that selects the CUDA device for all launches.
    """

    def kernel(attr: str) -> Callable[..., object]:
        def launch(tensor: torch.Tensor, *args: object) -> object:
            # Enter a device context only when needed: the context manager costs ~1.7 us of host
            # time per call, which the common same-device call should not pay.
            if tensor.device.index == torch.cuda.current_device():
                return getattr(backend(), attr)(tensor, *args)
            with torch.cuda.device(tensor.device):
                return getattr(backend(), attr)(tensor, *args)

        return launch

    for op, attr in kernels.items():
        torch.library.impl(f"attn_gym::{op}", "CUDA", kernel(attr))
