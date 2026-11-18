"""Correctness tests for the PyTorch short convolution.

The PyTorch backend serves ``attn_gym.linear.short_conv`` wherever the CuTeDSL backend does not
import, so these tests import it directly and run on any device. References convolve every
sequence independently in FP64 from its explicit history.
"""

import importlib
from itertools import pairwise

import pytest
import torch
import torch.nn.functional as F

from attn_gym.linear import short_conv
from attn_gym.linear._lazy import lazy_exports
from attn_gym.linear.short_conv import portable
from attn_gym.linear.short_conv.portable import (
    causal_conv1d,
    causal_conv1d_decode,
    paged_causal_conv1d,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA graphs")

# Norm-wise relative error budgets against FP64: one output rounding plus FP32 accumulation.
RELATIVE_BUDGET = {torch.float16: 2e-3, torch.bfloat16: 1e-2, torch.float32: 1e-5}


def _assert_close(actual, expected, dtype):
    assert torch.isfinite(actual).all()
    actual, expected = actual.double().cpu(), expected.double().cpu()
    error = ((actual - expected).norm() / expected.norm().clamp_min(1e-30)).item()
    assert error <= RELATIVE_BUDGET[dtype], error


def _convolve(extended, weight, activation):
    """Convolve one ``[L + W - 1, C]`` sequence that starts with its ``W - 1`` history."""
    output = F.conv1d(extended.t()[None], weight[:, None], groups=extended.shape[1])[0].t()
    return F.silu(output) if activation == "silu" else output


def _reference(x, weight, cu_seqlens, initial_state, activation="silu"):
    """FP64 per-sequence outputs and final histories for dense or packed input."""
    x, weight = x.double(), weight.double()
    width = weight.shape[1]
    if cu_seqlens is None:
        spans = [(batch, 0, x.shape[1]) for batch in range(x.shape[0])]
    else:
        spans = [(0, start, end) for start, end in pairwise(cu_seqlens.tolist())]
    output = torch.zeros_like(x)
    final_states = []
    for sequence, (batch, start, end) in enumerate(spans):
        if initial_state is None:
            history = x.new_zeros(width - 1, x.shape[2])
        else:
            history = initial_state[sequence].double()
        extended = torch.cat((history, x[batch, start:end]))
        if end > start:
            output[batch, start:end] = _convolve(extended, weight, activation)
        final_states.append(extended[extended.shape[0] - (width - 1) :])
    return output, torch.stack(final_states)


def test_operations_resolve_to_the_cutedsl_backend_or_this_one():
    try:
        expected = importlib.import_module("attn_gym.linear.short_conv.cute")
    except ImportError:
        expected = portable
    for name in ("causal_conv1d", "causal_conv1d_decode", "paged_causal_conv1d"):
        assert getattr(short_conv, name) is getattr(expected, name)


def test_lazy_exports_prefers_the_first_importable_owner():
    resolve = lazy_exports(
        "fake",
        {"sqrt": ("cmath", "math"), "pi": ("attn_gym_missing_backend", "math")},
        requirement="test backend",
    )
    assert resolve("sqrt") is importlib.import_module("cmath").sqrt
    assert resolve("pi") is importlib.import_module("math").pi
    missing = lazy_exports(
        "fake", {"name": ("attn_gym_missing_a", "attn_gym_missing_b")}, requirement="test backend"
    )
    with pytest.raises(ImportError, match="name requires the optional test backend") as error:
        missing("name")
    assert error.value.__cause__.name == "attn_gym_missing_a"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("width", [1, 4])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("with_history", [False, True])
def test_forward_backward_and_final_state_match_reference(with_history, packed, width, dtype):
    """Packed input holds empty and length-1 documents and a NaN suffix past the endpoint."""
    torch.manual_seed(0)
    channels = 12
    if packed:
        boundaries = [0, 1, 9, 9, 30, 31, 45]
        cu_seqlens = torch.tensor(boundaries, device=DEVICE, dtype=torch.int32)
        x = torch.randn(1, 50, channels, device=DEVICE, dtype=dtype)
        x[:, boundaries[-1] :] = float("nan")
        sequences, active = len(boundaries) - 1, boundaries[-1]
    else:
        cu_seqlens, sequences, active = None, 3, 2
        x = torch.randn(3, 2, channels, device=DEVICE, dtype=dtype)
    weight = torch.randn(channels, width, device=DEVICE, dtype=dtype)
    history = torch.randn(sequences, width - 1, channels, device=DEVICE, dtype=dtype)
    inputs = (x, weight, history) if with_history else (x, weight)
    for tensor in inputs:
        tensor.requires_grad_()
    output, final_state = causal_conv1d(
        x,
        weight,
        activation="silu",
        cu_seqlens=cu_seqlens,
        initial_state=history if with_history else None,
        return_final_state=True,
    )
    assert output.shape == x.shape and output.dtype == dtype and output.is_contiguous()
    references = [tensor.detach().double().requires_grad_() for tensor in inputs]
    expected, expected_final = _reference(
        *references[:2], cu_seqlens, references[2] if with_history else None
    )
    _assert_close(output[:, :active], expected[:, :active], dtype)
    torch.testing.assert_close(final_state.double(), expected_final, rtol=0, atol=0)

    # Gradients see only active positions, even from a non-zero cotangent past the endpoint.
    grad_output, grad_final = torch.randn_like(output), torch.randn_like(final_state)
    grads = torch.autograd.grad(
        (output, final_state), inputs, (grad_output, grad_final), allow_unused=True
    )
    expected_grads = torch.autograd.grad(
        (expected, expected_final),
        references,
        (grad_output.double(), grad_final.double()),
        allow_unused=True,
    )
    for index, (grad, expected_grad) in enumerate(zip(grads, expected_grads, strict=True)):
        if expected_grad is None or expected_grad.numel() == 0:
            continue
        assert grad.dtype == dtype
        if index == 0:
            grad, expected_grad = grad[:, :active], expected_grad[:, :active]
        _assert_close(grad, expected_grad, dtype)


def _paged_reference(x, weight, state, state_indices, has_initial_state, cu_seqlens):
    """Gather each live slot's history, convolve in FP64, and scatter final histories."""
    slots = state_indices.tolist()
    resumed = [True] * len(slots) if has_initial_state is None else has_initial_state.tolist()
    width = weight.shape[1]
    initial_state = torch.stack(
        [
            state[slot].double()
            if slot > 0 and resumed[n]
            else state.new_zeros(width - 1, state.shape[2]).double()
            for n, slot in enumerate(slots)
        ]
    )
    output, final_state = _reference(x, weight, cu_seqlens, initial_state)
    expected_state = state.clone()
    if cu_seqlens is None:
        spans = [(n, 0, x.shape[1]) for n in range(x.shape[0])]
    else:
        spans = [(0, start, end) for start, end in pairwise(cu_seqlens.tolist())]
    for n, (slot, (batch, start, end)) in enumerate(zip(slots, spans, strict=True)):
        if slot > 0:
            expected_state[slot] = final_state[n].to(state.dtype)
        else:
            output[batch, start:end] = 0
    return output, expected_state


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("packed", [False, True])
def test_paged_prefill_fresh_resumed_empty_and_null_sequences(packed, dtype):
    """Fresh slots ignore NaN history, null routes write nothing, page padding stays intact.

    An empty fresh sequence clears its slot, and an empty resumed one preserves it.
    """
    torch.manual_seed(1)
    channels, width = 12, 4
    if packed:
        boundaries = (0, 5, 6, 6, 21, 23, 23, 40)
        cu_seqlens = torch.tensor(boundaries, device=DEVICE, dtype=torch.int32)
        x = torch.randn(1, 44, channels, device=DEVICE, dtype=dtype)
        x[:, boundaries[-1] :] = float("nan")
        routes = (4, 2, 6, 0, 5, 7, -1)
        resumed = (False, True, False, True, True, True, False)
    else:
        cu_seqlens = None
        x = torch.randn(4, 2, channels, device=DEVICE, dtype=dtype)
        routes, resumed = (4, 0, 5, 2), (False, True, True, True)
    state_indices = torch.tensor(routes, device=DEVICE, dtype=torch.int32)
    has_initial_state = torch.tensor(resumed, device=DEVICE)
    weight = torch.randn(channels, width, device=DEVICE, dtype=dtype)
    elements = (width - 1) * channels
    storage = torch.randn(9, elements + 5, device=DEVICE, dtype=dtype)
    state = storage[:, :elements].view(9, width - 1, channels)
    state[0] = float("nan")
    state[4] = float("nan")
    state[6] = float("nan")
    expected_output, expected_state = _paged_reference(
        x.cpu(),
        weight.cpu(),
        state.cpu(),
        state_indices.cpu(),
        has_initial_state.cpu(),
        None if cu_seqlens is None else cu_seqlens.cpu(),
    )
    expected_storage = storage.cpu().clone()
    expected_storage[:, :elements].view_as(expected_state).copy_(expected_state)
    with torch.no_grad():
        output = paged_causal_conv1d(
            x,
            weight,
            state,
            state_indices,
            activation="silu",
            cu_seqlens=cu_seqlens,
            has_initial_state=has_initial_state,
        )
    active = x.shape[1] if cu_seqlens is None else boundaries[-1]
    assert output.shape == x.shape and output.dtype == dtype and output.is_contiguous()
    _assert_close(output[:, :active], expected_output[:, :active], dtype)
    null = expected_output[:, :active] == 0
    assert (output[:, :active].cpu()[null] == 0).all()
    torch.testing.assert_close(storage.cpu(), expected_storage, rtol=0, atol=0, equal_nan=True)


def _decode_reference(x, weight, state, state_indices, has_initial_state, activation):
    slots = list(range(x.shape[0])) if state_indices is None else state_indices.tolist()
    resumed = [True] * x.shape[0] if has_initial_state is None else has_initial_state.tolist()
    output = torch.zeros(x.shape, dtype=torch.float64)
    expected_state = state.clone()
    for n, slot in enumerate(slots):
        if state_indices is not None and slot <= 0:
            continue
        history = state[slot] if resumed[n] else torch.zeros_like(state[slot])
        window = torch.cat((history, x[n : n + 1]))
        output[n] = _convolve(window.double(), weight.double(), activation)[-1]
        expected_state[slot] = window[1:]
    return output, expected_state


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("activation", [None, "silu"])
@pytest.mark.parametrize("width", [2, 4])
def test_decode_null_fresh_and_implicit_rows(width, activation, dtype):
    torch.manual_seed(2)
    channels = 20
    x = torch.randn(6, channels, device=DEVICE, dtype=dtype)
    x[3:5] = float("nan")
    weight = torch.randn(channels, width, device=DEVICE, dtype=dtype)
    state_indices = torch.tensor((4, 2, 5, 0, -1, 1), device=DEVICE, dtype=torch.int32)
    has_initial_state = torch.tensor((False, True, True, False, True, True), device=DEVICE)
    elements = (width - 1) * channels
    storage = torch.randn(7, elements + 3, device=DEVICE, dtype=dtype)
    state = storage[:, :elements].view(7, width - 1, channels)
    state[0] = float("nan")
    state[4] = float("nan")
    for mask in (has_initial_state, None):
        if mask is None:
            state[4] = 3.0
        expected, expected_state = _decode_reference(
            x.cpu(),
            weight.cpu(),
            state.cpu(),
            state_indices.cpu(),
            None if mask is None else mask.cpu(),
            activation,
        )
        expected_storage = storage.cpu().clone()
        expected_storage[:, :elements].view_as(expected_state).copy_(expected_state)
        with torch.no_grad():
            output = causal_conv1d_decode(
                x,
                weight,
                state,
                activation=activation,
                state_indices=state_indices,
                has_initial_state=mask,
            )
        assert output.dtype == dtype and output.is_contiguous()
        assert (output[3:5] == 0).all()
        _assert_close(output, expected, dtype)
        torch.testing.assert_close(storage.cpu(), expected_storage, rtol=0, atol=0, equal_nan=True)

    # Without state_indices, row i reads and advances slot i; slot zero is an ordinary slot.
    x, state = x[:3], state[:3].clone()
    state[0] = float("nan")
    expected, expected_state = _decode_reference(
        x.cpu(), weight.cpu(), state.cpu(), None, has_initial_state[:3].cpu(), activation
    )
    output = causal_conv1d_decode(
        x, weight, state, activation=activation, has_initial_state=has_initial_state[:3]
    )
    _assert_close(output, expected, dtype)
    torch.testing.assert_close(state.cpu(), expected_state, rtol=0, atol=0)


@requires_cuda
def test_decode_cuda_graph_replay_is_bitwise_eager():
    """Capture one decode step, then replay changed rows, routes, freshness, and history."""
    torch.manual_seed(4)
    dtype, channels, width, rows = torch.bfloat16, 64, 4, 8
    weight = torch.randn(channels, width, device="cuda", dtype=dtype)
    x = torch.randn(rows, channels, device="cuda", dtype=dtype)
    state = torch.randn(12, width - 1, channels, device="cuda", dtype=dtype)
    state_indices = torch.tensor((3, 1, 0, 7, 9, -1, 4, 11), device="cuda", dtype=torch.int32)
    mask = torch.ones(rows, device="cuda", dtype=torch.bool)

    def step(pool):
        return causal_conv1d_decode(
            x,
            weight,
            pool,
            activation="silu",
            state_indices=state_indices,
            has_initial_state=mask,
        )

    with torch.no_grad():
        step(state.clone())
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = step(state)
        for replay in range(3):
            x.copy_(torch.randn_like(x))
            state.copy_(torch.randn_like(state))
            state_indices.copy_(torch.randperm(11, device="cuda")[:rows].to(torch.int32) + 1)
            state_indices[replay] = 0
            mask.copy_(torch.rand(rows, device="cuda") < 0.5)
            eager_state = state.clone()
            eager = step(eager_state)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(output, eager, rtol=0, atol=0)
            torch.testing.assert_close(state, eager_state, rtol=0, atol=0)
            assert (output[replay] == 0).all()


@requires_cuda
def test_paged_prefill_cuda_graph_replays_boundaries_and_routing():
    torch.manual_seed(5)
    dtype, channels, width = torch.bfloat16, 12, 4
    x = torch.randn(1, 16, channels, device="cuda", dtype=dtype)
    weight = torch.randn(channels, width, device="cuda", dtype=dtype)
    state = torch.randn(7, width - 1, channels, device="cuda", dtype=dtype)
    state_indices = torch.tensor((4, 2, 0), device="cuda", dtype=torch.int32)
    has_initial_state = torch.tensor((True, False, True), device="cuda")
    cu_seqlens = torch.tensor((0, 3, 8, 16), device="cuda", dtype=torch.int32)

    def run(pool):
        return paged_causal_conv1d(
            x,
            weight,
            pool,
            state_indices,
            activation="silu",
            cu_seqlens=cu_seqlens,
            has_initial_state=has_initial_state,
        )

    with torch.no_grad():
        run(state.clone())
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = run(state)
        x.add_(0.25)
        state.normal_()
        state_indices.copy_(torch.tensor((5, -1, 1), device="cuda", dtype=torch.int32))
        has_initial_state.copy_(torch.tensor((False, True, True), device="cuda"))
        cu_seqlens.copy_(torch.tensor((0, 1, 1, 11), device="cuda", dtype=torch.int32))
        eager_state = state.clone()
        eager = run(eager_state)
        graph.replay()
        torch.cuda.synchronize()
    torch.testing.assert_close(output[:, :11], eager[:, :11], rtol=0, atol=0)
    torch.testing.assert_close(state, eager_state, rtol=0, atol=0)


def test_rejected_inputs():
    x = torch.randn(1, 4, 12, device=DEVICE, requires_grad=True)
    weight = torch.randn(12, 4, device=DEVICE)
    state = torch.randn(4, 3, 12, device=DEVICE)
    state_indices = torch.tensor((2,), device=DEVICE, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="inference-only"):
        paged_causal_conv1d(x, weight, state, state_indices)
    with pytest.raises(RuntimeError, match="inference-only"):
        causal_conv1d_decode(x[:, 0], weight, state[:1])
    x = x.detach()
    with pytest.raises(ValueError, match="W >= 2"):
        paged_causal_conv1d(x, weight[:, :1], state[:, :0], state_indices)
    with pytest.raises(ValueError, match="one slot per sequence"):
        causal_conv1d_decode(x[:, 0], weight, state)
    with pytest.raises(ValueError, match="has_initial_state must have shape"):
        paged_causal_conv1d(
            x,
            weight,
            state,
            state_indices,
            has_initial_state=torch.ones(2, device=DEVICE, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="packed input"):
        causal_conv1d(x.expand(2, -1, -1), weight, cu_seqlens=torch.tensor((0, 4), device=DEVICE))
    with pytest.raises(ValueError, match="initial_state must have shape"):
        causal_conv1d(x, weight, initial_state=torch.zeros(1, 4, 12, device=DEVICE))
    with pytest.raises(ValueError, match="unknown activation"):
        causal_conv1d(x, weight, activation="tanh")
