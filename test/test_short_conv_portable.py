"""Correctness tests for the portable PyTorch short convolution.

The portable backend serves ``attn_gym.linear.short_conv`` wherever the CuTeDSL backend does
not import, so these tests import it directly and run on any device. References convolve every
sequence independently in FP64 from its explicit history.
"""

import importlib
import sys
from itertools import pairwise

import pytest
import torch
import torch.nn.functional as F

from attn_gym.linear import short_conv
from attn_gym.linear._lazy import lazy_exports
from attn_gym.linear.short_conv import portable
from attn_gym.linear.short_conv.common import ShortConvConfig, ShortConvTunedConfig
from attn_gym.linear.short_conv.portable import (
    causal_conv1d,
    causal_conv1d_decode,
    paged_causal_conv1d,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA graphs")

# Norm-wise relative error budgets against FP64: one output rounding plus FP32 accumulation.
RELATIVE_BUDGET = {torch.float16: 2e-3, torch.bfloat16: 1e-2, torch.float32: 1e-5}


@pytest.fixture(params=["torch", "triton"])
def impl(request, monkeypatch):
    """Run every operation through the PyTorch path or the GPU path's Triton kernels."""
    if request.param == "torch":
        monkeypatch.setattr(portable, "_triton_kernels", None)
    elif DEVICE != "cuda" or portable._triton_kernels is None:
        pytest.skip("the Triton kernels need a GPU and Triton")
    return request.param


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual, expected = actual.double(), expected.double()
    return ((actual - expected).norm() / expected.norm().clamp_min(1e-30)).item()


def _assert_close_relative(actual, expected, dtype, *, scale: float = 1.0):
    assert torch.isfinite(actual).all()
    error = _relative_error(actual, expected)
    assert error <= scale * RELATIVE_BUDGET[dtype], error


def _activate(value: torch.Tensor, activation: str | None) -> torch.Tensor:
    return F.silu(value) if activation == "silu" else value


def _convolve(extended: torch.Tensor, weight: torch.Tensor, activation: str | None):
    """Convolve one ``[L + W - 1, C]`` sequence that starts with its ``W - 1`` history."""
    output = F.conv1d(extended.t()[None], weight[:, None], groups=extended.shape[1])[0].t()
    return _activate(output, activation)


def _sequence_reference(x, weight, cu_seqlens, initial_state, activation):
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
        history = (
            x.new_zeros(width - 1, x.shape[2])
            if initial_state is None
            else initial_state[sequence].double()
        )
        extended = torch.cat((history, x[batch, start:end]))
        if end > start:
            output[batch, start:end] = _convolve(extended, weight, activation)
        final_states.append(extended[extended.shape[0] - (width - 1) :])
    return output, torch.stack(final_states)


def _inputs(batch, tokens, channels, width, dtype, *, requires_grad=True):
    x = torch.randn(batch, tokens, channels, device=DEVICE, dtype=dtype)
    weight = torch.randn(channels, width, device=DEVICE, dtype=dtype)
    return x.requires_grad_(requires_grad), weight.requires_grad_(requires_grad)


def test_short_conv_exports_resolve_to_an_importable_backend():
    """The CuTeDSL backend wins whenever it imports; the portable one serves otherwise."""
    try:
        importlib.import_module("attn_gym.linear.short_conv.cute")
    except ImportError:
        expected = portable
    else:
        expected = sys.modules["attn_gym.linear.short_conv.cute"]
    for name in ("causal_conv1d", "causal_conv1d_decode", "paged_causal_conv1d"):
        assert getattr(short_conv, name) is getattr(expected, name)
    assert short_conv.ShortConvConfig is ShortConvConfig
    assert portable.ShortConvTunedConfig is ShortConvTunedConfig


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
    assert isinstance(error.value.__cause__, ModuleNotFoundError)
    assert error.value.__cause__.name == "attn_gym_missing_a"
    with pytest.raises(AttributeError):
        missing("other")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    ("width", "batch", "tokens", "activation"),
    [
        (1, 1, 17, "silu"),
        (2, 2, 5, "silu"),
        (4, 1, 17, "silu"),
        (4, 3, 19, None),
        (5, 2, 3, "silu"),
    ],
)
def test_dense_forward_and_backward_match_reference(width, batch, tokens, activation, dtype, impl):
    torch.manual_seed(0)
    x, weight = _inputs(batch, tokens, 12, width, dtype)
    grad_output = torch.randn_like(x)
    actual = causal_conv1d(x, weight, activation=activation)
    assert actual.dtype == dtype and actual.shape == x.shape and actual.is_contiguous()
    reference_inputs = [tensor.detach().double().requires_grad_() for tensor in (x, weight)]
    expected, _ = _sequence_reference(*reference_inputs, None, None, activation)
    _assert_close_relative(actual, expected, dtype)
    actual_grads = torch.autograd.grad(actual, (x, weight), grad_output)
    expected_grads = torch.autograd.grad(expected, reference_inputs, grad_output.double())
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        assert actual_grad.dtype == dtype
        _assert_close_relative(actual_grad, expected_grad, dtype)


def test_strided_input_matches_contiguous_copy(impl):
    """Column slices of a fused projection are read in place and give identical bits."""
    torch.manual_seed(1)
    base = torch.randn(1, 40, 3 * 16, device=DEVICE, dtype=torch.bfloat16)
    x = base[..., 16:32]
    weight = torch.randn(16, 4, device=DEVICE, dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, 7, 40], device=DEVICE, dtype=torch.int32)
    strided = causal_conv1d(x, weight, activation="silu", cu_seqlens=cu_seqlens)
    contiguous = causal_conv1d(x.contiguous(), weight, activation="silu", cu_seqlens=cu_seqlens)
    torch.testing.assert_close(strided, contiguous, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("width", [1, 4, 5])
def test_packed_documents_match_independent_sequences(width, dtype, impl):
    """Per-document resets, length-1 and empty documents, and a NaN suffix past the endpoint."""
    torch.manual_seed(2)
    boundaries = [0, 1, 64, 64, 129, 130, 131, 133, 433]
    capacity = 450
    cu_seqlens = torch.tensor(boundaries, device=DEVICE, dtype=torch.int32)
    x, weight = _inputs(1, capacity, 24, width, dtype, requires_grad=False)
    with torch.no_grad():
        x[:, boundaries[-1] :] = float("nan")
    x.requires_grad_()
    weight.requires_grad_()
    active = boundaries[-1]
    actual = causal_conv1d(x, weight, activation="silu", cu_seqlens=cu_seqlens)
    reference_inputs = [tensor.detach().double().requires_grad_() for tensor in (x, weight)]
    expected, _ = _sequence_reference(*reference_inputs, cu_seqlens.cpu(), None, "silu")
    _assert_close_relative(actual[:, :active], expected[:, :active], dtype)
    # Gradients see only active positions, even from a non-zero cotangent past the endpoint.
    grad_output = torch.randn_like(x)
    actual_grads = torch.autograd.grad(actual, (x, weight), grad_output)
    masked = grad_output.double().clone()
    masked[:, active:] = 0
    expected_grads = torch.autograd.grad(expected, reference_inputs, masked)
    _assert_close_relative(actual_grads[0][:, :active], expected_grads[0][:, :active], dtype)
    _assert_close_relative(actual_grads[1], expected_grads[1], dtype)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("width", [1, 4])
def test_initial_and_final_state_with_gradients(packed, width, impl):
    torch.manual_seed(3)
    dtype = torch.float32
    if packed:
        cu_seqlens = torch.tensor([0, 2, 2, 9, 30], device=DEVICE, dtype=torch.int32)
        x, weight = _inputs(1, 30, 8, width, dtype)
        sequences = 4
    else:
        cu_seqlens = None
        x, weight = _inputs(3, 2, 8, width, dtype)
        sequences = 3
    initial_state = torch.randn(
        sequences, width - 1, 8, device=DEVICE, dtype=dtype, requires_grad=True
    )
    actual, final_state = causal_conv1d(
        x,
        weight,
        activation="silu",
        cu_seqlens=cu_seqlens,
        initial_state=initial_state,
        return_final_state=True,
    )
    inputs = (x, weight, initial_state)
    reference_inputs = [tensor.detach().double().requires_grad_() for tensor in inputs]
    expected, expected_final = _sequence_reference(
        reference_inputs[0],
        reference_inputs[1],
        None if cu_seqlens is None else cu_seqlens.cpu(),
        reference_inputs[2],
        "silu",
    )
    _assert_close_relative(actual, expected, dtype)
    torch.testing.assert_close(final_state.double(), expected_final, rtol=0, atol=0)
    grad_output, grad_final = torch.randn_like(actual), torch.randn_like(final_state)
    actual_grads = torch.autograd.grad((actual, final_state), inputs, (grad_output, grad_final))
    expected_grads = torch.autograd.grad(
        (expected, expected_final), reference_inputs, (grad_output.double(), grad_final.double())
    )
    for index, (actual_grad, expected_grad) in enumerate(
        zip(actual_grads, expected_grads, strict=True)
    ):
        if expected_grad.numel():
            _assert_close_relative(actual_grad, expected_grad, dtype)
        else:
            assert actual_grad.shape == expected_grad.shape, index


@requires_cuda
@pytest.mark.parametrize("packed", [False, True])
def test_trainer_triton_matches_torch_path_and_opcheck(monkeypatch, packed):
    """The Triton training op agrees with the PyTorch path and is a well-formed custom op."""
    if portable._triton_kernels is None:
        pytest.skip("needs Triton")
    torch.manual_seed(14)
    channels, width = 1000, 4
    if packed:
        cu_seqlens = torch.tensor((0, 1, 1, 77, 300, 301), device="cuda", dtype=torch.int32)
        x = torch.randn(1, 320, channels, device="cuda", dtype=torch.bfloat16)
        x[:, 301:] = float("nan")
        sequences, active = 5, 301
    else:
        cu_seqlens = None
        x = torch.randn(3, 40, channels + 8, device="cuda", dtype=torch.bfloat16)[..., :channels]
        sequences, active = 3, 40
    weight = torch.randn(channels, width, device="cuda", dtype=torch.bfloat16)
    state = torch.randn(sequences, width - 1, channels, device="cuda", dtype=torch.bfloat16)
    grad_output = torch.randn(x.shape, device="cuda", dtype=torch.bfloat16)
    results = []
    for kernels in (portable._triton_kernels, None):
        monkeypatch.setattr(portable, "_triton_kernels", kernels)
        leaves = [tensor.detach().clone().requires_grad_() for tensor in (x, weight, state)]
        output = causal_conv1d(
            leaves[0], leaves[1], activation="silu", cu_seqlens=cu_seqlens, initial_state=leaves[2]
        )
        results.append((output[:, :active], *torch.autograd.grad(output, leaves, grad_output)))
    for name, triton_value, torch_value in zip(
        ("out", "dx", "dweight", "dstate"), *results, strict=True
    ):
        if name == "dx":
            triton_value, torch_value = triton_value[:, :active], torch_value[:, :active]
        assert torch.isfinite(triton_value).all(), name
        assert _relative_error(triton_value, torch_value) < 2e-3, name
    torch.library.opcheck(
        torch.ops.attn_gym._portable_short_conv_fwd.default,
        (
            torch.randn(1, 37, 16, device="cuda", requires_grad=True),
            torch.randn(16, width, device="cuda", requires_grad=True),
            torch.tensor((0, 5, 37), device="cuda", dtype=torch.int32),
            torch.randn(2, width - 1, 16, device="cuda", requires_grad=True),
            "silu",
        ),
    )


def test_kernel_options_are_validated_and_accepted():
    torch.manual_seed(4)
    x, weight = _inputs(1, 9, 12, 4, torch.bfloat16)
    tuned = ShortConvTunedConfig.default(torch.bfloat16).kernel_options()
    torch.testing.assert_close(
        causal_conv1d(x, weight, activation="silu", kernel_options=tuned),
        causal_conv1d(x, weight, activation="silu"),
        rtol=0,
        atol=0,
    )
    with pytest.raises(ValueError, match="unsupported causal_conv1d kernel options"):
        causal_conv1d(x, weight, kernel_options={"autotune": True})
    with pytest.raises(ValueError, match="warp multiple"):
        causal_conv1d(x, weight, kernel_options={"forward_config": ShortConvConfig(48, 1, 1)})
    with pytest.raises(ValueError, match="persistent TMA"):
        causal_conv1d(x, weight, kernel_options={"persistent_tma_input_gradient": True})
    with pytest.raises(ValueError, match="unknown activation"):
        causal_conv1d(x, weight, activation="tanh")


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
    output, final_state = _sequence_reference(x, weight, cu_seqlens, initial_state, "silu")
    expected_state = state.clone()
    spans = (
        [(n, 0, x.shape[1]) for n in range(x.shape[0])]
        if cu_seqlens is None
        else [(0, start, end) for start, end in pairwise(cu_seqlens.tolist())]
    )
    for n, (slot, (batch, start, end)) in enumerate(zip(slots, spans, strict=True)):
        if slot > 0:
            expected_state[slot] = final_state[n].to(state.dtype)
        else:
            output[batch, start:end] = 0
    return output, expected_state


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("packed", [False, True])
def test_paged_prefill_fresh_resumed_and_null_slots(packed, dtype, impl):
    """Fresh slots ignore NaN history, null routes write nothing, page padding stays intact."""
    torch.manual_seed(5)
    channels, width = 12, 4
    if packed:
        boundaries = (0, 5, 6, 6, 21, 23, 40)
        cu_seqlens = torch.tensor(boundaries, device=DEVICE, dtype=torch.int32)
        x = torch.randn(1, 44, channels, device=DEVICE, dtype=dtype)
        x[:, boundaries[-1] :] = float("nan")
        routes, resumed = (4, 2, 6, 0, 5, -1), (False, True, False, True, True, False)
    else:
        cu_seqlens = None
        x = torch.randn(4, 2, channels, device=DEVICE, dtype=dtype)
        routes, resumed = (4, 0, 5, 2), (False, True, True, True)
    state_indices = torch.tensor(routes, device=DEVICE, dtype=torch.int32)
    has_initial_state = torch.tensor(resumed, device=DEVICE)
    weight = torch.randn(channels, width, device=DEVICE, dtype=dtype)
    elements = (width - 1) * channels
    storage = torch.randn(8, elements + 5, device=DEVICE, dtype=dtype)
    state = storage[:, :elements].view(8, width - 1, channels)
    state[0] = float("nan")
    state[4] = float("nan")
    expected_output, expected_state = _paged_reference(
        x.cpu(), weight.cpu(), state.cpu(), state_indices.cpu(), has_initial_state.cpu(),
        None if cu_seqlens is None else cu_seqlens.cpu(),
    )
    expected_storage = storage.cpu().clone()
    expected_storage[:, :elements].view_as(expected_state).copy_(expected_state)

    with torch.no_grad():
        actual = paged_causal_conv1d(
            x,
            weight,
            state,
            state_indices,
            activation="silu",
            cu_seqlens=cu_seqlens,
            has_initial_state=has_initial_state,
        )
    active = x.shape[1] if cu_seqlens is None else boundaries[-1]
    assert actual.shape == x.shape and actual.dtype == dtype and actual.is_contiguous()
    _assert_close_relative(actual[:, :active].cpu(), expected_output[:, :active], dtype)
    null = expected_output[:, :active] == 0
    assert (actual[:, :active].cpu()[null] == 0).all()
    torch.testing.assert_close(storage.cpu(), expected_storage, rtol=0, atol=0, equal_nan=True)


def test_paged_prefill_empty_fresh_and_resumed_slots(impl):
    torch.manual_seed(6)
    x = torch.randn(1, 4, 12, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(12, 4, device=DEVICE, dtype=x.dtype)
    state = torch.randn(6, 3, 12, device=DEVICE, dtype=x.dtype)
    original = state.clone()
    state_indices = torch.tensor((4, 2), device=DEVICE, dtype=torch.int32)
    has_initial_state = torch.tensor((False, True), device=DEVICE)
    cu_seqlens = torch.tensor((0, 0, 0), device=DEVICE, dtype=torch.int32)
    with torch.no_grad():
        paged_causal_conv1d(
            x,
            weight,
            state,
            state_indices,
            activation="silu",
            cu_seqlens=cu_seqlens,
            has_initial_state=has_initial_state,
        )
    torch.testing.assert_close(state[4], torch.zeros_like(state[4]), rtol=0, atol=0)
    torch.testing.assert_close(state[[0, 1, 2, 3, 5]], original[[0, 1, 2, 3, 5]], rtol=0, atol=0)


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
@pytest.mark.parametrize("width", [2, 4, 5])
def test_decode_null_rows_fresh_rows_and_padded_pool(width, activation, dtype, impl):
    torch.manual_seed(7)
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
            x.cpu(), weight.cpu(), state.cpu(), state_indices.cpu(),
            None if mask is None else mask.cpu(), activation,
        )
        expected_storage = storage.cpu().clone()
        expected_storage[:, :elements].view_as(expected_state).copy_(expected_state)
        with torch.no_grad():
            actual = causal_conv1d_decode(
                x,
                weight,
                state,
                activation=activation,
                state_indices=state_indices,
                has_initial_state=mask,
            )
        assert actual.dtype == dtype and actual.is_contiguous()
        assert (actual[3:5] == 0).all()
        _assert_close_relative(actual.cpu(), expected, dtype)
        torch.testing.assert_close(
            storage.cpu(), expected_storage, rtol=0, atol=0, equal_nan=True
        )


def test_decode_implicit_slots_and_strided_rows(impl):
    torch.manual_seed(8)
    base = torch.randn(3, 40, device=DEVICE, dtype=torch.bfloat16)
    x = base[:, 8:24]
    weight = torch.randn(16, 4, device=DEVICE, dtype=x.dtype)
    state = torch.randn(3, 3, 16, device=DEVICE, dtype=x.dtype)
    state[0] = float("nan")
    mask = torch.tensor((False, True, True), device=DEVICE)
    expected, expected_state = _decode_reference(
        x.cpu(), weight.cpu(), state.cpu(), None, mask.cpu(), "silu"
    )
    actual = causal_conv1d_decode(x, weight, state, activation="silu", has_initial_state=mask)
    _assert_close_relative(actual.cpu(), expected, torch.bfloat16)
    torch.testing.assert_close(state.cpu(), expected_state, rtol=0, atol=0)


def test_prefill_continues_in_decode_like_one_long_prefill(impl):
    """Prefill a prefix, decode the remaining tokens, and match a single prefill of all."""
    torch.manual_seed(9)
    dtype, channels, width = torch.float32, 12, 4
    lengths, decode_steps = (1, 5, 2), 4
    total = [length + decode_steps for length in lengths]
    sequences = [torch.randn(n, channels, device=DEVICE, dtype=dtype) for n in total]
    weight = torch.randn(channels, width, device=DEVICE, dtype=dtype)
    state_indices = torch.tensor((3, 1, 6), device=DEVICE, dtype=torch.int32)

    def offsets(counts):
        return torch.tensor(
            [0, *torch.tensor(counts).cumsum(0).tolist()], device=DEVICE, dtype=torch.int32
        )

    with torch.no_grad():
        full_state = torch.randn(7, width - 1, channels, device=DEVICE, dtype=dtype)
        state = full_state.clone()
        expected = paged_causal_conv1d(
            torch.cat(sequences)[None], weight, full_state, state_indices,
            activation="silu", cu_seqlens=offsets(total),
        )
        prefix = paged_causal_conv1d(
            torch.cat([s[:n] for s, n in zip(sequences, lengths, strict=True)])[None],
            weight, state, state_indices, activation="silu", cu_seqlens=offsets(lengths),
        )
        steps = [
            causal_conv1d_decode(
                torch.stack([s[n + step] for s, n in zip(sequences, lengths, strict=True)]),
                weight, state, activation="silu", state_indices=state_indices,
            )
            for step in range(decode_steps)
        ]
    starts = offsets(total)[:-1].tolist()
    prefix_starts = offsets(lengths)[:-1].tolist()
    for n, (start, length) in enumerate(zip(starts, lengths, strict=True)):
        torch.testing.assert_close(
            prefix[0, prefix_starts[n] : prefix_starts[n] + length],
            expected[0, start : start + length],
            rtol=1e-6,
            atol=1e-6,
        )
        for step in range(decode_steps):
            torch.testing.assert_close(
                steps[step][n], expected[0, start + length + step], rtol=1e-6, atol=1e-6
            )
    torch.testing.assert_close(state, full_state, rtol=0, atol=0)


@requires_cuda
def test_decode_triton_matches_torch_path(monkeypatch):
    """Both decode paths advance the pool identically and agree on outputs to FP32 rounding."""
    if portable._triton_kernels is None:
        pytest.skip("needs Triton")
    torch.manual_seed(12)
    x = torch.randn(64, 8192, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(8192, 4, device="cuda", dtype=torch.bfloat16)
    state = torch.randn(80, 3, 8192, device="cuda", dtype=torch.bfloat16)
    state_indices = (torch.randperm(79, device="cuda")[:64] + 1).to(torch.int32)
    state_indices[::7] = 0
    mask = torch.rand(64, device="cuda") < 0.8
    triton_state, torch_state = state.clone(), state.clone()
    with torch.no_grad():
        triton_output = causal_conv1d_decode(
            x, weight, triton_state, activation="silu", state_indices=state_indices,
            has_initial_state=mask,
        )
        monkeypatch.setattr(portable, "_triton_kernels", None)
        torch_output = causal_conv1d_decode(
            x, weight, torch_state, activation="silu", state_indices=state_indices,
            has_initial_state=mask,
        )
    torch.testing.assert_close(triton_state, torch_state, rtol=0, atol=0)
    torch.testing.assert_close(triton_output, torch_output, rtol=1.6e-2, atol=1e-5)
    assert _relative_error(triton_output, torch_output) < 1e-3


@requires_cuda
def test_decode_cuda_graph_replay_is_bitwise_eager(impl):
    """Capture one decode step, then replay changed rows, routes, freshness, and history."""
    torch.manual_seed(10)
    dtype, channels, width, rows = torch.bfloat16, 64, 4, 8
    weight = torch.randn(channels, width, device="cuda", dtype=dtype)
    x = torch.randn(rows, channels, device="cuda", dtype=dtype)
    state = torch.randn(12, width - 1, channels, device="cuda", dtype=dtype)
    state_indices = torch.tensor((3, 1, 0, 7, 9, -1, 4, 11), device="cuda", dtype=torch.int32)
    mask = torch.ones(rows, device="cuda", dtype=torch.bool)

    def step():
        return causal_conv1d_decode(
            x, weight, state, activation="silu", state_indices=state_indices,
            has_initial_state=mask,
        )

    with torch.no_grad():
        step()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = step()
        for replay in range(3):
            x.copy_(torch.randn_like(x))
            state.copy_(torch.randn_like(state))
            state_indices.copy_(torch.randperm(11, device="cuda")[:rows].to(torch.int32) + 1)
            state_indices[replay] = 0
            mask.copy_(torch.rand(rows, device="cuda") < 0.5)
            eager_state = state.clone()
            eager = causal_conv1d_decode(
                x, weight, eager_state, activation="silu", state_indices=state_indices,
                has_initial_state=mask,
            )
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(output, eager, rtol=0, atol=0)
            torch.testing.assert_close(state, eager_state, rtol=0, atol=0)
            assert (output[replay] == 0).all()


@requires_cuda
@pytest.mark.parametrize("packed", [False, True])
def test_paged_prefill_triton_matches_torch_path(monkeypatch, packed):
    if portable._triton_kernels is None:
        pytest.skip("needs Triton")
    torch.manual_seed(13)
    channels, width = 1000, 4
    if packed:
        cu_seqlens = torch.tensor((0, 1, 1, 77, 300, 301), device="cuda", dtype=torch.int32)
        x = torch.randn(1, 320, channels, device="cuda", dtype=torch.bfloat16)
        x[:, 301:] = float("nan")
        routes, active = (3, 0, 9, 4, -1), 301
    else:
        cu_seqlens = None
        x = torch.randn(6, 40, channels + 8, device="cuda", dtype=torch.bfloat16)[..., :channels]
        routes, active = (3, 0, 9, 4, -1, 2), 40
    state_indices = torch.tensor(routes, device="cuda", dtype=torch.int32)
    has_initial_state = torch.tensor([n % 2 == 0 for n in range(len(routes))], device="cuda")
    weight = torch.randn(channels, width, device="cuda", dtype=torch.bfloat16)
    state = torch.randn(10, width - 1, channels, device="cuda", dtype=torch.bfloat16)
    outputs, states = [], []
    for kernels in (portable._triton_kernels, None):
        monkeypatch.setattr(portable, "_triton_kernels", kernels)
        pool = state.clone()
        with torch.no_grad():
            outputs.append(
                paged_causal_conv1d(
                    x, weight, pool, state_indices, activation="silu", cu_seqlens=cu_seqlens,
                    has_initial_state=has_initial_state,
                )[:, :active]
            )
        states.append(pool)
    torch.testing.assert_close(states[0], states[1], rtol=0, atol=0)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=1.6e-2, atol=1e-5)
    assert _relative_error(outputs[0], outputs[1]) < 1e-3


@requires_cuda
def test_paged_prefill_cuda_graph_replays_boundaries_and_routing(impl):
    torch.manual_seed(11)
    dtype, channels, width = torch.bfloat16, 12, 4
    x = torch.randn(1, 16, channels, device="cuda", dtype=dtype)
    weight = torch.randn(channels, width, device="cuda", dtype=dtype)
    state = torch.randn(7, width - 1, channels, device="cuda", dtype=dtype)
    state_indices = torch.tensor((4, 2, 0), device="cuda", dtype=torch.int32)
    has_initial_state = torch.tensor((True, False, True), device="cuda")
    cu_seqlens = torch.tensor((0, 3, 8, 16), device="cuda", dtype=torch.int32)

    def run(pool):
        return paged_causal_conv1d(
            x, weight, pool, state_indices, activation="silu", cu_seqlens=cu_seqlens,
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


@requires_cuda
@pytest.mark.parametrize("packed", [False, True])
def test_fullgraph_compile_matches_eager(impl, packed):
    """Compiled graphs launch the same kernels on the same buffers as eager calls."""
    torch._dynamo.reset()
    torch.manual_seed(15)
    channels, width = 96, 4
    weight = torch.randn(channels, width, device="cuda", dtype=torch.bfloat16)
    if packed:
        cu_seqlens = torch.tensor((0, 3, 3, 30), device="cuda", dtype=torch.int32)
        x = torch.randn(1, 40, channels, device="cuda", dtype=torch.bfloat16)
        active, sequences = 30, 3
    else:
        cu_seqlens = None
        x = torch.randn(3, 9, channels, device="cuda", dtype=torch.bfloat16)
        active, sequences = 9, 3
    slots = torch.tensor((2, 0, 5), device="cuda", dtype=torch.int32)
    fresh = torch.tensor((False, True, True), device="cuda")
    pool = torch.randn(7, width - 1, channels, device="cuda", dtype=torch.bfloat16)
    rows = torch.randn(3, channels, device="cuda", dtype=torch.bfloat16)

    def prefill(pool):
        return paged_causal_conv1d(
            x, weight, pool, slots, activation="silu", cu_seqlens=cu_seqlens,
            has_initial_state=fresh,
        )

    def decode(pool):
        return causal_conv1d_decode(
            rows, weight, pool, activation="silu", state_indices=slots, has_initial_state=fresh
        )

    with torch.no_grad():
        for run in (prefill, decode):
            eager_pool, compiled_pool = pool.clone(), pool.clone()
            eager = run(eager_pool)
            compiled = torch.compile(run, fullgraph=True)(compiled_pool)
            if eager.ndim == 3:
                eager, compiled = eager[:, :active], compiled[:, :active]
            torch.testing.assert_close(compiled, eager, rtol=0, atol=0)
            torch.testing.assert_close(compiled_pool, eager_pool, rtol=0, atol=0)

    state = torch.randn(sequences, width - 1, channels, device="cuda", dtype=torch.bfloat16)

    def train(*leaves):
        return causal_conv1d(
            leaves[0], leaves[1], activation="silu", cu_seqlens=cu_seqlens, initial_state=leaves[2]
        )

    grad_output = torch.randn(x.shape, device="cuda", dtype=torch.bfloat16)
    results = []
    for run in (train, torch.compile(train, fullgraph=True)):
        leaves = [tensor.clone().requires_grad_() for tensor in (x, weight, state)]
        output = run(*leaves)
        grads = torch.autograd.grad(output, leaves, grad_output)
        results.append((output[:, :active], grads[0][:, :active], *grads[1:]))
    for compiled, eager in zip(results[1], results[0], strict=True):
        torch.testing.assert_close(compiled, eager, rtol=0, atol=1e-4)


def test_stateful_operations_reject_autograd_and_malformed_inputs():
    x, weight = _inputs(1, 4, 12, 4, torch.bfloat16)
    state = torch.randn(4, 3, 12, device=DEVICE, dtype=x.dtype)
    state_indices = torch.tensor((2,), device=DEVICE, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="inference-only"):
        paged_causal_conv1d(x, weight, state, state_indices)
    with pytest.raises(RuntimeError, match="inference-only"):
        causal_conv1d_decode(x[:, 0], weight, state[:1])
    x, weight = x.detach(), weight.detach()
    with pytest.raises(ValueError, match="contiguous channels"):
        paged_causal_conv1d(
            x.transpose(1, 2).contiguous().transpose(1, 2), weight, state, state_indices
        )
    with pytest.raises(ValueError, match="W >= 2"):
        paged_causal_conv1d(x, weight[:, :1].contiguous(), state[:, :0], state_indices)
    with pytest.raises(ValueError, match="has_initial_state"):
        paged_causal_conv1d(
            x, weight, state, state_indices,
            has_initial_state=torch.ones(2, device=DEVICE, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="state must have shape"):
        causal_conv1d_decode(x[:, 0], weight, state)
    with pytest.raises(ValueError, match="unknown activation"):
        causal_conv1d_decode(x[:, 0], weight, state[:1], activation="missing")
