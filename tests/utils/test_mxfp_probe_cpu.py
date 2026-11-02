import json
from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from verl.utils.qat import mxfp_probe, mxfp_quant
from verl.utils.qat.mxfp_probe import (
    begin_quant_error_probe_step,
    finish_quant_error_probe_step,
    quantize_with_probe,
)


def _config(**kwargs):
    defaults = {
        "fake_quant_targets": ("fprop",),
        "quant_backend": "torch",
        "rounding_mode": "rint",
        "probe_quant_error": True,
        "probe_quant_error_use_high_precision": True,
        "layer_name": "model.layers.3.self_attn.q_proj",
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_probe_deduplicates_key_uses_high_precision_and_writes_html(tmp_path, monkeypatch):
    output = tmp_path / "probe.json"
    probe = mxfp_probe.MXFPQuantErrorProbe(output)
    monkeypatch.setattr(mxfp_probe, "_PROBE", probe)
    calls = 0

    def fake_quantize(tensor, fmt, **kwargs):
        nonlocal calls
        calls += 1
        return torch.round(tensor * 2) / 2

    monkeypatch.setattr(mxfp_quant, "fake_quantize", fake_quantize)
    step = begin_quant_error_probe_step(output, 5)
    source = torch.arange(64, dtype=torch.float32).reshape(2, 32) / 7

    first = quantize_with_probe(source, "mxfp8", _config(), target="fprop", operand="activation")
    second = quantize_with_probe(source + 1, "mxfp8", _config(), target="fprop", operand="activation")
    assert torch.equal(first, source)
    assert torch.equal(second, source + 1)
    assert calls == 1

    finish_quant_error_probe_step(step)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert len(payload["records"]) == 1
    record = payload["records"][0]
    assert (record["step"], record["layer"], record["module"]) == (5, 3, "q")
    assert record["target"] == "fprop"
    assert record["operand"] == "activation"
    assert record["count"] == source.numel()
    assert record["mae"] > 0
    html = output.with_suffix(".html").read_text(encoding="utf-8")
    assert "MXFP QDQ Error Probe" in html
    assert "q" in html


def test_probe_low_precision_path_returns_qdq_and_keeps_one_record(tmp_path, monkeypatch):
    output = tmp_path / "probe"
    probe = mxfp_probe.MXFPQuantErrorProbe(output)
    monkeypatch.setattr(mxfp_probe, "_PROBE", probe)
    calls = 0

    def fake_quantize(tensor, fmt, **kwargs):
        nonlocal calls
        calls += 1
        return tensor + 0.25

    monkeypatch.setattr(mxfp_quant, "fake_quantize", fake_quantize)
    step = begin_quant_error_probe_step(output, 6)
    source = torch.ones((1, 32), dtype=torch.float32)
    config = _config(probe_quant_error_use_high_precision=False)

    first = quantize_with_probe(source, "mxfp8", config, target="fprop", operand="weight")
    second = quantize_with_probe(source * 2, "mxfp8", config, target="fprop", operand="weight")
    assert torch.equal(first, source + 0.25)
    assert torch.equal(second, source * 2 + 0.25)
    assert calls == 2  # QDQ remains in the matmul path when explicitly requested.

    finish_quant_error_probe_step(step)
    payload = json.loads((tmp_path / "probe" / "mxfp_quant_error.json").read_text(encoding="utf-8"))
    assert len(payload["records"]) == 1
    assert (tmp_path / "probe" / "mxfp_quant_error.html").exists()


@pytest.mark.parametrize(("high_precision", "expected_quantize_calls"), ((True, 1), (False, 2)))
def test_probe_is_stable_during_checkpoint_recompute(tmp_path, monkeypatch, high_precision, expected_quantize_calls):
    output = tmp_path / "probe"
    probe = mxfp_probe.MXFPQuantErrorProbe(output)
    monkeypatch.setattr(mxfp_probe, "_PROBE", probe)
    quantize_calls = []

    def fake_quantize(tensor, fmt, **kwargs):
        quantize_calls.append((torch.is_grad_enabled(), tensor.requires_grad))
        return torch.round(tensor * 2) / 2

    monkeypatch.setattr(mxfp_quant, "fake_quantize", fake_quantize)
    step = begin_quant_error_probe_step(output, 7)
    source = torch.randn((2, 32), requires_grad=True)

    def run(value):
        quantized = quantize_with_probe(
            value,
            "mxfp8",
            _config(probe_quant_error_use_high_precision=high_precision),
            target="fprop",
            operand="activation",
        )
        return quantized.sin()

    result = checkpoint(run, source, use_reentrant=False)
    assert result.shape == source.shape
    assert result.dtype == source.dtype

    # Simulate checkpoint recomputation after the probe ContextVar is no longer active.
    finish_quant_error_probe_step(step)
    result.sum().backward()

    assert quantize_calls == [(False, False)] * expected_quantize_calls
    qdq_input = source.detach() if high_precision else torch.round(source.detach() * 2) / 2
    torch.testing.assert_close(result.detach(), qdq_input.sin())
    torch.testing.assert_close(source.grad, qdq_input.cos())


def test_qat_without_probe_preserves_fake_quant_grad_context(monkeypatch):
    # Keep a live probe object around to verify the config switch alone keeps
    # the normal QAT path unchanged.
    monkeypatch.setattr(mxfp_probe, "_PROBE", SimpleNamespace(active=True))
    quantize_calls = []

    def fake_quantize(tensor, fmt, **kwargs):
        quantize_calls.append((torch.is_grad_enabled(), tensor.requires_grad))
        return tensor + 0.25

    monkeypatch.setattr(mxfp_quant, "fake_quantize", fake_quantize)
    source = torch.ones((1, 32), requires_grad=True)

    result = quantize_with_probe(
        source,
        "mxfp8",
        _config(probe_quant_error=False),
        target="fprop",
        operand="activation",
    )

    torch.testing.assert_close(result, source + 0.25)
    assert quantize_calls == [(True, True)]
    result.sum().backward()
    torch.testing.assert_close(source.grad, torch.ones_like(source))
