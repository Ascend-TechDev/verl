import pytest
import torch

from verl.utils.qat.mxfp_quant import (
    MXFPMode,
    dequantize_mxfp4,
    dequantize_mxfp8,
    fake_quantize,
    normalize_mxfp_mode,
    quantize_mxfp4,
    quantize_mxfp8,
)
from verl.utils.qat.mxfp_rotation import RotationConfig, _matrix, rotate_blocks


@pytest.mark.parametrize("mode", ["mxfp_w8a8", "w8a8_mxfp8", "mxfp_w4a8", "mxfp_w4a4"])
def test_mode_names_normalize(mode):
    assert normalize_mxfp_mode(mode) in {MXFPMode.W8A8, MXFPMode.W4A8, MXFPMode.W4A4}


@pytest.mark.parametrize("quantize,dequantize", [(quantize_mxfp8, dequantize_mxfp8), (quantize_mxfp4, dequantize_mxfp4)])
def test_mxfp_qdq_preserves_shape_and_is_finite(quantize, dequantize):
    x = torch.linspace(-2, 2, 96, dtype=torch.float32).reshape(3, 32)
    payload, scale = quantize(x)
    y = dequantize(payload, scale, x.dtype)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert torch.count_nonzero(y).item() > 0


def test_mxfp8_zero_block_stays_zero():
    x = torch.zeros((2, 64), dtype=torch.float32)
    payload, scale = quantize_mxfp8(x)
    assert torch.count_nonzero(dequantize_mxfp8(payload, scale, x.dtype)).item() == 0


@pytest.mark.parametrize("fmt", ["mxfp8", "mxfp4"])
def test_fake_quant_pads_and_restores_non_aligned_dimension(fmt):
    x = torch.randn((2, 37), dtype=torch.float32)
    y = fake_quantize(x, fmt)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_rotation_is_seeded_and_orthogonal():
    config = RotationConfig(enable=True, targets=("fprop",), block_size=8, seed=17)
    matrix_a = _matrix(8, 17, torch.device("cpu"), torch.float32)
    matrix_b = _matrix(8, 17, torch.device("cpu"), torch.float32)
    assert torch.equal(matrix_a, matrix_b)
    assert torch.allclose(matrix_a @ matrix_a.T, torch.eye(8), atol=1e-6)
    x = torch.randn((3, 19))
    assert rotate_blocks(x, config).shape == x.shape
