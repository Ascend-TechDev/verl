import torch
from torch import nn

from verl.utils.qat.mxfp_linear import MXFPQATConfig
from verl.utils.qat.mxfp_moe import (
    _transform_tokens,
    _transform_weight,
    attach_mxfp_expert_qat,
    get_mxfp_expert_qat_config,
    prepare_mxfp_fprop_operands,
)


class _FakeMoeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(32, 4, bias=False)
        self.experts = nn.Module()


def _config():
    return MXFPQATConfig(
        mode="mxfp_w8a8",
        quant_backend="torch",
        fake_quant_targets=("fprop", "dgrad", "wgrad"),
        backward_reduction_axis_fake_quant=True,
        rotation_enable=True,
        rotation_targets=("fprop", "dgrad", "wgrad"),
        rotation_block_size=32,
        rotation_seed=7,
    )


def test_attach_only_marks_moe_blocks():
    model = nn.Sequential(_FakeMoeBlock(), nn.Linear(32, 32))
    config = _config()
    assert attach_mxfp_expert_qat(model, config) == 1
    assert get_mxfp_expert_qat_config(model[0]) is config
    assert get_mxfp_expert_qat_config(model[1]) is None


def test_grouped_weight_transforms_use_reduction_axis():
    config = _config()
    weight = torch.randn(4, 32, 64)
    transformed = _transform_weight(weight, config, "fprop", config.weight_format, reduction_dim=1)
    assert transformed.shape == weight.shape
    dgrad_weight = _transform_weight(weight, config, "dgrad", config.weight_format, reduction_dim=2)
    assert dgrad_weight.shape == weight.shape
    assert torch.isfinite(transformed).all()
    assert torch.isfinite(dgrad_weight).all()


def test_wgrad_transform_quantizes_token_reduction_axis():
    config = _config()
    x = torch.randn(65, 32)
    transformed = _transform_tokens(x, config, "wgrad", config.activation_format)
    assert transformed.shape == x.shape
    assert torch.isfinite(transformed).all()


def test_gate_up_can_share_detached_fprop_input():
    config = _config()
    x = torch.randn(65, 32, requires_grad=True)
    weight = torch.randn(4, 32, 64, requires_grad=True)
    x_fprop, weight_fprop = prepare_mxfp_fprop_operands(x, weight, config)
    assert x_fprop.shape == x.shape
    assert weight_fprop.shape == weight.shape
    assert not x_fprop.requires_grad
    assert not weight_fprop.requires_grad
