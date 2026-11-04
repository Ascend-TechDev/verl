"""Full activation-residual numerics with real QDQ, rotation, and autograd.

Only the Ascend grouped-matmul hardware boundary is replaced by CPU matmuls.
"""

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from verl.utils.qat.mxfp_linear import MXFPQATConfig, MXFPQATLinear, _target_tensor
from verl.utils.qat.mxfp_moe import _transform_weight, mxfp_grouped_matmul, prepare_mxfp_fprop_input
from verl.utils.qat.mxfp_rotation import rotate_blocks


def _config(**kwargs):
    return MXFPQATConfig(mode="mxfp_w4a8", quant_backend="torch", **kwargs)


def _grouped_matmul(x, weight, group_list, group_list_type, *, split_item=2, group_type=0):
    counts = group_list.tolist()
    if group_list_type == 0:
        counts = [count - (counts[index - 1] if index else 0) for index, count in enumerate(counts)]
    offset = 0
    outputs = []
    for expert, count in enumerate(counts):
        if split_item == 2:
            outputs.append(x[offset : offset + count] @ weight[expert])
        else:
            outputs.append(x[:, offset : offset + count] @ weight[offset : offset + count])
        offset += count
    return torch.cat(outputs, dim=0) if split_item == 2 else torch.stack(outputs)


def test_activation_residual_switches_default_off():
    config = _config()
    assert config.activation_residual_fprop is False
    assert config.activation_residual_bprop is False


@pytest.mark.parametrize("rotation", [False, True])
@pytest.mark.parametrize("grad_path", ["no_grad", "autograd", "custom"])
def test_dense_full_activation_residual_uses_two_gemms_and_preserves_weight_error(monkeypatch, rotation, grad_path):
    torch.manual_seed(51)
    config = _config(
        activation_residual_fprop=True,
        fake_quant_targets=("fprop", "dgrad", "wgrad") if grad_path == "custom" else ("fprop",),
        rotation_enable=rotation,
        rotation_targets=("fprop", "dgrad", "wgrad"),
        rotation_seed=3,
    )
    qat = MXFPQATLinear(32, 64, True, config)
    x = torch.randn(2, 5, 32, requires_grad=grad_path != "no_grad")
    linear_calls = []
    linear = F.linear

    def counted_linear(input, weight, bias=None):
        linear_calls.append((input, weight, bias))
        return linear(input, weight, bias)

    monkeypatch.setattr(F, "linear", counted_linear)
    with torch.set_grad_enabled(grad_path != "no_grad"):
        output = qat(x)
    assert len(linear_calls) == 2
    assert linear_calls[0][1] is linear_calls[1][1]
    assert linear_calls[0][2] is qat.bias
    assert linear_calls[1][2] is None
    values = rotate_blocks(x, config.rotation) if rotation else x
    weight = _target_tensor(qat.weight, config, "fprop", config.weight_format, probe_operand="weight")
    expected = linear(values, weight, qat.bias)
    torch.testing.assert_close(output, expected, rtol=2e-5, atol=2e-6)
    assert not torch.allclose(output, linear(x, qat.weight, qat.bias), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("targets", [("fprop",), ("fprop", "dgrad", "wgrad")])
@pytest.mark.parametrize("rotation", [False, True])
@pytest.mark.parametrize("cached_backward", [False, True])
def test_dense_fprop_switch_preserves_existing_backward_with_fixed_output_gradient(targets, rotation, cached_backward):
    torch.manual_seed(53)
    baseline = MXFPQATLinear(
        32,
        64,
        True,
        _config(
            fake_quant_targets=targets,
            backward_reduction_axis_fake_quant=cached_backward,
            rotation_enable=rotation,
            rotation_targets=("fprop", "dgrad", "wgrad"),
            rotation_seed=9,
        ),
    )
    corrected = MXFPQATLinear(32, 64, True, replace(baseline.qat_config, activation_residual_fprop=True))
    corrected.load_state_dict(baseline.state_dict())
    x = torch.randn(5, 32, requires_grad=True)
    x_corrected = x.detach().clone().requires_grad_()
    grad_output = torch.randn(5, 64)
    baseline(x).backward(grad_output)
    corrected(x_corrected).backward(grad_output)
    torch.testing.assert_close(x.grad, x_corrected.grad, rtol=0, atol=0)
    torch.testing.assert_close(baseline.weight.grad, corrected.weight.grad, rtol=0, atol=0)
    torch.testing.assert_close(baseline.bias.grad, corrected.bias.grad, rtol=0, atol=0)


def test_dense_disabled_switch_is_original_one_gemm_and_parameter_identity(monkeypatch):
    linear = torch.nn.Linear(32, 64)
    qat = MXFPQATLinear.from_linear(linear, _config())
    assert qat.weight is linear.weight
    assert qat.bias is linear.bias
    x = torch.randn(3, 32)
    weight = _target_tensor(qat.weight, qat.qat_config, "fprop", "mxfp4", probe_operand="weight")
    values = _target_tensor(x, qat.qat_config, "fprop", "mxfp8")
    expected = F.linear(values, weight, qat.bias)
    calls = []
    original = F.linear

    def counted_linear(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(F, "linear", counted_linear)
    assert torch.equal(qat(x), expected)
    assert len(calls) == 1


@pytest.mark.parametrize("bias", [False, True])
def test_dense_frozen_input_and_weight_branch_keeps_bias_gradient(bias):
    torch.manual_seed(57)
    config = _config(activation_residual_fprop=True)
    qat = MXFPQATLinear(32, 64, bias, config)
    qat.weight.requires_grad_(False)
    x = torch.randn(3, 32)
    output = qat(x)
    weight_q = _target_tensor(qat.weight, config, "fprop", config.weight_format, probe_operand="weight")
    torch.testing.assert_close(output, F.linear(x, weight_q, qat.bias), rtol=2e-5, atol=2e-6)
    if bias:
        output.sum().backward()
        torch.testing.assert_close(qat.bias.grad, torch.full_like(qat.bias, 3))


def test_dense_fake_quant_disabled_bypasses_residual(monkeypatch):
    config = _config(activation_residual_fprop=True)
    qat = MXFPQATLinear(32, 64, True, config)
    qat.fake_quant_enabled = False
    x = torch.randn(3, 32)
    calls = []
    original = F.linear

    def counted_linear(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(F, "linear", counted_linear)
    output = qat(x)
    assert len(calls) == 1
    torch.testing.assert_close(output, original(x, qat.weight, qat.bias), rtol=0, atol=0)


@pytest.mark.parametrize("rotation", [False, True])
@pytest.mark.parametrize("shared_input", [False, True])
@pytest.mark.parametrize("group_list_type", [0, 1])
def test_moe_fprop_full_residual_uses_same_weight_groups_and_empty_experts(
    monkeypatch, rotation, shared_input, group_list_type
):
    torch.manual_seed(59)
    calls = []

    def counted_gmm(*args, **kwargs):
        calls.append((args, kwargs))
        return _grouped_matmul(*args, **kwargs)

    monkeypatch.setattr("verl.utils.qat.mxfp_moe._call_grouped_matmul", counted_gmm)
    config = _config(activation_residual_fprop=True, rotation_enable=rotation, rotation_seed=5)
    x = torch.randn(7, 32, requires_grad=True)
    weight = torch.randn(3, 32, 64, requires_grad=True)
    groups = torch.tensor([3, 3, 7] if group_list_type == 0 else [3, 0, 4])
    shared = prepare_mxfp_fprop_input(x, config) if shared_input else None
    weight_q = _transform_weight(weight, config, "fprop", config.weight_format, reduction_dim=1)
    output = mxfp_grouped_matmul(
        x,
        weight,
        groups,
        config,
        name="gate_proj",
        group_list_type=group_list_type,
        fprop_x=shared,
        fprop_weight=weight_q,
    )
    assert len(calls) == 2
    assert calls[0][0][1] is calls[1][0][1] is weight_q
    assert calls[0][0][2] is calls[1][0][2] is groups
    if shared_input:
        assert calls[0][0][0] is shared
    values = rotate_blocks(x, config.rotation) if rotation else x
    expected = _grouped_matmul(values, weight_q, groups, group_list_type)
    torch.testing.assert_close(output, expected, rtol=2e-5, atol=3e-6)


@pytest.mark.parametrize("targets", [("fprop",), ("fprop", "dgrad", "wgrad")])
@pytest.mark.parametrize("rotation", [False, True])
def test_moe_fprop_switch_preserves_existing_backward_with_fixed_output_gradient(monkeypatch, targets, rotation):
    monkeypatch.setattr("verl.utils.qat.mxfp_moe._call_grouped_matmul", _grouped_matmul)
    torch.manual_seed(61)
    config = _config(
        fake_quant_targets=targets,
        rotation_enable=rotation,
        rotation_targets=("fprop", "dgrad", "wgrad"),
        rotation_seed=11,
    )
    x = torch.randn(7, 32, requires_grad=True)
    weight = torch.randn(3, 32, 64, requires_grad=True)
    x_corrected = x.detach().clone().requires_grad_()
    weight_corrected = weight.detach().clone().requires_grad_()
    groups = torch.tensor([3, 0, 4])
    grad_output = torch.randn(7, 64)
    mxfp_grouped_matmul(x, weight, groups, config, name="up_proj").backward(grad_output)
    mxfp_grouped_matmul(
        x_corrected,
        weight_corrected,
        groups,
        replace(config, activation_residual_fprop=True),
        name="up_proj",
        fprop_x=prepare_mxfp_fprop_input(x_corrected, config),
    ).backward(grad_output)
    torch.testing.assert_close(x.grad, x_corrected.grad, rtol=0, atol=0)
    torch.testing.assert_close(weight.grad, weight_corrected.grad, rtol=0, atol=0)


def test_zero_activation_residual_still_runs_second_gmm(monkeypatch):
    calls = []

    def counted_gmm(*args, **kwargs):
        calls.append(args)
        return _grouped_matmul(*args, **kwargs)

    monkeypatch.setattr("verl.utils.qat.mxfp_moe._call_grouped_matmul", counted_gmm)
    x = torch.randn(4, 32)
    weight = torch.randn(1, 32, 64)
    config = _config(fake_quant_targets=(), activation_residual_fprop=True)
    result = mxfp_grouped_matmul(x, weight, torch.tensor([4]), config, name="zero-residual")
    assert len(calls) == 2
    assert torch.count_nonzero(calls[1][0]) == 0
    torch.testing.assert_close(result, x @ weight[0])
