"""Regression tests for evidence captured by the real DeepSeek-V4 fixture path."""

from __future__ import annotations

import torch
from torch import nn

from fixtures.validate import validate_gate, validate_mhc_maps
from routeaudit.model import gate_math
from routeaudit.model.archspec import ArchSpec
from routeaudit.model.gate_math import GateSpec
from routeaudit_deepseek_v4.hooks import DeepSeekV4HookManager as MoEHookManager


class _FakeHyperConnection(nn.Module):
    def forward(self, hidden_streams):
        n = hidden_streams.shape[-2]
        post = torch.ones(*hidden_streams.shape[:-1])
        comb = torch.eye(n).expand(*hidden_streams.shape[:-2], n, n).clone()
        collapsed = hidden_streams.mean(dim=-2)
        return post, comb, collapsed


class _FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_hc = _FakeHyperConnection()
        self.ffn_hc = _FakeHyperConnection()

    def forward(self, hidden_streams):
        self.attn_hc(hidden_streams)
        self.ffn_hc(hidden_streams)
        return hidden_streams


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_FakeLayer()])

    def forward(self, hidden_streams):
        return self.model.layers[0](hidden_streams)


def test_real_hyperconnection_hooks_capture_both_sites():
    model = _FakeModel()
    spec = ArchSpec(base_attr="model", layers_attr="layers", d_model=8)
    hidden = torch.randn(1, 3, 4, 8)

    with MoEHookManager(model, spec) as hooks:
        hooks.capture_mhc_maps()
        model(hidden)

    assert set(hooks.capture.mhc_maps[0]) == {"attn", "ffn"}
    for captured in hooks.capture.mhc_maps[0].values():
        assert captured.hidden_streams.shape == (1, 3, 4, 8)
        assert captured.post.shape == (1, 3, 4)
        assert captured.comb.shape == (1, 3, 4, 4)
        assert captured.collapsed.shape == (1, 3, 8)


def test_official_v4_router_tuple_is_identified_without_confusing_logits():
    model = _FakeModel()
    hooks = MoEHookManager(model, ArchSpec(base_attr="model", layers_attr="layers"))
    logits = torch.randn(5, 8)
    weights = torch.randn(5, 2)
    indices = torch.randint(0, 8, (5, 2))

    got_weights, got_indices = hooks._official_topk_from_output(
        (logits, weights, indices), top_k=2
    )

    assert torch.equal(got_weights, weights)
    assert torch.equal(got_indices, indices)


def _gate_fixture(*, official: bool = True, one_ulp: bool = False):
    torch.manual_seed(7)
    gs = GateSpec(
        scoring_func="sqrtsoftplus",
        top_k=2,
        use_bias=True,
        routed_scaling_factor=1.5,
    )
    rr = gate_math.route(torch.randn(5, 8), torch.randn(8) * 0.1, gs)
    saved_weights = rr.weights.clone()
    if one_ulp:
        saved_weights[0, 0] = torch.nextafter(
            saved_weights[0, 0], torch.tensor(float("inf"))
        )
    gate = {
        "scores": rr.scores,
        "sel_scores": rr.sel_scores,
        "indices": rr.indices,
        "weights": saved_weights,
    }
    if official:
        gate.update(official_indices=rr.indices.clone(), official_weights=saved_weights.clone())
    return {"meta": {"gate_spec": vars(gs)}, "gate": gate}


def test_gate_validation_separates_exact_official_parity_from_portable_replay():
    results = validate_gate(_gate_fixture(one_ulp=True), atol=2e-7)
    assert all(status is True for status, _ in results)
    assert any("same-device official" in line for _, line in results)
    assert any("portable weights" in line for _, line in results)


def test_legacy_gate_fixture_is_partial_not_false_success():
    results = validate_gate(_gate_fixture(official=False, one_ulp=True), atol=2e-7)
    assert any(status is None and "legacy fixture" in line for status, line in results)
    assert not any(status is False for status, _ in results)


def test_real_mhc_map_validation_checks_applied_b_path():
    hidden = torch.randn(1, 3, 4, 8)
    comb = torch.eye(4).expand(1, 3, 4, 4).clone()
    site = {
        "hidden_streams": hidden,
        "post": torch.ones(1, 3, 4),
        "comb": comb,
        "collapsed": hidden.mean(dim=-2),
    }
    fx = {"mhc_maps": {"layer": 0, "sites": {"attn": site, "ffn": site}}}
    results = validate_mhc_maps(fx)
    assert all(status is True for status, _ in results)


def test_missing_real_mhc_maps_is_reported_as_unmeasured():
    assert validate_mhc_maps({})[0][0] is None
