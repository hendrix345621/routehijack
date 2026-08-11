from __future__ import annotations

import json
from types import SimpleNamespace

import gate_jacobian
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from routeaudit.model.archspec import ArchSpec
from routeaudit.model.gate_math import GateSpec


class V4Gate(nn.Module):
    def __init__(self, d=8, e=12, k=3):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(e, d) * 0.3, requires_grad=False)
        self.register_buffer("e_score_correction_bias", torch.randn(e) * 0.2)
        self.k = k

    def forward(self, h):
        logits = F.linear(h, self.weight)
        scores = F.softplus(logits).sqrt()
        indices = (scores + self.e_score_correction_bias).topk(self.k, -1).indices
        weights = scores.gather(-1, indices)
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20) * 1.5
        return logits, weights, indices


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = V4Gate()
        self.experts = nn.ModuleList([nn.Identity() for _ in range(12)])

    def forward(self, hidden):
        self.gate(hidden)
        return hidden


class TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = TinyBlock()

    def forward(self, hidden):
        return self.mlp(hidden)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed = nn.Embedding(32, 8)
        self.model.layers = nn.ModuleList([TinyLayer(), TinyLayer()])

    def forward(self, input_ids=None, use_cache=False):
        hidden = self.model.embed(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)
        return SimpleNamespace(logits=hidden)


class TinyTokenizer:
    chat_template = None

    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=[2 + ord(char) % 30 for char in text] or [2])


@pytest.fixture
def dm():
    return SimpleNamespace(
        spec=ArchSpec(name="deepseek", router_output="recompute", n_experts=12, d_model=8),
        gate_spec=GateSpec(
            scoring_func="sqrtsoftplus", top_k=3, use_bias=True,
            n_group=1, routed_scaling_factor=1.5,
        ),
    )


def test_rank_one_spectrum_is_detected():
    direction = torch.arange(1, 9, dtype=torch.float32)
    result = gate_jacobian.gradient_spectrum([direction * scale for scale in (1, -2, 4, -8)])
    assert result["rank90"] == 1
    assert result["effective_rank"] == pytest.approx(1.0, abs=1e-6)
    assert result["top1_energy"] == pytest.approx(1.0, abs=1e-6)


def test_orthogonal_spectrum_has_expected_rank():
    gradients = [torch.eye(8)[i] for i in range(4)]
    result = gate_jacobian.gradient_spectrum(gradients)
    assert result["rank90"] == 4
    assert result["effective_rank"] == pytest.approx(4.0, abs=1e-6)
    assert result["top1_energy"] == pytest.approx(0.25, abs=1e-6)


def test_v4_mass_and_margin_gradients_are_finite(dm):
    torch.manual_seed(3)
    gate = V4Gate()
    hidden = torch.randn(8)
    with torch.no_grad():
        _, weights, indices = gate(hidden[None])
        target = [int(indices[0, int(weights[0].argmax())])]

    gradients = gate_jacobian.local_gate_gradients(dm, gate, hidden, target)
    assert set(gradients) == {"mass", "margin"}
    for grad in gradients.values():
        assert grad.shape == hidden.shape
        assert torch.isfinite(grad).all()
        assert grad.norm() > 0


def test_expert_map_accepts_routeaudit_artifact(tmp_path):
    path = tmp_path / "experts.json"
    path.write_text(json.dumps([
        {"layer": 7, "expert": 4, "score": 0.8},
        {"layer": 7, "expert": 2, "score": 0.7},
        {"layer": 7, "expert": 4, "score": 0.6},
    ]), encoding="utf-8")
    assert gate_jacobian.load_expert_map(path) == {7: [2, 4]}


def test_end_to_end_spectrum_excludes_hash_layer():
    torch.manual_seed(5)
    model = TinyModel().eval()
    spec = ArchSpec(
        name="deepseek", router_output="recompute", n_layers=2, n_experts=12,
        top_k=3, d_model=8,
    )
    gs = GateSpec(
        scoring_func="sqrtsoftplus", top_k=3, use_bias=True,
        routed_scaling_factor=1.5, num_hash_layers=1,
    )
    tiny_dm = SimpleNamespace(
        model=model, tok=TinyTokenizer(), spec=spec, gate_spec=gs, learned_layers=[1]
    )
    result = gate_jacobian.gate_jacobian_spectrum(
        tiny_dm, ["alpha", "beta", "gamma"], objectives=("mass",), want_template=False
    )
    layers = result["objectives"]["mass"]["layers"]
    assert set(layers) == {"1"}
    assert layers["1"]["n_gradients"] == 3
    assert result["hash_layers_excluded"] is True
