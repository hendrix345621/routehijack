"""Device and allocation-sensitive checks for GPU-facing utility paths."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from routeaudit.identify.activation_freq import _expert_membership_counts
from routeaudit.model import gate_math
from routeaudit.model import loader as loader_mod
from routeaudit.model.gate_math import GateSpec
from routeaudit.model.loader import _resolve_dtype

DEVICES = [torch.device("cpu")]
if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda"))


def test_loader_auto_dtype_honors_prequantized_checkpoint_metadata():
    assert _resolve_dtype(SimpleNamespace(dtype="auto")) == "auto"


def test_loader_passes_checkpoint_revision_and_native_expert_backend(monkeypatch):
    calls = {}

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.zeros(()))
            self.config = SimpleNamespace(use_cache=True)

    def fake_tokenizer(model_id, **kwargs):
        calls["tokenizer"] = (model_id, kwargs)
        return SimpleNamespace(chat_template=None)

    def fake_model(model_id, **kwargs):
        calls["model"] = (model_id, kwargs)
        return FakeModel()

    monkeypatch.setattr(loader_mod.AutoTokenizer, "from_pretrained", fake_tokenizer)
    monkeypatch.setattr(loader_mod.AutoModelForCausalLM, "from_pretrained", fake_model)
    revision = "a" * 40
    cfg = SimpleNamespace(
        model=SimpleNamespace(
            hf_id="example/model",
            revision=revision,
            dtype="bfloat16",
            device_map="auto",
            n_layers=1,
            n_experts=2,
            top_k=1,
            d_model=4,
            arch=SimpleNamespace(name="deepseek"),
            load=SimpleNamespace(experts_implementation="deepgemm", attn_implementation=None),
        )
    )

    loader_mod.load_model(cfg)

    assert calls["tokenizer"][1]["revision"] == revision
    assert calls["model"][1]["revision"] == revision
    assert calls["model"][1]["experts_implementation"] == "deepgemm"
    assert calls["model"][1]["dtype"] is torch.bfloat16
    assert "torch_dtype" not in calls["model"][1]
    assert "attn_implementation" not in calls["model"][1]


def test_grouped_gate_keeps_excluded_groups_out_with_negative_bias():
    spec = GateSpec(
        scoring_func="sigmoid",
        top_k=1,
        use_bias=True,
        n_group=2,
        topk_group=1,
    )
    # Group 0 wins the group contest; excluded experts must stay ineligible even when
    # the selection bias makes eligible scores negative.
    scores = torch.tensor([[0.9, 0.8, 0.7, 0.6]])
    bias = torch.tensor([[-1.0, -1.0, -2.0, -2.0]])
    selected = gate_math.select(scores, bias, spec)

    assert selected.item() in (0, 1)


def test_grouped_gate_matches_transformers_5_9_reference():
    v3 = pytest.importorskip("transformers.models.deepseek_v3.modeling_deepseek_v3")
    cfg = v3.DeepseekV3Config(
        hidden_size=16,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
    )
    reference = v3.DeepseekV3MoE(cfg).eval()
    torch.manual_seed(1)
    with torch.no_grad():
        reference.gate.weight.normal_(0, 0.5)
        reference.gate.e_score_correction_bias.normal_(-0.5, 0.2)
        gate_output = reference.gate(torch.randn(12, cfg.hidden_size))
        if isinstance(gate_output, tuple):
            logits, weights, indices = gate_output
        else:
            logits = gate_output
            indices, weights = reference.route_tokens_to_experts(logits)

    spec = GateSpec(
        scoring_func="sigmoid",
        top_k=cfg.num_experts_per_tok,
        use_bias=True,
        n_group=cfg.n_group,
        topk_group=cfg.topk_group,
        norm_topk_prob=cfg.norm_topk_prob,
        routed_scaling_factor=cfg.routed_scaling_factor,
    )
    actual = gate_math.route(logits, reference.gate.e_score_correction_bias, spec)
    dense_reference = torch.zeros_like(logits).scatter_(1, indices, weights)

    assert torch.equal(actual.indices.sort(-1).values, indices.sort(-1).values)
    assert torch.allclose(actual.dense, dense_reference, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES, ids=str)
def test_expert_membership_counting_stays_on_device_and_matches_dense(device):
    idx = torch.tensor(
        [[[0, 2], [1, 3], [0, 1]], [[2, 3], [0, 3], [1, 2]]],
        device=device,
    )
    mask = torch.tensor([[True, False, True], [False, True, True]], device=device)

    actual = _expert_membership_counts(idx, mask, n_experts=4)
    dense = torch.zeros(2, 3, 4, dtype=torch.bool, device=device)
    dense.scatter_(-1, idx, True)
    expected = (dense & mask.unsqueeze(-1)).sum((0, 1))

    assert actual.device == device
    assert torch.equal(actual, expected)
