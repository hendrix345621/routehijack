"""The hook layer must not have changed for the models that already work.

`model/hooks.py`, `model/archspec.py` and `config.py` are on the critical path of every
existing pipeline phase (harvest, suffix search, eval). Adding DeepSeek/mHC support to
them is only safe if the softmax-gate path is bit-for-bit what it was: these tests pin
that, so a future change to the shared code can't quietly shift OLMoE/Qwen numbers.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from routeaudit.model import gate_math
from routeaudit.model.archspec import PRESETS, ArchSpec
from routeaudit.model.hooks import captured_forward
from routeaudit_deepseek_v4.hooks import DeepSeekV4HookManager as MoEHookManager


class _TinySoftMoE(nn.Module):
    """OLMoE/Qwen-shaped model: `model.layers[i].mlp` with a plain `nn.Linear` gate
    returning raw (T, E) logits, and an `experts` ModuleList."""

    class _Block(nn.Module):
        def __init__(self, d, e):
            super().__init__()
            self.gate = nn.Linear(d, e, bias=False)
            self.experts = nn.ModuleList(nn.Linear(d, d) for _ in range(e))

        def forward(self, h):
            probs = self.gate(h).softmax(-1)
            out = torch.stack([x(h) for x in self.experts], -2)     # (..., E, d)
            return torch.einsum("bte,bted->btd", probs, out)

    class _Layer(nn.Module):
        def __init__(self, d, e):
            super().__init__()
            self.mlp = _TinySoftMoE._Block(d, e)

        def forward(self, h):
            return h + self.mlp(h)

    def __init__(self, vocab=64, d=16, e=4, n_layers=3):
        super().__init__()
        self.model = nn.Module()
        self.model.embed = nn.Embedding(vocab, d)
        self.model.layers = nn.ModuleList(_TinySoftMoE._Layer(d, e) for _ in range(n_layers))
        self.lm_head = nn.Linear(d, vocab, bias=False)

    def forward(self, input_ids=None, use_cache=False, **kw):
        h = self.model.embed(input_ids)
        for layer in self.model.layers:
            h = layer(h)
        return SimpleNamespace(logits=self.lm_head(h))


def _build():
    torch.manual_seed(0)
    return _TinySoftMoE(), torch.tensor([[1, 2, 3, 4]])


# ── the existing capture path ────────────────────────────────────────────────

def test_router_logits_capture_is_the_raw_gate_output():
    """Unchanged contract: `capture.router_logits[l]` is exactly what the gate Linear
    returned, flattened (B*T, E) and detached. Every consumer reshapes it itself."""
    model, ids = _build()
    with MoEHookManager(model, ArchSpec.from_config(SimpleNamespace())) as hm:
        hm.capture_router_logits()
        model(input_ids=ids)
        captured = dict(hm.capture.router_logits)

    assert set(captured) == {0, 1, 2}
    h = model.model.embed(ids)
    for i, layer in enumerate(model.model.layers):
        assert torch.equal(captured[i], layer.mlp.gate(h)), f"layer {i} logits changed"
        h = h + layer.mlp(h)


def test_capture_switches_are_independent():
    """Only what was asked for gets stored — the new gate_input/routing captures must not
    turn on implicitly and must not populate when unused."""
    model, ids = _build()
    with MoEHookManager(model, ArchSpec.from_config(SimpleNamespace())) as hm:
        hm.capture_router_logits()
        model(input_ids=ids)
        assert hm.capture.router_logits and not hm.capture.gate_input
        assert not hm.capture.routing and not hm.capture.residual


def test_captured_forward_still_works():
    model, ids = _build()
    with captured_forward(model, router=True) as cap:
        model(input_ids=ids)
    assert len(cap.router_logits) == 3


def test_hooks_are_removed_on_exit():
    model, ids = _build()
    with MoEHookManager(model, ArchSpec.from_config(SimpleNamespace())) as hm:
        model(input_ids=ids)
    hm.capture.clear()
    model(input_ids=ids)
    assert not hm.capture.router_logits, "a hook survived context exit"


def test_router_mutator_still_rebuilds_selection():
    """The defense/steering path: a mutator on the raw-logit gate must reach the model's
    output. (`eval/generate.py` depends on this.)"""
    model, ids = _build()
    clean = model(input_ids=ids).logits.clone()
    with MoEHookManager(model, ArchSpec.from_config(SimpleNamespace())) as hm:
        hm.set_router_mutator(lambda logits, _l, _s: logits * 0 + 1.0)   # uniform routing
        mutated = model(input_ids=ids).logits
    assert not torch.allclose(clean, mutated), "router mutator had no effect"


# ── single-stream residual behavior ──────────────────────────────────────────

def test_residual_capture_reports_one_stream_on_a_standard_model():
    """A standard model must report n=1 so `reduce_residual` passes its residual through
    untouched. Reporting >1 here would make every residual analysis average over tokens."""
    model, ids = _build()
    spec = ArchSpec(name="olmoe", d_model=16)
    with MoEHookManager(model, spec) as hm:
        hm.capture_residual()
        model(input_ids=ids)
        assert set(hm.capture.residual_streams.values()) == {1}
        assert all(t.shape == (1, 4, 16) for t in hm.capture.residual.values())


# ── registry integrity ───────────────────────────────────────────────────────

def test_existing_presets_unchanged():
    """The four families that already worked must keep their exact module layout, and
    none may have inherited DeepSeek's recompute behavior."""
    for name in ("olmoe", "mixtral", "qwen", "phimoe"):
        s = ArchSpec.from_config(SimpleNamespace(arch=SimpleNamespace(name=name)))
        assert s.router_attr == "gate" and s.experts_attr == "experts"
        assert s.router_output in ("auto", "logits"), f"{name} must not be 'recompute'"
    assert PRESETS["olmoe"]["moe_block_attrs"] == ("mlp",)
    assert PRESETS["mixtral"]["moe_block_attrs"] == ("block_sparse_moe", "mlp")


def test_deepseek_preset_is_recompute():
    s = ArchSpec.from_config(SimpleNamespace(arch=SimpleNamespace(name="deepseek")))
    assert s.router_output == "recompute"
    assert s.router_bias_attr == "e_score_correction_bias"


def test_default_gate_spec_matches_legacy_softmax_behavior():
    """A config with no `routing:` block yields the flat-softmax gate — i.e. adding
    GateSpec changed nothing for the existing families."""
    gs = gate_math.GateSpec.from_config(SimpleNamespace(top_k=8))
    assert gs.scoring_func == "softmax" and not gs.grouped and not gs.use_bias
    assert gate_math.learned_router_layers(16, gs) == list(range(16))


def test_hf_autodetect_still_maps_known_families():
    from routeaudit.config import _model_ns_from_hf
    hf = SimpleNamespace(model_type="qwen3_moe", num_experts=128, num_experts_per_tok=8,
                         num_hidden_layers=48, hidden_size=2048, moe_intermediate_size=768)
    ns = _model_ns_from_hf(hf, "Qwen/Qwen3-30B-A3B", dtype="bfloat16", device_map="auto")
    assert ns.arch.name == "qwen"
    assert not hasattr(ns, "routing"), "a softmax family must get no routing block"
    assert not hasattr(ns, "mhc")


def test_is_plain_topk_gates_the_legacy_harvest_path():
    """`topk(router_logits)` is only correct for a plain softmax gate. Every DeepSeek
    variant must fail this predicate, or harvest would silently count the wrong experts."""
    assert gate_math.GateSpec(scoring_func="softmax", top_k=8).is_plain_topk
    for gs in (gate_math.GateSpec(scoring_func="sqrtsoftplus", top_k=6),
               gate_math.GateSpec(scoring_func="softmax", top_k=6, use_bias=True),
               gate_math.GateSpec(scoring_func="softmax", top_k=6, n_group=8, topk_group=4),
               gate_math.GateSpec(scoring_func="softmax", top_k=6, num_hash_layers=3)):
        assert not gs.is_plain_topk


def test_unroutable_layers_are_masked_out_of_expert_selection():
    """Hash layers register no activations, so every one of their cells scores exactly
    0.0 — which OUTRANKS real experts with negative Score_safe and would flood the
    safety set with cells no input can move. They must be masked, not left at zero."""
    from routeaudit.pipeline import _mask_unroutable
    gs = gate_math.GateSpec(scoring_func="sqrtsoftplus", top_k=6, num_hash_layers=3)
    score = torch.full((8, 4), -0.5)     # every real expert scores below the 0.0 wall
    score[:3] = 0.0                      # the hash layers, as harvest would leave them
    assert select_would_pick_hash(score), "precondition: unmasked, hash cells win"
    masked = _mask_unroutable(score, 8, gs)
    assert torch.isinf(masked[:3]).all() and (masked[3:] == -0.5).all()
    assert not select_would_pick_hash(masked)


def select_would_pick_hash(score: torch.Tensor) -> bool:
    from routeaudit.identify.select import select_safety_experts
    return any(e.layer < 3 for e in select_safety_experts(score, top_pct=0.20))


def test_mask_unroutable_is_a_noop_on_content_routed_families():
    from routeaudit.pipeline import _mask_unroutable
    score = torch.randn(16, 8)
    gs = gate_math.GateSpec(scoring_func="softmax", top_k=8)
    assert torch.equal(_mask_unroutable(score, 16, gs), score)


def test_suffix_search_refuses_unsupported_gates_and_accepts_the_old_ones():
    """The attack must fail at construction with an actionable message, not several
    minutes into a GPU run with an IndexError from `probs[expert_ids]`."""
    from routeaudit.attacks import UnsupportedGateError
    from routeaudit.attacks.suffix_search import _require_supported_gate

    softmax_spec = ArchSpec.from_config(SimpleNamespace(arch=SimpleNamespace(name="olmoe")))
    _require_supported_gate(softmax_spec, None)                                # legacy callers
    _require_supported_gate(softmax_spec, gate_math.GateSpec(top_k=8))         # explicit softmax

    deepseek = ArchSpec.from_config(SimpleNamespace(arch=SimpleNamespace(name="deepseek")))
    for spec, gs in ((deepseek, gate_math.GateSpec(scoring_func="sqrtsoftplus", top_k=6)),
                     (softmax_spec, gate_math.GateSpec(top_k=6, use_bias=True)),
                     (softmax_spec, gate_math.GateSpec(top_k=6, n_group=8, topk_group=4))):
        try:
            _require_supported_gate(spec, gs)
        except UnsupportedGateError as e:
            assert "architecture adapter" in str(e) and "Harvest and evaluation" in str(e)
        else:
            raise AssertionError(f"gate {gs.scoring_func} should be rejected")


def test_loader_dtype_resolution():
    """A QAT-native checkpoint has no bf16 original to cast to — `torch_dtype='auto'`
    honors what shipped. Anything unrecognized must raise, not silently pick a default."""
    from routeaudit.model.loader import _resolve_dtype
    assert _resolve_dtype(SimpleNamespace(dtype="bfloat16")) is torch.bfloat16
    assert _resolve_dtype(SimpleNamespace(dtype="fp8", expert_dtype="fp4")) == "auto"
    try:
        _resolve_dtype(SimpleNamespace(dtype="int3"))
    except ValueError as e:
        assert "not a supported load dtype" in str(e)
    else:
        raise AssertionError("unknown dtype must raise")


def test_hf_autodetect_builds_a_deepseek_routing_block():
    from routeaudit_deepseek_v4.config import model_ns_from_hf
    hf = SimpleNamespace(model_type="deepseek_v4", n_routed_experts=256,
                         num_experts_per_tok=6, num_hidden_layers=43, hidden_size=4096,
                         num_hash_layers=3, hc_mult=4)
    ns = model_ns_from_hf(hf, "deepseek-ai/DeepSeek-V4-Flash", dtype="fp8", device_map="auto")
    assert ns.arch.name == "deepseek"
    gs = gate_math.GateSpec.from_config(ns)
    assert gs.scoring_func == "sqrtsoftplus" and not gs.grouped
    assert gs.num_hash_layers == 3 and gs.routed_scaling_factor == 1.5
    assert ns.mhc.hc_mult == 4 and ns.mhc.hc_sinkhorn_iters == 20
