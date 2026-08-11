"""Differential parity against the OFFICIAL DeepSeek implementation, on CPU.

The validation ladder assumed Level 1 ("does our gate reproduce the released one?")
needed the released *weights*. It does not — it needs the released *code*. transformers
ships `DeepseekV3TopkRouter`, and instantiating it at toy size with
random weights exercises exactly the same arithmetic the 671B checkpoint runs. Random
weights are in fact a STRONGER test than a real checkpoint: they sweep regions of score
space a trained model may never visit, so ties, group masking and bias inversions all get
hit.

What this closes and what it doesn't:

  ✓ selection semantics       — sigmoid affinity, selection-only bias, node-limited top-k
  ✓ weighting semantics       — bias-free gather, `+1e-20` renormalization, scaling
  ✓ module discovery + hooks  — do we find `mlp.gate` / `mlp.experts` and read the right
                                tensor out of a real HF MoE block?
  ✗ DeepSeek-V4's two deltas  — `sqrt(softplus)` instead of sigmoid, and flat instead of
                                grouped selection. Those are one-line differences from the
                                verified V3 path; they get their own tests below against a
                                transcription of `DeepseekV4TopKRouter.forward`.
  ✗ semantics                 — random weights have no behavior. Nothing here says
                                anything about refusal, safety experts, or ASR.

Skips cleanly when the installed transformers predates a given model.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from routeaudit.model import gate_math
from routeaudit.model.gate_math import GateSpec

v3 = pytest.importorskip("transformers.models.deepseek_v3.modeling_deepseek_v3",
                         reason="transformers without deepseek_v3")

E, K, D, NG, TG = 32, 6, 64, 4, 2


def _v3_config(**kw):
    cfg = v3.DeepseekV3Config(
        hidden_size=D, intermediate_size=2 * D, moe_intermediate_size=2 * D,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        n_routed_experts=E, n_shared_experts=1, num_experts_per_tok=K,
        n_group=NG, topk_group=TG, norm_topk_prob=True, routed_scaling_factor=2.5,
        first_k_dense_replace=1, vocab_size=128, max_position_embeddings=64,
        q_lora_rank=None, kv_lora_rank=16, qk_rope_head_dim=8, v_head_dim=16,
        qk_nope_head_dim=8,
    )
    for k, val in kw.items():
        setattr(cfg, k, val)
    return cfg


def _our_spec(cfg) -> GateSpec:
    return GateSpec(scoring_func="sigmoid", top_k=cfg.num_experts_per_tok, use_bias=True,
                    n_group=cfg.n_group, topk_group=cfg.topk_group,
                    norm_topk_prob=cfg.norm_topk_prob,
                    routed_scaling_factor=cfg.routed_scaling_factor)


def _align(w, idx, w_ref, idx_ref):
    """Reference top-k is computed with `sorted=False`, so the two implementations may
    emit the same experts in different order. Compare as sets, then reorder the reference
    weights into our index order before comparing values — an ordering difference must not
    be able to masquerade as a numerical one."""
    order, order_ref = idx.argsort(-1), idx_ref.argsort(-1)
    out = torch.empty_like(w_ref)
    out.scatter_(-1, order, torch.gather(w_ref, -1, order_ref))
    return out


def _official_v3_route(moe, hidden):
    """Handle both old fused-router and current split-router Transformers APIs."""
    output = moe.gate(hidden)
    if isinstance(output, tuple):
        return output
    indices, weights = moe.route_tokens_to_experts(output)
    return output, weights, indices


# ── the grouped/biased gate (DeepSeek-V2/V3) ─────────────────────────────────

@pytest.mark.parametrize("bias_scale", [0.0, 0.05, 0.5])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_grouped_gate_matches_official_v3(seed, bias_scale):
    """Our `route()` must reproduce `DeepseekV3TopkRouter.forward` exactly."""
    torch.manual_seed(seed)
    cfg = _v3_config()
    moe = v3.DeepseekV3MoE(cfg).eval()
    with torch.no_grad():
        moe.gate.weight.normal_(0, 0.5)
        moe.gate.e_score_correction_bias.normal_(0, bias_scale)

    h = torch.randn(24, D)
    with torch.no_grad():
        logits, w_ref, idx_ref = _official_v3_route(moe, h)
        rr = gate_math.route(logits, moe.gate.e_score_correction_bias, _our_spec(cfg))

    assert torch.equal(rr.indices.sort(-1).values, idx_ref.sort(-1).values), \
        "different experts selected"
    aligned = _align(rr.weights, rr.indices, w_ref, idx_ref)
    assert torch.allclose(rr.weights, aligned, atol=0, rtol=0), \
        f"gating weights diverge by {(rr.weights - aligned).abs().max():.3e}"


def test_dense_routing_weights_match_official_v3():
    """The dense (T, E) form our analyses consume must place the reference weights on the
    reference experts — the scatter is where an index/weight misalignment would hide."""
    torch.manual_seed(3)
    cfg = _v3_config()
    moe = v3.DeepseekV3MoE(cfg).eval()
    with torch.no_grad():
        moe.gate.weight.normal_(0, 0.5)
        moe.gate.e_score_correction_bias.normal_(0, 0.1)
        h = torch.randn(16, D)
        logits, w_ref, idx_ref = _official_v3_route(moe, h)
        dense_ref = torch.zeros(16, E).scatter_(1, idx_ref, w_ref)
        rr = gate_math.route(logits, moe.gate.e_score_correction_bias, _our_spec(cfg))
    assert torch.allclose(rr.dense, dense_ref, atol=1e-6)


def test_bias_is_selection_only_in_the_official_implementation():
    """Confirms the premise our loss/analysis split rests on, against the reference:
    changing the bias changes `indices` but never the underlying scores."""
    torch.manual_seed(4)
    cfg = _v3_config()
    moe = v3.DeepseekV3MoE(cfg).eval()
    with torch.no_grad():
        moe.gate.weight.normal_(0, 0.5)
        h = torch.randn(32, D)
        logits, w_a, idx_a = _official_v3_route(moe, h)
        moe.gate.e_score_correction_bias.normal_(0, 1.0)
        logits_b, w_b, idx_b = _official_v3_route(moe, h)
    assert torch.equal(logits, logits_b), "selection bias changed the router logits"
    changed = (idx_a.sort(-1).values != idx_b.sort(-1).values).any(-1)
    assert changed.any(), "bias scale too small to move selection — test is vacuous"
    # Where selection did NOT change, weights must be bit-identical: the bias never
    # touches the weighting path.
    same = ~changed
    if same.any():
        assert torch.equal(_align(w_a[same], idx_a[same], w_b[same], idx_b[same]), w_a[same])


@pytest.mark.parametrize("bias_mean", [-0.5, -1.0, -1.5, -3.0])
def test_group_mask_fill_value_matches_under_a_negative_bias(bias_mean):
    """Negative bias makes the mask sentinel observable, so pin it to the reference.

    Current Transformers routers use negative infinity for excluded groups, preventing
    them from re-entering top-k under an unusually negative selection bias.
    """
    torch.manual_seed(0)
    cfg = _v3_config()
    moe = v3.DeepseekV3MoE(cfg).eval()
    with torch.no_grad():
        moe.gate.weight.normal_(0, 0.5)
        moe.gate.e_score_correction_bias.normal_(bias_mean, 0.2)
        h = torch.randn(64, D)
        logits, w_ref, idx_ref = _official_v3_route(moe, h)
        rr = gate_math.route(logits, moe.gate.e_score_correction_bias, _our_spec(cfg))
    assert torch.equal(rr.indices.sort(-1).values, idx_ref.sort(-1).values), \
        "group-mask fill value diverges from the reference under a negative bias"
    assert torch.allclose(rr.weights, _align(rr.weights, rr.indices, w_ref, idx_ref),
                          atol=1e-6)


def test_ineligible_experts_are_unreachable_in_the_margin():
    """Excluded groups have ``-inf`` margin and the explicit mask agrees."""
    torch.manual_seed(1)
    cfg = _v3_config()
    moe = v3.DeepseekV3MoE(cfg).eval()
    with torch.no_grad():
        moe.gate.weight.normal_(0, 0.5)
        logits, _, _ = _official_v3_route(moe, torch.randn(16, D))
        rr = gate_math.route(logits, moe.gate.e_score_correction_bias, _our_spec(cfg))

    assert rr.eligible is not None and not rr.eligible.all(), "grouped gate must mask some"
    naive = gate_math.selection_margin(rr.sel_scores, _our_spec(cfg))
    masked = gate_math.selection_margin(rr.sel_scores, _our_spec(cfg), eligible=rr.eligible)
    assert torch.isinf(masked[~rr.eligible]).all() and (masked[~rr.eligible] < 0).all()
    assert torch.equal(masked[rr.eligible], naive[rr.eligible])


# ── hooks against a real HF MoE model ────────────────────────────────────────

def test_hooks_capture_the_official_models_own_routing():
    """End-to-end: run a real (tiny, random) DeepseekV3ForCausalLM and check our hook
    layer discovers its MoE blocks and recomputes the routing the model itself used.

    This is the integration half of the ladder — module paths, the gate's return shape,
    dense-vs-MoE layer skipping, and the recompute path, all against official code.
    """
    from routeaudit.model.archspec import ArchSpec
    from routeaudit_deepseek_v4.hooks import DeepSeekV4HookManager as MoEHookManager

    torch.manual_seed(5)
    cfg = _v3_config(num_hidden_layers=3, first_k_dense_replace=1)
    model = v3.DeepseekV3ForCausalLM(cfg).eval()
    with torch.no_grad():
        for layer in model.model.layers:
            if hasattr(layer.mlp, "gate"):
                layer.mlp.gate.weight.normal_(0, 0.5)
                layer.mlp.gate.e_score_correction_bias.normal_(0, 0.1)

    spec = ArchSpec(name="deepseek", base_attr="model", layers_attr="layers",
                    moe_block_attrs=("mlp",), router_attr="gate", experts_attr="experts",
                    router_output="recompute", n_layers=cfg.num_hidden_layers,
                    n_experts=E, top_k=K, d_model=D)
    gs = _our_spec(cfg)
    gs = GateSpec(**{**vars(gs), "first_k_dense_replace": cfg.first_k_dense_replace})

    ids = torch.randint(0, cfg.vocab_size, (1, 12))
    with MoEHookManager(model, spec) as hm, torch.no_grad():
        hm.capture_routing(gs)
        model(input_ids=ids)
        captured = dict(hm.capture.routing)

    moe_layers = [i for i, l in enumerate(model.model.layers) if hasattr(l.mlp, "gate")]
    assert set(captured) == set(moe_layers), \
        f"captured {sorted(captured)}, model has MoE at {moe_layers} (dense layer skipped?)"

    # Re-drive each block's own routing from the captured gate input and compare.
    with MoEHookManager(model, spec) as hm, torch.no_grad():
        hm.capture_gate_input().capture_routing(gs)
        model(input_ids=ids)
        for li in moe_layers:
            block = model.model.layers[li].mlp
            output = block.gate(hm.capture.gate_input[li])
            if isinstance(output, tuple):
                _, w_ref, idx_ref = output
            else:
                idx_ref, w_ref = block.route_tokens_to_experts(output)
            rr = hm.capture.routing[li]
            if isinstance(output, tuple):
                assert rr.official_weights is not None and rr.official_indices is not None
                assert torch.equal(rr.official_indices, idx_ref)
                assert torch.equal(rr.official_weights, w_ref)
            assert torch.equal(rr.indices.sort(-1).values, idx_ref.sort(-1).values), \
                f"layer {li}: hook-captured selection differs from the model's own"
            assert torch.allclose(rr.weights, _align(rr.weights, rr.indices, w_ref, idx_ref),
                                  atol=1e-6), f"layer {li}: weights differ"


# ── DeepSeek-V4's two deltas, against a transcription of the released router ──

def _v4_reference(logits, bias, top_k=6, route_scale=1.5):
    """Transcription of `DeepseekV4TopKRouter.forward` (transformers
    `models/deepseek_v4/modeling_deepseek_v4.py`). Kept separate from our implementation
    so the test compares two independently-written code paths rather than one.

    Not skippable via importorskip: this transformers build predates deepseek_v4, so the
    transcription IS the reference until the module ships. When it does, swap this for the
    real import — the assertions below need no change.
    """
    scores = F.softplus(logits).sqrt()
    indices = torch.topk(scores + bias, top_k, dim=-1, sorted=False).indices
    weights = scores.gather(1, indices)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return logits, weights * route_scale, indices


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_flat_sqrtsoftplus_gate_matches_the_v4_reference(seed):
    torch.manual_seed(seed)
    logits = torch.randn(48, 256) * 2
    bias = torch.randn(256) * 0.1
    gs = GateSpec(scoring_func="sqrtsoftplus", top_k=6, use_bias=True, n_group=1,
                  norm_topk_prob=True, routed_scaling_factor=1.5)

    _, w_ref, idx_ref = _v4_reference(logits, bias)
    rr = gate_math.route(logits, bias, gs)

    assert torch.equal(rr.indices.sort(-1).values, idx_ref.sort(-1).values)
    assert torch.allclose(rr.weights, _align(rr.weights, rr.indices, w_ref, idx_ref),
                          atol=1e-6)


def test_v4_weights_sum_to_the_route_scale():
    """`norm_topk_prob` then `* routed_scaling_factor` means the per-token gating mass is
    exactly the scale factor — a cheap invariant that catches a dropped renormalization."""
    torch.manual_seed(9)
    gs = GateSpec(scoring_func="sqrtsoftplus", top_k=6, use_bias=True, n_group=1,
                  routed_scaling_factor=1.5)
    rr = gate_math.route(torch.randn(64, 256) * 3, torch.randn(256) * 0.2, gs)
    assert torch.allclose(rr.weights.sum(-1), torch.full((64,), 1.5), atol=1e-5)
