"""Shared plumbing for the mHC diagnostic tests — SEPARATE EXPERIMENT (gitignored).

Loads a frontier model as cheaply as the precision policy allows and gives forward-only
access to the two signals the tests need at the boundary token t*:
  • next-token logits   → refusal / affirmative receptivity
  • routing at t*        → which experts gate the decision, plus the selection scores
                           the margin census needs

There is now ONE routing path for every architecture. Gate semantics live in
`routeaudit.model.gate_math` and are selected by the config's `routing:` block, so a
softmax gate (OLMoE/Qwen), a sigmoid + node-limited gate (DeepSeek-V2/V3) and a
sqrt(softplus) + flat-top-k gate (DeepSeek-V4-Flash) all come back as the same
`RouteResult`. The old `if grouped: ... else: ...` fork — and its hardcoded sigmoid,
which computed the wrong affinity for V4 — is gone.

Reuses the installed `routeaudit` package read-only; nothing here mutates it.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from routeaudit import config as cfg_mod
from routeaudit import ui
from routeaudit.model import gate_math
from routeaudit_deepseek_v4 import precision
from routeaudit.model.archspec import ArchSpec
from routeaudit.model.gate_math import GateSpec, RouteResult
from routeaudit_deepseek_v4.hooks import DeepSeekV4HookManager as MoEHookManager
from routeaudit.model.prompting import encode_prompt

_DTYPE = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

# Opener tokens for the two receptivity signals (first sub-token of each).
REFUSAL_OPENERS = ["I", " I", "Sorry", " Sorry", "I'm", " I'm", "As", " As", "I cannot", " I can"]
AFFIRM_OPENERS = ["Sure", " Sure", "Here", " Here", "Certainly", " Certainly", "Okay", " Okay", "```"]


@dataclass
class DiagModel:
    model: object
    tok: object
    spec: ArchSpec            # module layout
    gate_spec: GateSpec       # gate semantics
    cfg: object

    @property
    def learned_layers(self) -> list[int]:
        """Layers whose routing responds to content. Hash-routed and dense layers are
        excluded — routing there is a token-id lookup or nonexistent, so including them
        dilutes every per-layer statistic with layers that cannot move."""
        return gate_math.learned_router_layers(self.spec.n_layers, self.gate_spec)


def load_quantized(config: str, *, quant: str = "nf4", device_map: str = "auto",
                   max_memory=None, enable_thinking=None,
                   claim: precision.Claim = precision.Claim.STRUCTURAL) -> DiagModel:
    """Load via a routeaudit config (nickname/path) with diagnostic quantization, subject
    to the precision policy. quant ∈ {none, nf4, int8}.

    The policy matters here: NF4 on a bf16 sibling is a reasonable probing trade, but NF4
    on DeepSeek-V4-Flash is a mistake — those fp8/fp4 weights are QAT-native, so
    re-quantizing adds error the deployed model does not have. `check_quant_policy`
    refuses that combination rather than warning about it.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = cfg_mod.load(config)
    mid = cfg.model.hf_id

    ok, msg = precision.check_quant_policy(cfg.model, quant, claim)
    if not ok:
        raise ValueError(msg)
    if msg:
        ui.warn(msg)

    dtype = _DTYPE.get(getattr(cfg.model, "dtype", "bfloat16"), torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    kw = dict(device_map=device_map, trust_remote_code=True, attn_implementation="sdpa")
    if max_memory:
        kw["max_memory"] = max_memory
    if quant and quant != "none":
        from transformers import BitsAndBytesConfig
        if quant == "nf4":
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype)
        elif quant in ("int8", "8bit"):
            kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        kw["torch_dtype"] = dtype
    try:
        model = AutoModelForCausalLM.from_pretrained(mid, **kw)
    except (TypeError, ValueError):
        kw.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(mid, **kw)
    model.eval()

    if enable_thinking is None:
        enable_thinking = getattr(cfg.model, "enable_thinking", None)
    if enable_thinking is not None:
        from routeaudit.model import prompting
        prompting.set_chat_template_kwargs({"enable_thinking": bool(enable_thinking)})

    spec = ArchSpec.from_config(cfg.model)
    gs = GateSpec.from_config(cfg.model)
    ui.ok(f"loaded {mid}  (arch={spec.name}, gate={gs.scoring_func}, "
          f"{'grouped' if gs.grouped else 'flat'} top-{gs.top_k}, "
          f"hash_layers={gs.num_hash_layers}, quant={quant})")
    return DiagModel(model=model, tok=tok, spec=spec, gate_spec=gs, cfg=cfg)


def first_token_ids(tok, strings) -> list[int]:
    out = []
    for s in strings:
        ids = tok.encode(s, add_special_tokens=False)
        if ids:
            out.append(ids[0])
    return sorted(set(out))


# ─────────────────────────── the one routing path ───────────────────────────


def _boundary_slice(rr: RouteResult) -> RouteResult:
    """Keep only the boundary token t* (the last position), preserving the (1, E) rank so
    `gate_math.selection_margin` and friends still apply."""
    return RouteResult(scores=rr.scores[-1:].float().cpu(),
                       sel_scores=rr.sel_scores[-1:].float().cpu(),
                       indices=rr.indices[-1:].cpu(),
                       weights=rr.weights[-1:].float().cpu(),
                       dense=rr.dense[-1:].float().cpu())


@torch.no_grad()
def boundary_routing(dm: DiagModel, prompt: str, want_template=True):
    """Return (next_logits[V], {layer: RouteResult at t*}).

    The full RouteResult is kept — not just the gating weights — because the two tensors
    answer different questions: `dense`/`weights` are bias-free and say how much each
    expert contributes, `sel_scores` include the balancing bias and are the only place a
    selection margin can be measured.
    """
    model, tok = dm.model, dm.tok
    device = next(model.parameters()).device
    ids = encode_prompt(tok, prompt, want_template=want_template, device=device).unsqueeze(0)
    with MoEHookManager(model, dm.spec) as hm:
        hm.capture_routing(dm.gate_spec)
        out = model(input_ids=ids, use_cache=False)
        routing = {l: _boundary_slice(rr) for l, rr in hm.capture.routing.items()}
    return out.logits[0, -1].float().cpu(), routing


@torch.no_grad()
def boundary_logits_and_routing(dm: DiagModel, prompt: str, want_template=True):
    """Return (next_logits[V], {layer: gating weights[E]}) at t*.

    Kept for the tests that only want routing mass. Note the weights are the DENSE
    post-top-k gating weights (zero for experts that don't fire), not a softmax over all
    experts: that is what actually scales expert outputs, and it is the only definition
    that means the same thing across a softmax, a sigmoid and a sqrt(softplus) gate.
    """
    logits, routing = boundary_routing(dm, prompt, want_template)
    return logits, {l: rr.dense[0] for l, rr in routing.items()}
