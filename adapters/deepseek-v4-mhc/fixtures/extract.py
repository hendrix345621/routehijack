"""Extract ground-truth component fixtures from a released DeepSeek checkpoint.

Level 1 of the validation ladder. Level 0 tests the mechanism synthetically; this step
extracts independent component evidence from the released DeepSeek-V4-Flash weights.

    Level 0  synthetic model, random weights, fp32, CPU   -> RUNS NOW
    Level 1  component fixtures from the released model   -> this script
    Level 2  full forward parity on a fixed prompt        -> validate.py --full
    Level 3  semantic experiments on the real checkpoint  -> the diagnostics themselves

The point of fixtures rather than end-to-end comparison: each failure localizes to one
component. A whole-model logit mismatch tells you something is wrong; a gate fixture
mismatch tells you the scoring function is wrong.

    python fixtures/extract.py --config deepseek-v4-flash \\
        --out fixtures/v4_flash_fixtures.pt
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

# Prefer the repository's vendored runtime when invoked directly from a clone.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from routeaudit import config as cfg_mod
from routeaudit import ui
from routeaudit.model import load_model
from routeaudit.model.archspec import ArchSpec
from routeaudit.model.gate_math import GateSpec, learned_router_layers
from routeaudit_deepseek_v4 import mhc, precision
from routeaudit_deepseek_v4.hooks import DeepSeekV4HookManager as MoEHookManager
from routeaudit.model.prompting import encode_prompt

# A short, fixed prompt. Determinism matters more than content: same-device official
# parity is exact, and portable replay has a separately documented device tolerance.
FIXTURE_PROMPT = "The capital of France is"
FIXTURE_FORMAT_VERSION = 2


def _align_topk(
    values: torch.Tensor, indices: torch.Tensor, target_indices: torch.Tensor
) -> torch.Tensor:
    """Align per-expert top-k values to another implementation's index order."""
    order = indices.argsort(-1)
    target_order = target_indices.argsort(-1)
    aligned = torch.empty_like(values)
    aligned.scatter_(-1, target_order, torch.gather(values, -1, order))
    return aligned


@torch.no_grad()
def extract_hash_table(
    model, layer_idx: int, vocab_size: int, top_k: int, batch: int = 4096
) -> torch.Tensor | None:
    """Recover the token-id → expert table for a hash-routed layer.

    Works even when the table tensor isn't directly exposed: hash routing depends on the
    token id and nothing else, so feeding each id once determines its experts. Returns
    None when the module doesn't publish a `tid2eid`-style buffer and there is no
    supported way to sweep it — better a missing fixture than a fabricated one.
    """
    base = getattr(model, "model", model)
    layer = base.layers[layer_idx]
    block = getattr(layer, "mlp", None)
    gate = getattr(block, "gate", None)
    for owner in (block, gate):
        for attr in ("tid2eid", "hash_table", "token_expert_map"):
            t = getattr(owner, attr, None)
            if isinstance(t, torch.Tensor):
                return t.detach().cpu()
    ui.warn(
        f"layer {layer_idx}: no hash table buffer found — skipping the hash fixture. "
        f"(Sweeping the vocabulary needs a model-specific routing entry point.)"
    )
    return None


@torch.inference_mode()
def extract(config: str, out: Path, *, layer: int | None = None) -> dict:
    cfg = cfg_mod.load(config)
    _ok, msg = precision.check_quant_policy(cfg.model, "none", precision.Claim.QUANTITATIVE)
    if msg:
        ui.warn(msg)

    loaded = load_model(cfg)
    model, tok = loaded.model, loaded.tokenizer
    dmap = getattr(model, "hf_device_map", {}) or {}
    offloaded = sorted({str(v) for v in dmap.values() if str(v) == "cpu" or str(v).startswith("disk")})
    if offloaded:
        raise RuntimeError(
            f"model offloaded to {offloaded}; this is not the fully-GPU native fixture run. "
            "Use a Blackwell node with enough free aggregate VRAM."
        )
    spec: ArchSpec = loaded.spec
    gs = GateSpec.from_config(cfg.model)
    device = next(model.parameters()).device
    ids = encode_prompt(
        tok, FIXTURE_PROMPT, want_template=getattr(cfg.model, "use_chat_template", True), device=device
    ).unsqueeze(0)

    learned = learned_router_layers(spec.n_layers, gs)
    target = layer if layer is not None else (learned[len(learned) // 2] if learned else 0)

    with MoEHookManager(model, spec) as hm:
        hm.capture_gate_input().capture_routing(gs).capture_residual().capture_mhc_maps()
        out_logits = model(input_ids=ids, use_cache=False).logits

    if target not in hm.capture.gate_input or target not in hm.capture.routing:
        raise RuntimeError(f"layer {target}: gate input/routing capture is missing")
    if target not in hm.capture.residual:
        raise RuntimeError(f"layer {target}: residual capture is missing")
    n_streams = hm.capture.residual_streams.get(target, 1)
    if n_streams != 4:
        raise RuntimeError(f"layer {target}: expected four mHC residual streams, captured {n_streams}")

    gpus = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        gpus.append(
            {
                "index": i,
                "name": p.name,
                "compute_capability": f"{p.major}.{p.minor}",
                "total_memory": p.total_memory,
            }
        )

    fx: dict = {
        "meta": {
            "fixture_format_version": FIXTURE_FORMAT_VERSION,
            "hf_id": getattr(cfg.model, "hf_id", config),
            "requested_revision": getattr(cfg.model, "revision", None),
            "resolved_revision": getattr(model.config, "_commit_hash", None),
            "prompt": FIXTURE_PROMPT,
            "input_ids": ids.cpu(),
            "gate_spec": vars(gs),
            "arch_spec": vars(spec),
            "torch_version": str(torch.__version__),
            "transformers_version": importlib.metadata.version("transformers"),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cuda_runtime": torch.version.cuda,
            "gpus": gpus,
            "device_map": {str(k): str(v) for k, v in dmap.items()},
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "precision": precision.policy_banner(cfg.model, "none", precision.Claim.QUANTITATIVE),
        },
        "logits": out_logits[0, -1].float().cpu(),
    }

    # (a) one gate call, including the router's own returned values   [Blocker 3]
    rr = hm.capture.routing[target]
    official_set = (
        rr.official_indices is not None
        and torch.equal(rr.indices.sort(-1).values, rr.official_indices.sort(-1).values)
    )
    if official_set and rr.official_weights is not None:
        official_aligned = _align_topk(rr.official_weights, rr.official_indices, rr.indices)
        official_dev = float((rr.weights.float() - official_aligned.float()).abs().max())
        official_exact = torch.equal(rr.weights, official_aligned)
    else:
        official_dev = None
        official_exact = False
    fx["gate"] = {
        "layer": target,
        "gate_input": hm.capture.gate_input[target].float().cpu(),
        "scores": rr.scores.cpu(),
        "sel_scores": rr.sel_scores.cpu(),
        "indices": rr.indices.cpu(),
        "weights": rr.weights.cpu(),
        "official_indices": rr.official_indices.cpu() if rr.official_indices is not None else None,
        "official_weights": rr.official_weights.cpu() if rr.official_weights is not None else None,
        "same_device_parity": {
            "official_output_captured": rr.official_indices is not None and rr.official_weights is not None,
            "same_expert_set": official_set,
            "weights_exact": official_exact,
            "weights_max_abs_dev": official_dev,
        },
    }

    # (b) the residual state, with its stream count                    [Blockers 1-2]
    fx["residual"] = {
        "layer": target,
        "hidden": hm.capture.residual[target].float().cpu(),
        "n_streams": n_streams,
    }

    sites = {}
    for site, captured in hm.capture.mhc_maps.get(target, {}).items():
        b = mhc.residual_matrix(captured.comb)
        sites[site] = {
            "hidden_streams": captured.hidden_streams.float().cpu(),
            "post": captured.post.float().cpu(),
            "comb": captured.comb.float().cpu(),
            "collapsed": captured.collapsed.float().cpu(),
            "b_path": mhc.b_path_conservation_check(b, captured.hidden_streams),
        }
    if sites:
        fx["mhc_maps"] = {"layer": target, "sites": sites}

    # (c) hash table slice for the leading layers                      [Blocker 4]
    if gs.num_hash_layers:
        vocab = int(getattr(model.config, "vocab_size", 0) or 0)
        table = extract_hash_table(model, gs.first_k_dense_replace, vocab, gs.top_k)
        if table is None:
            raise RuntimeError("hash-routed layer exists but its token→expert table was not captured")
        fx["hash"] = {"layer": gs.first_k_dense_replace, "table": table[:4096]}

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fx, out)
    ui.ok(f"fixtures → {out}  ({', '.join(k for k in fx if k != 'meta')})")
    return fx


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--layer",
        type=int,
        default=None,
        help="which layer to fixture (default: a middle content-routed one)",
    )
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "v4_flash_fixtures.pt")
    args = ap.parse_args()
    extract(args.config, args.out, layer=args.layer)


if __name__ == "__main__":
    main()
