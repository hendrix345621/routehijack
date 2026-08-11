"""Response-driven expert activation frequency.

Implements the RouteHijack paper's Eq. 3 (p. 4):

    F_l(e | a) = (1/|a|) · Σ_t  𝟙[e ∈ TopK(logits_{l,t})]

`a` is a response sequence. Query/prompt tokens are masked out per the paper's
response-driven profiling (p. 5: response-driven beats prompt-driven,
69.3% vs 30.5% ASR — masking matters).

Performance notes
─────────────────
The naive version ran one sequence per forward (batch size 1): the GPU finished
each tiny forward in milliseconds, then sat idle while Python set up the next one
— latency-bound, low GPU utilisation. The reworked version:

  - **Pre-tokenizes everything** before touching the GPU (CPU work happens once).
  - **Batches `batch_size` sequences per forward**, right-padded to the batch's
    longest sequence, with an attention mask so real tokens never attend to pads.
    Router logits come back flattened (B*T, E); we reshape to (B, T, E) and mask
    out prompt + padding positions per sequence. This is the big win — it keeps
    the GPU busy and cuts wall-time ~10× on short sequences. Right-padding + the
    attention mask yields per-position logits identical to the batch-1 path, so
    the frequencies are unchanged — only speed differs.
  - **Sorts by length** before batching so each batch pads to a similar size
    (minimal wasted compute on padding).
  - **Reuses one hook manager** across the whole sweep.
  - **Accumulates counts on the GPU**, syncing to CPU only at the end.
  - **Truncates responses** to `max_response_tokens` so the long tail doesn't
    dominate wall-time.

Membership-count memory scales with `batch_size * counted_tokens * top_k`, not
`batch_size * max_total_tokens * n_experts`; no dense expert mask is materialised.
Lower `batch_size` if VRAM is tight."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch

from .. import ui
from ..model.hooks import MoEHookManager


@dataclass
class ExpertFreq:
    """Per-(layer, expert) activation frequency over a corpus.

    Stored as a dense (n_layers, n_experts) float tensor on CPU."""

    freq: torch.Tensor  # (L, E), float64
    n_tokens: int

    def __getitem__(self, key: tuple[int, int]) -> float:
        layer, expert = key
        return float(self.freq[layer, expert])


def _expert_membership_counts(
    indices: torch.Tensor,
    count_mask: torch.Tensor,
    n_experts: int,
) -> torch.Tensor:
    """Count selected ids without materialising a dense ``(B, T, E)`` mask.

    Top-k selection contains unique expert ids per token. Selecting only counted
    rows and applying ``bincount`` reduces temporary storage from ``B*T*E`` values
    to ``counted_tokens*K`` and keeps the calculation on the input device.
    """
    selected = indices[count_mask]
    if selected.numel() == 0:
        return torch.zeros(n_experts, dtype=torch.int64, device=indices.device)
    return torch.bincount(selected.reshape(-1), minlength=n_experts)


@torch.inference_mode()
def compute_expert_freq(
    model: torch.nn.Module,
    tokenizer,
    sequences: Iterable[dict],
    *,
    n_layers: int,
    n_experts: int,
    top_k: int,
    device: str | torch.device | None = None,
    desc: str = "freq",
    max_response_tokens: int | None = 256,
    max_total_tokens: int = 1024,
    batch_size: int = 16,
    spec=None,
    gate_spec=None,
    use_chat_template: bool = True,
    span: str = "all",
    max_think_tokens: int = 256,
) -> ExpertFreq:
    """Compute F_l(e | a) over a corpus of sequences.

    Each sequence is a dict with:
      - 'prompt'   : str, query text (its tokens are MASKED)
      - 'response' : str, response text (its tokens are COUNTED)

    Args:
      max_response_tokens: truncate the response to at most this many tokens.
                           Most refusals / completions don't need >256 tokens to
                           characterise routing. Set None to disable.
      max_total_tokens:   cap prompt+response combined. Prevents one pathological
                           long sequence from dominating wall-time.
      batch_size:         sequences per forward pass. Higher = better GPU
                          utilisation; lower if VRAM is tight.
      gate_spec:          a :class:`~routeaudit.model.gate_math.GateSpec`. Required for
                          any gate whose selection isn't `logits.topk(k)` — see below.
      span:               which response tokens to COUNT — "all" (every response
                          token, the non-reasoning default), "answer" (post-`</think>`
                          only), "think" (the trace), or "delimiter" (the `</think>`
                          window). On a corpus with no traces all four coincide.
      max_think_tokens:   cap on trace length. A long trace is truncated from its
                          HEAD, keeping the tail adjacent to `</think>` — head
                          truncation would delete the answer entirely, since the trace
                          precedes it. The kept tail is also the region the answer
                          attends to most, and where refusal intent collapses.
    """
    device = device or next(model.parameters()).device
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    # ── 1) Pre-tokenize everything on CPU before touching the GPU. ──
    # The tokenizer call is a hidden cost; doing it inside the GPU loop would
    # leave the GPU idle. Pull it out so the GPU gets a steady stream of work.
    # Each entry is (ids[CPU,long], n_prompt) — kept on CPU; batched onto the GPU below.
    from ..model.prompting import profiling_spans

    prepped: list[tuple[torch.Tensor, torch.Tensor]] = []
    n_dropped_empty = 0
    for item in ui.iter_with_progress(list(sequences), desc=f"{desc} (tokenize)"):
        prompt = item["prompt"]
        response = item["response"]
        if not response:
            continue
        # Render the query through the chat template (query + assistant marker are
        # the CONTEXT to mask); count only the requested response span (Eq. 3, §5.1).
        s = profiling_spans(tokenizer, prompt, response, span=span, want_template=use_chat_template)
        ids, mask, n_ctx = s.ids, s.mask, s.n_ctx
        if ids.shape[0] <= n_ctx:
            continue

        # Shrink an over-long trace from its HEAD, keeping the tail + `</think>` +
        # answer. Dropping the response tail instead (the old behaviour) removes the
        # answer entirely in thinking mode, since the trace comes first.
        if s.think_len > max_think_tokens:
            t0, t1 = s.think_span
            cut0 = t1 - max_think_tokens
            keep = torch.ones(ids.shape[0], dtype=torch.bool)
            keep[t0:cut0] = False
            ids, mask = ids[keep], mask[keep]
            n_ctx = int(keep[:n_ctx].sum())

        if max_response_tokens is not None and ids.shape[0] - n_ctx > max_response_tokens:
            # Trim the response TAIL only when the counted span survives it; otherwise
            # keep the sequence whole rather than silently emptying its mask.
            trimmed_mask = mask[: n_ctx + max_response_tokens]
            if bool(trimmed_mask.any()):
                ids, mask = ids[: n_ctx + max_response_tokens], trimmed_mask
        if ids.shape[0] > max_total_tokens:
            keep_n = max_total_tokens - n_ctx
            if keep_n <= 0:
                continue
            trimmed_mask = torch.cat([mask[:n_ctx], mask[n_ctx : n_ctx + keep_n]])
            if not bool(trimmed_mask.any()):
                continue
            ids = torch.cat([ids[:n_ctx], ids[n_ctx : n_ctx + keep_n]])
            mask = trimmed_mask

        if not bool(mask.any()):
            # No tokens in the requested span — e.g. `span="answer"` on a generation
            # that never closed its trace. Counted and reported, never counted as zero.
            n_dropped_empty += 1
            continue
        prepped.append((ids, mask))

    if n_dropped_empty:
        ui.warn(
            f"{desc}: dropped {n_dropped_empty} sequence(s) with no tokens in span "
            f"'{span}' (typically a trace that never reached `</think>`). They are "
            f"excluded from F, not counted as zeros."
        )
    if not prepped:
        raise RuntimeError(
            f"No valid sequences after tokenization (span={span!r}). If this is a "
            f"thinking-mode corpus, check the responses still contain their "
            f"`<think>…</think>` markup — stripping it leaves nothing to segment."
        )

    # Sort longest-first so each batch pads to a similar length (minimal waste).
    # Counting order is irrelevant — we only accumulate sums.
    prepped.sort(key=lambda p: p[0].shape[0], reverse=True)
    batches = [prepped[i : i + batch_size] for i in range(0, len(prepped), batch_size)]

    # ── 2) GPU-resident accumulators. Sync to CPU only at the end. ──
    # Counts are integers. Keeping them int64 avoids slow FP64 accumulation on
    # consumer GPUs; convert the small final (L,E) result on CPU once at the end.
    counts = torch.zeros(n_layers, n_experts, dtype=torch.int64, device=device)
    total_tokens = torch.zeros((), dtype=torch.int64, device=device)

    # ── 3) One persistent hook manager; one forward per batch. ──
    # Call the BASE transformer (model.model), not the causal-LM wrapper: the router
    # hooks live inside the decoder layers, so we get identical router logits while
    # skipping the lm_head, whose (B, T, vocab) logits tensor is a large VRAM spike
    # we never use. Falls back to the full model if there's no `.model` attribute.
    # Which capture the gate needs. `topk(router_logits)` (Eq. 3 as literally written)
    # is only correct when selection IS a plain top-k over the logits. A DeepSeek-style
    # gate breaks that three ways: a selection-only balancing bias shifts which experts
    # win, node-limited routing masks whole groups out, and the gate may emit no logit
    # tensor at all (some gates return `(weights, indices)` — top-k-ing that would count over
    # `top_k` phantom "experts" rather than the real E). So recompute the selection.
    use_selection = gate_spec is not None and (
        getattr(spec, "router_output", "") == "recompute" or not gate_spec.is_plain_topk
    )

    fwd = getattr(model, "model", model)
    with MoEHookManager(model, spec) as hm:
        if use_selection:
            hm.capture_expert_selection(gate_spec)
        else:
            hm.capture_router_logits()

        for batch in ui.iter_with_progress(batches, desc=desc):
            B = len(batch)
            lens = [ids.shape[0] for ids, _ in batch]
            T_pad = max(lens)

            input_ids = torch.full((B, T_pad), pad_id, dtype=torch.long)
            attn = torch.zeros((B, T_pad), dtype=torch.long)
            count_mask = torch.zeros((B, T_pad), dtype=torch.bool)
            for b, (ids, m) in enumerate(batch):
                L = ids.shape[0]
                input_ids[b, :L] = ids
                attn[b, :L] = 1
                count_mask[b, :L] = m
            input_ids = input_ids.to(device)
            attn = attn.to(device)
            # Per-sequence span mask. Padding is False by construction, and the query
            # is False from `profiling_spans`, so this replaces the old
            # [n_prompt, real_len) arange without changing non-reasoning behaviour.
            count_mask = count_mask.to(device)

            fwd(input_ids=input_ids, attention_mask=attn, use_cache=False)

            if use_selection:
                # Selection recomputed through the gate's real semantics. Hash-routed and
                # dense layers never appear here — they carry no content-driven routing,
                # so they stay at zero and are masked out before expert selection.
                for layer, sel_idx in hm.capture.expert_indices.items():
                    idx = sel_idx.view(B, T_pad, -1)
                    counts[layer] += _expert_membership_counts(idx, count_mask, n_experts)
            else:
                for layer, logits in hm.capture.router_logits.items():
                    E = logits.shape[-1]
                    if E != n_experts:
                        raise RuntimeError(
                            f"layer {layer}: the gate's output has {E} columns, not "
                            f"n_experts={n_experts}. This gate does not emit a router-logit "
                            f"tensor — DeepSeek's returns (weights, indices), so top-k-ing "
                            f"it would count over {E} positions as if they were experts. "
                            f"Pass a `gate_spec` (GateSpec.from_config(cfg.model)) so the "
                            f"selection is recomputed through the real gate; check that the "
                            f"config's `routing:` block and `arch.router_output` are set."
                        )
                    lg = logits.view(B, T_pad, E)  # un-flatten B*T
                    _, idx = lg.topk(top_k, dim=-1)
                    counts[layer] += _expert_membership_counts(idx, count_mask, n_experts)

            total_tokens += count_mask.sum()

    n_resp = int(total_tokens.item())
    if n_resp == 0:
        raise RuntimeError("No response tokens were counted — check your sequences.")
    # Single CPU sync at the very end.
    freq = counts.cpu().to(torch.float64).div_(n_resp)
    return ExpertFreq(freq=freq, n_tokens=n_resp)
