"""Chat-template-aware prompt assembly, shared by expert profiling, the
RouteAudit optimizer, and generation.

The RouteHijack paper (arXiv 2605.02946) targets the pre-truncation router distribution at
the **boundary token** t* — "the last input token before autoregressive decoding
begins" (§4.2). For an instruction-tuned model that boundary is the final token of
the chat template's assistant-generation prompt, NOT the last token of the raw
query. Running the model without its chat template puts t* in the wrong place and
drives generation off-distribution, so every stage must render prompts through the
template. When the tokenizer has no chat template (a base model) we fall back to
raw text and the boundary collapses to the last query/suffix token.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

# Rare sentinel (RECORD SEPARATOR) used to locate where the adversarial suffix
# sits inside the rendered user turn. Must not occur in real prompts.
_SUFFIX_SLOT = "␞"


def has_chat_template(tokenizer) -> bool:
    return getattr(tokenizer, "chat_template", None) is not None


def use_template(tokenizer, want: bool) -> bool:
    return bool(want) and has_chat_template(tokenizer)


# Extra kwargs forwarded to `apply_chat_template` for every render in this process.
# Set once (e.g. by the loader from `model.enable_thinking`).
#
# {'enable_thinking': False} on a REASONING model (Qwen3 family) makes t* the real
# answer-decision token: with thinking ON, t* is the start of a chain-of-thought, so
# expert localization, L_refusal, and the refusal detector all measure the preamble
# instead of the answer. Running with thinking ON is supported — but only through
# `model/thinking.py`, which re-anchors every one of those to the post-`</think>`
# answer span. Flipping this flag alone is NOT enough.
_CHAT_TEMPLATE_KWARGS: dict = {}
_CHAT_KW_OK = True
_CHAT_KW_REJECTED: dict = {}


def set_chat_template_kwargs(kw: dict | None) -> None:
    global _CHAT_KW_OK
    _CHAT_TEMPLATE_KWARGS.clear()
    _CHAT_TEMPLATE_KWARGS.update(kw or {})
    _CHAT_KW_OK = True
    _CHAT_KW_REJECTED.clear()


def chat_template_kwargs() -> dict:
    """The kwargs actually in force (empty once a template has rejected them)."""
    return dict(_CHAT_TEMPLATE_KWARGS) if _CHAT_KW_OK else {}


def thinking_requested() -> bool:
    """True when this process asked the template for chain-of-thought.

    Only ever the REQUEST. Whether the model actually produced a trace is a property
    of the generation — see `thinking.audit_format`, which is the source of truth.
    """
    return bool(_CHAT_TEMPLATE_KWARGS.get("enable_thinking", False)) and _CHAT_KW_OK


def chat_kwargs_rejected() -> dict:
    """Kwargs this tokenizer's template refused, if any (empty = none)."""
    return dict(_CHAT_KW_REJECTED)


def render_user_turn(tokenizer, content: str, *, want_template: bool = True) -> str:
    """Render a single user turn as the string actually fed to the model, with the
    assistant generation prompt appended. Raw `content` if no template."""
    if not use_template(tokenizer, want_template):
        return content
    global _CHAT_KW_OK
    msgs = [{"role": "user", "content": content}]
    if _CHAT_TEMPLATE_KWARGS and _CHAT_KW_OK:
        try:
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, **_CHAT_TEMPLATE_KWARGS)
        except TypeError:
            # This template ignores the kwargs. Falling back SILENTLY is how a run
            # ends up in the opposite mode from its config for its entire lifetime —
            # so latch it loudly and keep the record for the results payload.
            _CHAT_KW_OK = False
            _CHAT_KW_REJECTED.update(_CHAT_TEMPLATE_KWARGS)
            from .. import ui
            ui.warn(f"chat template rejected {sorted(_CHAT_TEMPLATE_KWARGS)} — rendering WITHOUT "
                    f"them for the rest of this process. If `enable_thinking` is among them, the "
                    f"model is now in its DEFAULT mode, not the configured one. Verify with the "
                    f"format audit before trusting any result.")
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def generation_prompt_tail(tokenizer, *, want_template: bool = True) -> str:
    """The template text appended after user content to start the assistant turn.

    Thinking-mode templates differ in whether they emit the opening `<think>` here
    (so the model never generates it) or leave the model to produce it. That choice
    moves t* by the length of the tag, and decides whether an A2-style thought target
    must carry its own `<think>` prefix. Recovered by rendering a sentinel rather than
    by assuming a family's convention.
    """
    if not use_template(tokenizer, want_template):
        return ""
    rendered = render_user_turn(tokenizer, _SUFFIX_SLOT, want_template=True)
    _, _, tail = rendered.partition(_SUFFIX_SLOT)
    return tail


def encode_prompt(tokenizer, content: str, *, want_template: bool = True,
                  device=None) -> torch.Tensor:
    """Token ids for a full user turn (template + generation prompt, or raw)."""
    templated = use_template(tokenizer, want_template)
    s = render_user_turn(tokenizer, content, want_template=want_template)
    # The template already injects BOS / special tokens; don't double them.
    ids = tokenizer(s, add_special_tokens=not templated).input_ids
    out = torch.tensor(ids, dtype=torch.long)
    return out.to(device) if device is not None else out


def profiling_ids(tokenizer, query: str, response: str, *, want_template: bool = True):
    """Token ids + response span for response-driven expert profiling (Eq. 3).

    Returns (full_ids, n_context) where positions [n_context, len) are the response
    tokens to COUNT and [0, n_context) is the query + chat-template special tokens
    to MASK (RouteHijack paper, §5.1). The response is appended without special tokens so
    only its content tokens are counted."""
    ctx = encode_prompt(tokenizer, query, want_template=want_template)        # query + assistant marker
    resp = torch.tensor(tokenizer(response, add_special_tokens=False).input_ids, dtype=torch.long)
    full = torch.cat([ctx, resp])
    return full, int(ctx.shape[0])


@dataclass
class ProfilingSample:
    """One profiling sequence with its span coordinates, all in FULL-ids indices.

    Carrying the coordinates (not just the mask) is what lets a caller truncate a
    long trace *without destroying the answer* — see `compute_expert_freq`, which
    keeps the trace tail adjacent to `</think>` rather than the head.
    """

    ids: torch.Tensor              # (L,) full sequence: template + query + response
    mask: torch.Tensor             # (L,) bool — positions to COUNT
    n_ctx: int                     # response starts here
    think_span: tuple[int, int]    # [start, end) of trace tokens; empty if none
    answer_onset: int | None       # first answer token, or None if never closed

    @property
    def think_len(self) -> int:
        return max(0, self.think_span[1] - self.think_span[0])


def profiling_spans(tokenizer, query: str, response: str, *, span: str = "answer",
                    want_template: bool = True) -> "ProfilingSample":
    """Token ids + a BOOLEAN MASK over the response, for thinking-aware profiling.

    `profiling_ids` returns a single split point, which is all a non-reasoning
    response needs: everything after the query counts. A reasoning response has three
    populations — trace, delimiter, answer — and pooling them is exactly the error
    that makes localization measure deliberation instead of decision. Trace tokens
    dominate by count, so on a long trace the answer can be <1% of the response and
    the "response-driven" frequency is really a trace-driven one.

    `span`: "answer" (post-`</think>`, the decision), "think" (the deliberation, for
    A2-style thought targeting), "delimiter" (the `</think>` region where refusal
    intent collapses), or "all" (old pooled behaviour).

    Returns a `ProfilingSample` whose mask is False over the query and over every
    response token outside the requested span. A response with no trace yields the
    same mask "answer" and "all" would, so non-reasoning corpora flow through
    unchanged and this can replace `profiling_ids` everywhere.
    """
    from .thinking import ThinkSpec, locate_answer, locate_answer_text

    ctx = encode_prompt(tokenizer, query, want_template=want_template)
    resp_list = tokenizer(response, add_special_tokens=False).input_ids
    resp = torch.tensor(resp_list, dtype=torch.long)
    full = torch.cat([ctx, resp])
    n_ctx, n_resp = int(ctx.shape[0]), int(resp.shape[0])

    spec = ThinkSpec.from_tokenizer(tokenizer)
    if spec.available:
        anchor = locate_answer(resp_list, spec)
        t_start, t_end = anchor.think_span
        onset = anchor.answer_onset
    else:
        # No single-token delimiter: fall back to text and re-tokenize the parts.
        # Approximate at the seam by construction, so it is used only when it must be.
        think_text, answer_text, _ = locate_answer_text(response)
        t_start = 0
        t_end = len(tokenizer(think_text, add_special_tokens=False).input_ids) if think_text else 0
        onset = (n_resp - len(tokenizer(answer_text, add_special_tokens=False).input_ids)
                 if answer_text else None)

    think_span = (n_ctx + t_start, n_ctx + t_end) if t_end > t_start else (n_ctx, n_ctx)
    answer_onset = (n_ctx + onset) if onset is not None else None

    mask = torch.zeros(full.shape[0], dtype=torch.bool)
    if span == "all":
        mask[n_ctx:] = True
    elif span == "think":
        mask[think_span[0]:think_span[1]] = True
    elif span == "delimiter":
        # The close tag plus the few tokens around it — a short, fixed-position window
        # rather than a span, so it needs far more examples to stabilise than the
        # others (three positions per sample against a 256-expert top-8 gate). This is
        # where refusal intent collapses, which is why it is worth profiling at all.
        if t_end > 0:
            mask[max(n_ctx, think_span[1] - 1): min(full.shape[0], think_span[1] + 2)] = True
    elif span == "answer":
        if answer_onset is not None:
            mask[answer_onset:] = True
    else:
        raise ValueError(f"span={span!r} not in ('answer','think','delimiter','all')")
    return ProfilingSample(ids=full, mask=mask, n_ctx=n_ctx,
                           think_span=think_span, answer_onset=answer_onset)


def suffix_slot_ids(tokenizer, query: str, *, want_template: bool = True, device=None):
    """Return (before_ids, after_ids) bracketing the adversarial suffix so the full
    input is `before_ids ++ <suffix> ++ after_ids` with the suffix at the end of the
    user content, just before the assistant generation prompt. The boundary token
    (last of `after_ids`) is the routing decision point t* (§4.2).

    Without a chat template the suffix sits at the very end (`after_ids` empty) and
    t* is the last suffix token — the original behavior."""
    if use_template(tokenizer, want_template):
        rendered = render_user_turn(tokenizer, f"{query} {_SUFFIX_SLOT}", want_template=True)
        left, _, right = rendered.partition(_SUFFIX_SLOT)
        before = tokenizer(left, add_special_tokens=False).input_ids
        after = tokenizer(right, add_special_tokens=False).input_ids
    else:
        before = tokenizer(query, add_special_tokens=True).input_ids
        after = []
    b = torch.tensor(before, dtype=torch.long)
    a = torch.tensor(after, dtype=torch.long)
    if device is not None:
        b, a = b.to(device), a.to(device)
    return b, a
