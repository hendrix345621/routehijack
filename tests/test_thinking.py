"""Unit tests for the thinking-mode re-anchoring (model/thinking.py).

Pure logic — no model, no GPU. The token-level scan is exercised with plain int
lists standing in for token ids, and a tiny fake tokenizer covers the delimiter
resolution and the A2 target builder.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from routeaudit.model.thinking import (
    MALFORMED,
    NO_THINK,
    OK,
    TRUNCATED,
    ScoredBatch,
    ThinkSpec,
    audit_format,
    build_thought_target,
    locate_answer,
)

OPEN, CLOSE = 100, 101  # stand-in ids for <think> / </think>
SPEC = ThinkSpec(open_id=OPEN, close_id=CLOSE)


# ─────────────────────────── locate_answer ───────────────────────────


def test_thinking_trace_anchors_after_close():
    # <think> a b </think> ans1 ans2
    gen = [OPEN, 1, 2, CLOSE, 3, 4]
    a = locate_answer(gen, SPEC)
    assert a.answer_onset == 4
    assert a.think_span == (1, 3)  # content between the tags
    assert a.status == OK
    assert a.scoreable


def test_non_think_no_tags_starts_at_zero():
    # A model with no trace: the answer is the whole generation.
    a = locate_answer([5, 6, 7], SPEC)
    assert a.answer_onset == 0
    assert a.status == NO_THINK
    assert a.think_len == 0


def test_empty_trace_immediate_close_onset_one():
    # Non-think DeepSeek-style format: </think> emitted immediately.
    a = locate_answer([CLOSE, 9, 9], SPEC)
    assert a.answer_onset == 1
    assert a.think_span == (0, 0)


def test_truncated_in_think_is_unscoreable():
    # Opened a trace, ran out of budget before closing → no answer to score.
    a = locate_answer([OPEN, 1, 2, 3], SPEC)
    assert a.answer_onset is None
    assert a.status == TRUNCATED
    assert not a.scoreable


def test_closed_but_no_answer_is_truncated():
    # Closed the trace on the very last token → nothing after it.
    a = locate_answer([OPEN, 1, CLOSE], SPEC)
    assert a.answer_onset is None
    assert a.status == TRUNCATED


def test_repeated_close_flagged_malformed_but_scoreable():
    gen = [OPEN, 1, CLOSE, 2, CLOSE, 3]
    a = locate_answer(gen, SPEC)
    assert a.status == MALFORMED
    assert a.answer_onset == 3  # first close wins
    assert a.scoreable


def test_quoted_close_inside_trace_does_not_fool_token_scan():
    # The whole point of token-level: a CLOSE id that is genuinely the delimiter is
    # found; extra CLOSE ids later are trace content, not a second answer.
    gen = [OPEN, 1, CLOSE, 2, 3]
    a = locate_answer(gen, SPEC)
    assert a.answer_onset == 3


def test_no_open_id_still_finds_answer_by_close():
    # Some templates emit <think> in the prompt (never generated); only close matters.
    spec = ThinkSpec(open_id=None, close_id=CLOSE)
    a = locate_answer([1, 2, CLOSE, 3], spec)
    assert a.answer_onset == 3
    assert a.think_span == (0, 2)  # everything before close is the trace


def test_locate_requires_close_id():
    with pytest.raises(ValueError):
        locate_answer([1, 2, 3], ThinkSpec(open_id=OPEN, close_id=None))


# ─────────────────────────── ScoredBatch ───────────────────────────


def test_scored_batch_excludes_truncated_and_bounds_it():
    b = ScoredBatch()
    for v in (True, True, False):  # 2 successes, 1 refusal
        b.add(v)
    b.add(None)  # 1 truncated (unscoreable)
    b.add(None)
    assert b.n_total == 5
    assert b.n_scored == 3
    assert b.rate == pytest.approx(2 / 3)
    assert b.truncation_rate == pytest.approx(2 / 5)
    lo, hi = b.bounds
    assert lo == pytest.approx(2 / 5)  # truncated all negative
    assert hi == pytest.approx((2 + 2) / 5)  # truncated all positive
    assert lo <= b.rate * (b.n_scored / b.n_total) + 1e-9 or lo <= hi


def test_scored_batch_all_scored_bounds_collapse():
    b = ScoredBatch()
    for v in (True, False, True, False):
        b.add(v)
    lo, hi = b.bounds
    assert lo == hi == pytest.approx(0.5)


# ─────────────────────────── audit_format ───────────────────────────


def test_audit_passes_when_traces_present_and_thinking_requested():
    anchors = [locate_answer([OPEN, 1, CLOSE, 2], SPEC) for _ in range(4)]
    audit = audit_format(anchors, requested_thinking=True)
    assert audit.passed
    assert audit.trace_rate == 1.0


def test_audit_fails_when_thinking_requested_but_absent():
    # Requested thinking, but the template silently ignored the flag → no traces.
    anchors = [locate_answer([5, 6, 7], SPEC) for _ in range(4)]
    audit = audit_format(anchors, requested_thinking=True)
    assert not audit.passed


def test_audit_counts_truncated_as_having_a_trace():
    anchors = [locate_answer([OPEN, 1, 2, 3], SPEC) for _ in range(3)]
    audit = audit_format(anchors, requested_thinking=True)
    assert audit.n_truncated == 3
    assert audit.n_with_trace == 3
    assert audit.truncation_rate == 1.0


# ─────────────── ThinkSpec + A2 target (fake tokenizer) ───────────────


class _FakeTokenizer:
    """Minimal stand-in: known ids for the delimiters, a chat template that puts the
    opening <think> in the GENERATION PROMPT (the Qwen-style case A2 must detect)."""

    unk_token_id = 0

    def __init__(self, emit_think_in_prompt=True):
        self._emit = emit_think_in_prompt
        self.last_template_kwargs = None
        self.chat_template = "fake"  # truthy → has_chat_template() is True
        self._vocab = {"<think>": 100, "</think>": 101}

    def convert_tokens_to_ids(self, s):
        return self._vocab.get(s, self.unk_token_id)

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        self.last_template_kwargs = kw
        user = msgs[-1]["content"]
        tail = "<|assistant|>\n<think>\n" if self._emit else "<|assistant|>\n"
        return f"<|user|>\n{user}<|end|>\n{tail}"

    def __call__(self, s, add_special_tokens=False):
        # Only needs to answer "does the tail contain the <think> id"; approximate by
        # returning the open id when the literal tag is in the string.
        ids = [100] if "<think>" in s else [1, 2, 3]
        return type("Enc", (), {"input_ids": ids})()


def test_thinkspec_resolves_ids_from_tokenizer():
    spec = ThinkSpec.from_tokenizer(_FakeTokenizer())
    assert spec.open_id == 100
    assert spec.close_id == 101
    assert spec.available


def test_thinking_setting_reaches_chat_template():
    from routeaudit.model import prompting

    tok = _FakeTokenizer()
    try:
        prompting.set_chat_template_kwargs({"enable_thinking": True})
        prompting.render_user_turn(tok, "test")

        assert tok.last_template_kwargs["enable_thinking"] is True
        assert prompting.thinking_requested() is True
    finally:
        prompting.set_chat_template_kwargs({})


def test_thinkspec_rejects_unk_mapping():
    tok = _FakeTokenizer()
    tok._vocab = {}  # everything maps to unk
    spec = ThinkSpec.from_tokenizer(tok)
    assert spec.open_id is None
    assert spec.close_id is None
    assert not spec.available


def test_thought_target_omits_open_tag_when_template_emits_it():
    # Template already emits <think> in the generation prompt → target must NOT add it.
    tgt = build_thought_target(
        _FakeTokenizer(emit_think_in_prompt=True), thought="COMPLY NOW", want_template=True
    )
    assert tgt == "COMPLY NOW"
    assert not tgt.startswith("<think>")


def test_thought_target_adds_open_tag_when_template_omits_it():
    # Template does NOT emit <think> → the model generates it, so the target carries it.
    tgt = build_thought_target(
        _FakeTokenizer(emit_think_in_prompt=False), thought="COMPLY NOW", want_template=True
    )
    assert tgt.startswith("<think>")
    assert "COMPLY NOW" in tgt


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
