"""MMLU multiple-choice accuracy.

Used to confirm that the attack doesn't tank utility — so a maintainer can tell
a real vulnerability (harm elicited, utility intact) from a model that was
merely broken by the attack.

Two paths:
  - `mmlu_logprob_accuracy`  : one forward, read the A/B/C/D letter logit at the last
                               prompt token. Cheap. But it uses a RAW prompt (no chat
                               template) and does NOT generate, so it CANNOT observe a
                               reasoning model's thinking — a think-mode model scored
                               this way is measured as if thinking were off.
  - `mmlu_generative_accuracy` : generate in the model's actual mode (thinking on/off),
                               segment the answer past `</think>`, and read the letter
                               from the ANSWER. ~50-100× costlier; the only path that
                               verifies "reasoning still works" under thinking mode.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

import torch

from ..model.hooks import MoEHookManager
from .generate import DefenseBundle

PROMPT_TEMPLATE = "Question: {q}\nA) {a}\nB) {b}\nC) {c}\nD) {d}\nAnswer:"
# Generative prompt asks for a reasoned answer ending in a parseable choice line.
GEN_TEMPLATE = ("{q}\nA) {a}\nB) {b}\nC) {c}\nD) {d}\n\n"
                "Answer with the single letter of the correct choice.")
_LETTER_RE = re.compile(r"\b([ABCD])\b")


@torch.no_grad()
def mmlu_logprob_accuracy(
    model,
    tokenizer,
    questions: Iterable[dict],
    *,
    defense: DefenseBundle = DefenseBundle(),
    device=None,
    spec=None,
    batch_size: int = 16,
) -> float:
    """`questions` items: {question, choices: [4 strings], answer: int 0..3}.

    Batched for GPU utilisation: questions are RIGHT-padded and run in chunks, and
    we read each row's last *real* token (index = attention_mask.sum-1). With right
    padding the real tokens keep positions 0..L-1, so this is numerically identical
    to scoring each question alone — just far fewer forward launches.
    """
    device = device or next(model.parameters()).device
    letter_ids = [tokenizer(" " + L, add_special_tokens=False).input_ids[-1] for L in "ABCD"]
    letter_ids_t = torch.tensor(letter_ids, device=device)

    qs = list(questions)
    prompts = [PROMPT_TEMPLATE.format(q=q["question"], a=q["choices"][0], b=q["choices"][1],
                                      c=q["choices"][2], d=q["choices"][3]) for q in qs]
    answers = [int(q["answer"]) for q in qs]

    prev_side = tokenizer.padding_side
    if tokenizer.pad_token_id is None:                 # decoder-only tokenizers often lack a pad token
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    correct = 0
    try:
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i:i + batch_size]
            enc = tokenizer(chunk, return_tensors="pt", padding=True).to(device)
            last_idx = enc["attention_mask"].sum(dim=1) - 1            # (B,) last real token per row

            with MoEHookManager(model, spec) as hm:
                if defense.router_mutator is not None:
                    hm.set_router_mutator(defense.router_mutator)
                logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                               use_cache=False).logits                 # (B, T, vocab)

            rows = logits[torch.arange(logits.shape[0], device=device), last_idx]   # (B, vocab)
            preds = rows[:, letter_ids_t].argmax(dim=-1)               # (B,) in 0..3
            for j, pred in enumerate(preds.tolist()):
                correct += int(pred == answers[i + j])
    finally:
        tokenizer.padding_side = prev_side
    return correct / max(1, len(prompts))


@torch.no_grad()
def mmlu_generative_accuracy(
    model,
    tokenizer,
    questions: Iterable[dict],
    *,
    suffix: str = "",
    max_new_tokens: int = 1024,
    device=None,
    batch_size: int = 8,
    want_template: bool = True,
) -> dict:
    """Generative MMLU in the model's actual mode, scored on the ANSWER span.

    Renders each question through the chat template (so thinking mode engages),
    generates, segments off any `<think>…</think>` trace, and reads the choice letter
    from the answer. This is the utility measurement that survives thinking mode — the
    log-prob path cannot, because it never lets the model think.

    `suffix`: appended to each question (before the answer instruction) to measure
    utility WITH the deployed jailbreak suffix present — the honest "does shipping this
    suffix wreck benign use" number, which the log-prob path never captures because it
    ignores the suffix entirely.

    Returns a dict: accuracy over PARSEABLE answers, plus the count that produced no
    letter (a truncated trace or a non-committal answer) so the accuracy isn't silently
    computed over a biased subset.
    """
    from .generate import generate_batch_ids
    from ..model.thinking import ThinkSpec, locate_answer

    qs = list(questions)
    if not qs:
        return {"accuracy": None, "n": 0, "n_parsed": 0, "n_unparsed": 0}
    answers = [int(q["answer"]) for q in qs]
    suf = f" {suffix}" if suffix else ""
    prompts = [GEN_TEMPLATE.format(q=q["question"], a=q["choices"][0], b=q["choices"][1],
                                   c=q["choices"][2], d=q["choices"][3]) + suf for q in qs]

    gen_ids = generate_batch_ids(model, tokenizer, prompts, max_new_tokens=max_new_tokens,
                                 batch_size=batch_size, want_template=want_template,
                                 device=device, desc="mmlu-gen")
    spec = ThinkSpec.from_tokenizer(tokenizer)
    correct = n_parsed = n_unparsed = 0
    for ids, gold in zip(gen_ids, answers):
        if spec.available:
            anchor = locate_answer(ids, spec)
            ans_ids = ids[anchor.answer_onset:] if anchor.scoreable else []
        else:
            ans_ids = ids
        text = tokenizer.decode(ans_ids, skip_special_tokens=True) if len(ans_ids) else ""
        # Take the LAST standalone letter — a reasoned answer states its choice last.
        matches = _LETTER_RE.findall(text.upper())
        if not matches:
            n_unparsed += 1
            continue
        n_parsed += 1
        correct += int("ABCD".index(matches[-1]) == gold)
    return {"accuracy": (correct / n_parsed) if n_parsed else None,
            "n": len(qs), "n_parsed": n_parsed, "n_unparsed": n_unparsed}
