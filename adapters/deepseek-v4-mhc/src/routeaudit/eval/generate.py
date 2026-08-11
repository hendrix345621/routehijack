"""Generation with a pluggable router mutator, via `MoEHookManager`."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from ..model.hooks import MoEHookManager

RouterMutator = Callable[[torch.Tensor, int, int], torch.Tensor]  # (logits, layer, step) -> logits


@dataclass
class DefenseBundle:
    """Holds an optional router mutator, wired into generation via `MoEHookManager`."""

    router_mutator: RouterMutator | None = None


@torch.inference_mode()
def generate_with_defense(
    model,
    tokenizer,
    prompt: str,
    *,
    defense: DefenseBundle | None = None,
    max_new_tokens: int = 128,
    do_sample: bool = False,
    temperature: float = 1.0,
    device: str | torch.device | None = None,
    spec=None,
    want_template: bool = True,
    return_ids: bool = False,
):
    """Returns the completion text, or the generated token ids when `return_ids`.

    Ids are needed in thinking mode: decoding drops the `<think>`/`</think>` special
    tokens, leaving nothing to segment the answer from the trace with.
    """
    from ..model.prompting import encode_prompt

    defense = defense or DefenseBundle()
    device = device or next(model.parameters()).device
    ids = encode_prompt(tokenizer, prompt, want_template=want_template, device=device).unsqueeze(0)

    with MoEHookManager(model, spec) as hm:
        if defense.router_mutator is not None:
            hm.set_router_mutator(defense.router_mutator)

        # Keep generated steps separately. Repeatedly concatenating the full sequence
        # makes long reasoning traces quadratic in copied token ids; with a KV cache
        # the model only needs the previous token after the first step.
        generated: list[torch.Tensor] = []
        past_key_values = None
        for _ in range(max_new_tokens):
            step_input = generated[-1] if past_key_values is not None else ids
            outputs = model(
                input_ids=step_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :]
            if do_sample:
                probs = (next_logits / max(temperature, 1e-5)).softmax(-1)
                next_id = torch.multinomial(probs, 1)
            else:
                next_id = next_logits.argmax(-1, keepdim=True)
            generated.append(next_id)
            hm.advance_step()
            if tokenizer.eos_token_id is not None and int(next_id.item()) == tokenizer.eos_token_id:
                break

    new_ids = torch.cat(generated, dim=-1)[0].tolist() if generated else []
    if new_ids and tokenizer.eos_token_id is not None and new_ids[-1] == tokenizer.eos_token_id:
        new_ids = new_ids[:-1]
    if return_ids:
        return new_ids
    return tokenizer.decode(new_ids, skip_special_tokens=True)


@torch.inference_mode()
def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    *,
    max_new_tokens: int = 128,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float | None = None,
    top_k: int | None = None,
    device: str | torch.device | None = None,
    batch_size: int = 8,
    want_template: bool = True,
    desc: str = "generate",
) -> list[str]:
    """Greedy (or sampled) generation over many prompts, BATCHED for GPU use.

    The per-prompt `generate_with_defense` decodes one sequence at a time (batch-1),
    which starves the GPU — fine when you need router/expert mutators, wasteful for
    plain scoring. This path has no mutators and uses the model's own `generate`
    with **left-padding** (so decoder-only KV-caching and position ids stay correct
    across a padded batch), chunked by `batch_size` so a large prompt set doesn't
    blow up the KV cache. Lower `batch_size` if VRAM-tight; raise it to use more GPU.

    Each prompt is rendered through the chat template (if present) so generation is
    in-distribution for an instruct model. Returns one completion per prompt, in order.

    NOTE: decodes with `skip_special_tokens=True`, which DELETES `<think>`/`</think>`
    when they are special tokens — the text alone can't be segmented afterwards. Use
    `generate_batch_ids` for anything thinking-aware.
    """
    ids = generate_batch_ids(
        model,
        tokenizer,
        prompts,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        device=device,
        batch_size=batch_size,
        want_template=want_template,
        desc=desc,
    )
    return [tokenizer.decode(x, skip_special_tokens=True) for x in ids]


@torch.inference_mode()
def generate_batch_ids(
    model,
    tokenizer,
    prompts: list[str],
    *,
    max_new_tokens: int = 128,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float | None = None,
    top_k: int | None = None,
    device: str | torch.device | None = None,
    batch_size: int = 8,
    want_template: bool = True,
    desc: str = "generate",
) -> list[list[int]]:
    """Same batched generation, returning GENERATED TOKEN IDS per prompt.

    Ids, not text, because thinking-mode segmentation has to happen at the token
    level: `</think>` appears verbatim inside traces as ordinary text, and decoding
    with `skip_special_tokens` erases the real delimiter entirely. Trailing pad/EOS
    is trimmed so a short completion isn't padded out to the batch's longest.
    """
    from .. import ui
    from ..model.prompting import render_user_turn, use_template

    if not prompts:
        return []
    device = device or next(model.parameters()).device
    templated = use_template(tokenizer, want_template)

    prev_side = tokenizer.padding_side
    if tokenizer.pad_token_id is None:  # decoder-only tokenizers often lack a pad token
        tokenizer.pad_token = tokenizer.eos_token  # standard, harmless to leave set
    tokenizer.padding_side = "left"

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": True,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature  # only pass when sampling (avoids a warning)
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
        if top_k is not None:
            gen_kwargs["top_k"] = top_k

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    stop_ids = {i for i in (eos_id, pad_id) if i is not None}

    out_ids: list[list[int]] = []
    try:
        chunks = [prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)]
        with ui.progress_bar(len(prompts), desc=desc) as (prog, tid):
            for chunk in chunks:
                rendered = [render_user_turn(tokenizer, c, want_template=want_template) for c in chunk]
                enc = tokenizer(
                    rendered, return_tensors="pt", padding=True, add_special_tokens=not templated
                ).to(device)
                out = model.generate(**enc, **gen_kwargs)
                new = out[:, enc["input_ids"].shape[1] :]  # drop the (left-padded) prompt
                for row in new.tolist():
                    # Cut at the first EOS/pad: everything after it is batch padding,
                    # and counting it would inflate every think-length statistic.
                    cut = len(row)
                    for j, t in enumerate(row):
                        if t in stop_ids:
                            cut = j
                            break
                    out_ids.append(row[:cut])
                prog.advance(tid, len(chunk))
    finally:
        tokenizer.padding_side = prev_side
    return out_ids
