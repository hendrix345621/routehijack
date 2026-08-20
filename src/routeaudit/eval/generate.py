"""Batched model generation helpers."""

from __future__ import annotations

import torch


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
    """Generate token IDs in left-padded batches, excluding prompt and padding IDs."""
    from .. import ui
    from ..model.prompting import render_user_turn, use_template

    if not prompts:
        return []
    device = device or next(model.parameters()).device
    templated = use_template(tokenizer, want_template)
    previous_side = tokenizer.padding_side
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    generation = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": True,
    }
    if do_sample:
        generation["temperature"] = temperature
        if top_p is not None:
            generation["top_p"] = top_p
        if top_k is not None:
            generation["top_k"] = top_k

    stop_ids = {token for token in (tokenizer.eos_token_id, tokenizer.pad_token_id) if token is not None}
    output_ids: list[list[int]] = []
    try:
        chunks = [prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)]
        with ui.progress_bar(len(prompts), desc=desc) as (progress, task_id):
            for chunk in chunks:
                rendered = [
                    render_user_turn(tokenizer, prompt, want_template=want_template) for prompt in chunk
                ]
                encoded = tokenizer(
                    rendered,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=not templated,
                ).to(device)
                output = model.generate(**encoded, **generation)
                generated = output[:, encoded["input_ids"].shape[1] :]
                for row in generated.tolist():
                    end = next((i for i, token in enumerate(row) if token in stop_ids), len(row))
                    output_ids.append(row[:end])
                progress.advance(task_id, len(chunk))
    finally:
        tokenizer.padding_side = previous_side
    return output_ids
