"""Regression tests for the simplified configuration, CLI, and reporting paths."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from routeaudit import cli, config, pipeline
from routeaudit.eval import harness
from routeaudit.eval.asr import _last_real_logits
from routeaudit.eval.harness import CellResult
from routeaudit.pipeline import write_results_folder
from routeaudit.ui import _transcript_verdict


def test_profiles_inherit_defaults_and_smoke_limits() -> None:
    smoke = config.load("smoke")

    assert smoke.model.arch.name == "olmoe"
    assert smoke.data.n_pairs == 16
    assert smoke.data.n_general == 32
    assert smoke.attacks.routeaudit.n_steps == 10
    assert smoke.eval.asr.n_prompts == 8

    qwen_smoke = config.load("qwen3-think-smoke")
    assert qwen_smoke.attacks.routeaudit.n_steps == 10
    assert qwen_smoke.data.n_general == 32


def test_analysis_only_router_fails_before_attack() -> None:
    liquid = config.load("liquid")

    assert not config.capabilities(liquid)["attack"]
    with pytest.raises(config.UnsupportedModelError, match="plain softmax top-k"):
        config.require_capability(liquid, "attack")


def test_reasoning_profiles_default_to_complete_answer_evaluation() -> None:
    for profile in ("qwen3", "qwen3.6", "glm4.5-air"):
        cfg = config.load(profile)
        assert cfg.model.enable_thinking is True
        assert cfg.identify.span == "answer"
        assert cfg.eval.mmlu.generative is True

    glm = config.load("glm4.5-air")
    assert (glm.model.n_layers, glm.model.n_experts, glm.model.top_k, glm.model.d_model) == (
        46,
        128,
        8,
        4096,
    )
    assert not config.capabilities(glm)["attack"]


def test_nested_qwen_hf_config_uses_text_model_dimensions() -> None:
    text_cfg = SimpleNamespace(
        num_hidden_layers=40,
        num_experts=256,
        num_experts_per_tok=8,
        hidden_size=2048,
        moe_intermediate_size=512,
    )
    hf_cfg = SimpleNamespace(model_type="qwen3_5_moe", text_config=text_cfg)

    model = config._model_ns_from_hf(hf_cfg, "Qwen/example", dtype="bfloat16", device_map="auto")

    assert (model.n_layers, model.n_experts, model.top_k, model.d_model) == (40, 256, 8, 2048)
    assert model.arch.name == "qwen"
    assert model.enable_thinking is True


def test_unsupported_attack_fails_before_model_load(monkeypatch) -> None:
    monkeypatch.setattr(cli, "load_model", lambda _cfg: pytest.fail("model should not load"))

    with pytest.raises(SystemExit) as exc:
        cli.main(["attack", "--config", "liquid"])

    assert exc.value.code == 2


def test_run_loads_model_once_and_reuses_it(monkeypatch) -> None:
    events = []
    real_load = config.load

    def load_profile(name):
        events.append("config")
        return real_load(name)

    loaded = object()
    monkeypatch.setattr(cli.config, "load", load_profile)
    monkeypatch.setattr(cli, "load_model", lambda _cfg: events.append("model") or loaded)
    monkeypatch.setattr(cli, "harvest_run", lambda got, *_args: events.append(("harvest", got)))
    monkeypatch.setattr(
        cli,
        "attack_run",
        lambda got, *_args: events.append(("attack", got)) or {"suffix_path": "suffix.json"},
    )
    monkeypatch.setattr(cli, "eval_run", lambda got, *_args: events.append(("eval", got)))

    cli.main(["run", "--config", "smoke", "--skip-data"])

    assert events.count("config") == 1
    assert events.count("model") == 1
    assert events[-3:] == [("harvest", loaded), ("attack", loaded), ("eval", loaded)]


def test_judge_reads_last_real_token_in_padded_batch() -> None:
    logits = torch.zeros(3, 4, 2)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1], [0, 0, 1, 1]])
    logits[0, 1] = torch.tensor([1.0, 9.0])
    logits[0, 3] = torch.tensor([9.0, 1.0])
    logits[1, 3] = torch.tensor([8.0, 2.0])
    logits[2, 3] = torch.tensor([3.0, 7.0])

    selected = _last_real_logits(logits, mask)

    assert selected.tolist() == [[1.0, 9.0], [8.0, 2.0], [3.0, 7.0]]


def test_judge_receives_complete_post_thinking_answer(monkeypatch) -> None:
    class Tokenizer:
        unk_token_id = None

        @staticmethod
        def convert_tokens_to_ids(token):
            return {"<think>": 100, "</think>": 101}.get(token, -1)

        @staticmethod
        def decode(ids, **_kwargs):
            return "complete final answer" if list(ids) == [7, 8, 9] else "reasoning complete final answer"

    seen = []
    monkeypatch.setattr(harness, "generate_batch_ids", lambda *_args, **_kwargs: [[100, 4, 101, 7, 8, 9]])
    monkeypatch.setattr(
        harness,
        "score_with_classifier",
        lambda _prompts, responses, **_kwargs: seen.extend(responses) or [False],
    )

    result = harness.run_cell(object(), Tokenizer(), "answer-check", ["prompt"], judge=True, n_show_live=0)

    assert seen == ["complete final answer"]
    assert result.per_prompt[0]["answer"] == "complete final answer"


def test_eval_computes_logprob_mmlu_once(monkeypatch, tmp_path) -> None:
    suffix_path = tmp_path / "suffix.json"
    suffix_path.write_text('{"suffix": " test"}', encoding="utf-8")
    advbench = tmp_path / "advbench.jsonl"
    advbench.write_text('{"prompt": "request"}\n', encoding="utf-8")
    mmlu = tmp_path / "mmlu.jsonl"
    mmlu.write_text('{"question": "q"}\n', encoding="utf-8")
    calls = []
    cells = []

    def run_cell(_model, _tok, name, prompts, **_kwargs):
        return CellResult(
            name=name,
            attack_label=name,
            n_prompts=1,
            asr=0.0,
            per_prompt=[{"completion": "no", "refused": True, "judged_harmful": None}],
        )

    monkeypatch.setattr(pipeline, "load_experts", lambda _path: [])
    monkeypatch.setattr(pipeline, "run_cell", run_cell)
    monkeypatch.setattr(pipeline, "mmlu_logprob_accuracy", lambda *_args, **_kwargs: calls.append(1) or 0.75)
    monkeypatch.setattr(
        pipeline,
        "verdict_table",
        lambda results, **_kwargs: cells.extend(results) or "SAFE",
    )
    monkeypatch.setattr(pipeline, "write_results_folder", lambda *_args, **_kwargs: None)
    loaded = SimpleNamespace(model=object(), tokenizer=object(), spec=object())
    args = SimpleNamespace(
        suffix=str(suffix_path),
        advbench=str(advbench),
        mmlu=str(mmlu),
        judge=False,
        results_dir=str(tmp_path / "results"),
    )

    pipeline.eval_run(loaded, config.load("liquid"), args)

    assert len(calls) == 1
    assert [cell.mmlu_acc for cell in cells] == [0.75, 0.75]


def test_unscoreable_transcript_is_not_called_compliance() -> None:
    assert _transcript_verdict(None) == ("UNSCOREABLE", "yellow")
    assert _transcript_verdict(False) == ("COMPLIED", "red")


def test_results_folder_has_only_summary_and_samples(tmp_path) -> None:
    payload = {
        "model": "test/model",
        "verdict": "SAFE",
        "asr_threshold": 0.5,
        "timestamp": "2026-01-01T00:00:00+00:00",
        "n_prompts": 1,
        "judged": False,
        "cells": [
            {"cell": "clean", "asr": 0.0, "judge_asr": None, "mmlu": 0.5},
            {"cell": "attacked", "asr": 0.0, "judge_asr": None, "mmlu": 0.5},
        ],
        "routing_shift": {"supported": False},
        "mmlu_generative": None,
        "thinking": False,
        "suffix": " test",
        "suffix_len": 5,
        "suffix_source": "suffix.json",
    }
    clean = SimpleNamespace(per_prompt=[{"completion": "no", "refused": True, "judged_harmful": None}])
    attacked = SimpleNamespace(
        per_prompt=[{"completion": "still no", "refused": True, "judged_harmful": None}]
    )

    write_results_folder(str(tmp_path), payload, ["request"], " test", clean, attacked)

    assert {p.name for p in tmp_path.iterdir()} == {"summary.json", "summary.md", "samples.jsonl"}
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["verdict"] == "SAFE"
