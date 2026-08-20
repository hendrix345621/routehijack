"""RouteAudit command-line interface."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

from . import config, ui
from .data import prepare_datasets
from .model.loader import load_model
from .pipeline import attack_run, eval_run, harvest_run


def _data_paths(cfg, data_dir: str) -> None:
    root = Path(data_dir)
    cfg.identify.pairs_path = str(root / "llm_lat_pairs.jsonl")
    cfg.identify.general_corpus_path = str(root / "c4_general.jsonl")


def _common_paths(data_dir: str) -> dict:
    root = Path(data_dir)
    return {
        "advbench": str(root / "advbench.jsonl"),
        "mmlu": str(root / "mmlu_subset.jsonl"),
        "safety": "artifacts/safety_experts.json",
        "harmful": "artifacts/harmful_experts.json",
        "suffix": "artifacts/routeaudit_universal.json",
        "results_dir": "artifacts/results",
    }


def _phase_args(args, **extra):
    values = vars(args).copy()
    values.update(_common_paths(args.data_dir))
    values.update(extra)
    return SimpleNamespace(**values)


def _prepare(cfg, args) -> None:
    data_cfg = cfg.data
    prepare_datasets(
        Path(args.data_dir),
        n_pairs=args.n_pairs or data_cfg.n_pairs,
        n_general=args.n_general or data_cfg.n_general,
        n_mmlu=args.n_mmlu or data_cfg.n_mmlu,
    )


def _load(args, capability: str | None = None):
    cfg = config.load(args.config)
    if capability:
        config.require_capability(cfg, capability)
    _data_paths(cfg, args.data_dir)
    return cfg, load_model(cfg)


def _run(args) -> None:
    cfg = config.load(args.config)
    _data_paths(cfg, args.data_dir)
    if args.stop_after in {"attack", "eval"} and not args.suffix_input:
        config.require_capability(cfg, "attack")
    if not args.skip_data:
        _prepare(cfg, args)
    if args.stop_after == "data":
        return

    loaded = load_model(cfg)
    common = _phase_args(args)
    harvest_run(loaded, cfg, common)
    if args.stop_after == "harvest":
        return

    if args.suffix_input:
        common.suffix = args.suffix_input
    else:
        result = attack_run(loaded, cfg, common)
        common.suffix = result["suffix_path"]
    if args.stop_after == "attack":
        return
    eval_run(loaded, cfg, common)


def _dispatch(args) -> None:
    if args.command == "data":
        cfg = config.load(args.config)
        _prepare(cfg, args)
        return
    if args.command == "run":
        _run(args)
        return

    cfg, loaded = _load(args, capability="attack" if args.command == "attack" else None)
    phase_args = _phase_args(args)
    if args.command == "harvest":
        harvest_run(loaded, cfg, phase_args)
    elif args.command == "attack":
        attack_run(loaded, cfg, phase_args)
    elif args.command == "eval":
        eval_run(loaded, cfg, phase_args)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="base", help="profile, YAML path, or Hugging Face model id")
    parser.add_argument("--data-dir", default="data")


def _add_data(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n-pairs", type=int)
    parser.add_argument("--n-general", type=int)
    parser.add_argument("--n-mmlu", type=int)


def _add_harvest(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--freq-batch-size", type=int, default=16)


def _add_attack(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n-prompts", type=int)
    parser.add_argument("--n-steps", type=int)
    parser.add_argument("--candidates-per-step", type=int, default=128)
    parser.add_argument("--candidate-prompt-subsample", type=int, default=0)
    parser.add_argument("--candidate-batch-size", type=int, default=0)
    parser.add_argument("--grad-batch-size", type=int, default=8)
    parser.add_argument("--early-stop-patience", type=int, default=30)
    parser.add_argument("--auto-batch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grad-checkpointing", action="store_true")
    parser.add_argument("--ascii-suffix", action="store_true")
    parser.add_argument("--checkpoint")


def _add_eval(parser: argparse.ArgumentParser, *, include_n_prompts: bool = True) -> None:
    parser.add_argument("--judge", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--judge-device", default="cuda")
    parser.add_argument("--asr-threshold", type=float, default=0.5)
    if include_n_prompts:
        parser.add_argument("--n-prompts", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--gen-batch-size", type=int, default=8)
    parser.add_argument("--show-samples", type=int, default=3)
    parser.add_argument("--mmlu-batch-size", type=int, default=16)
    parser.add_argument("--mmlu-generative", action="store_true", default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="routeaudit")
    sub = parser.add_subparsers(dest="command", required=True)

    data = sub.add_parser("data", help="download evaluation datasets")
    _add_common(data)
    _add_data(data)

    harvest = sub.add_parser("harvest", help="identify safety and harmful experts")
    _add_common(harvest)
    _add_harvest(harvest)

    attack = sub.add_parser("attack", help="optimize a universal suffix")
    _add_common(attack)
    _add_harvest(attack)
    _add_attack(attack)

    evaluate = sub.add_parser("eval", help="evaluate an existing suffix")
    _add_common(evaluate)
    _add_eval(evaluate)

    run = sub.add_parser("run", help="run the pipeline with one model load")
    _add_common(run)
    _add_data(run)
    _add_harvest(run)
    _add_attack(run)
    _add_eval(run, include_n_prompts=False)
    run.add_argument("--skip-data", action="store_true")
    run.add_argument("--suffix-input", help="evaluate a transferred suffix instead of optimizing one")
    run.add_argument("--stop-after", choices=("data", "harvest", "attack", "eval"), default="eval")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        _dispatch(args)
    except (config.UnsupportedModelError, FileNotFoundError, ValueError) as exc:
        ui.fail(str(exc))
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
