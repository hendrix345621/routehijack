"""One-command, auditable DeepSeek-V4 mHC compatibility run.

Runs strict preflight, the free synthetic checks, one real-checkpoint fixture forward,
and CPU-side fixture validation. Every command and environment fact is written to an
artifact so a failed rental still leaves enough evidence to diagnose without guessing.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_ADAPTER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ADAPTER_ROOT / "src"))

import torch
from mhc_preflight import run_preflight
from routeaudit_deepseek_v4.config import CONFIG_PATH
from torch.torch_version import TorchVersion


class _StopRun(Exception):
    """Internal non-error control flow; exit_code/status are already populated."""


def _git(args: list[str]) -> str:
    result = subprocess.run(["git", *args], check=False, capture_output=True, text=True)
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_stage(name: str, command: list[str], log_path: Path) -> dict:
    started = time.monotonic()
    print(f"\n=== {name}: {' '.join(command)} ===", flush=True)
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n=== {name}: {' '.join(command)} ===\n")
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = proc.wait()
    return {
        "name": name,
        "command": command,
        "return_code": return_code,
        "seconds": round(time.monotonic() - started, 3),
    }


def _fixture_summary(path: Path) -> dict:
    with torch.serialization.safe_globals([TorchVersion]):
        fx = torch.load(path, map_location="cpu", weights_only=True)
    residual = fx.get("residual") or {}
    gate = fx.get("gate") or {}
    mhc_maps = fx.get("mhc_maps") or {}
    sites = mhc_maps.get("sites") or {}
    required = {
        "gate": "gate" in fx,
        "official_gate_output": isinstance(gate.get("official_indices"), torch.Tensor)
        and isinstance(gate.get("official_weights"), torch.Tensor),
        "residual": "residual" in fx,
        "four_streams": int(residual.get("n_streams", 0)) == 4,
        "real_mhc_maps": set(sites) == {"attn", "ffn"},
        "hash": "hash" in fx,
        "logits": "logits" in fx,
    }
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "required_captures": required,
        "strict_capture_passed": all(required.values()),
        "fixture_format_version": (fx.get("meta") or {}).get("fixture_format_version", 1),
        "meta": fx.get("meta", {}),
        "gate_layer": gate.get("layer"),
        "gate_same_device_parity": gate.get("same_device_parity"),
        "mhc_layer": mhc_maps.get("layer"),
        "mhc_sites": sorted(sites),
        "residual_layer": residual.get("layer"),
        "residual_shape": list(residual["hidden"].shape) if "hidden" in residual else None,
        "residual_streams": residual.get("n_streams"),
        "hash_layer": (fx.get("hash") or {}).get("layer"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/mhc_real"))
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    preflight_path = args.out_dir / "preflight.json"
    fixture_path = args.out_dir / "v4_flash_fixtures.pt"
    log_path = args.out_dir / "run.log"
    manifest_path = args.out_dir / "manifest.json"
    log_path.write_text("", encoding="utf-8")

    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "config": args.config,
        "python_executable": sys.executable,
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_status": _git(["status", "--short"]),
        "artifacts": {
            "preflight": str(preflight_path),
            "fixture": str(fixture_path),
            "log": str(log_path),
            "manifest": str(manifest_path),
        },
        "stages": [],
    }
    exit_code = 1
    try:
        preflight = run_preflight(args.config, preflight_path)
        manifest["preflight_passed"] = preflight["passed"]
        if not preflight["passed"]:
            manifest["status"] = "preflight_failed"
            exit_code = 2
            raise _StopRun
        if args.preflight_only:
            manifest["status"] = "preflight_passed"
            exit_code = 0
            raise _StopRun

        adapter_root = Path(__file__).resolve().parents[1]
        commands = [
            (
                "synthetic_level_0",
                [sys.executable, str(adapter_root / "tests" / "run_synthetic.py")],
            ),
            (
                "real_fixture_extract",
                [
                    sys.executable,
                    str(adapter_root / "fixtures" / "extract.py"),
                    "--config",
                    args.config,
                    "--out",
                    str(fixture_path),
                ],
            ),
            (
                "fixture_validate",
                [
                    sys.executable,
                    str(adapter_root / "fixtures" / "validate.py"),
                    "--fixtures",
                    str(fixture_path),
                    "--atol",
                    "2e-7",
                    "--require-complete",
                ],
            ),
        ]
        for name, command in commands:
            stage = _run_stage(name, command, log_path)
            manifest["stages"].append(stage)
            if name == "real_fixture_extract" and stage["return_code"] == 0:
                manifest["fixture"] = _fixture_summary(fixture_path)
            if stage["return_code"] != 0:
                manifest["status"] = f"{name}_failed"
                exit_code = stage["return_code"] or 1
                raise _StopRun

        summary = manifest.get("fixture") or _fixture_summary(fixture_path)
        manifest["fixture"] = summary
        if not summary["strict_capture_passed"]:
            manifest["status"] = "fixture_incomplete"
            exit_code = 1
            raise _StopRun
        manifest["status"] = "passed"
        exit_code = 0
    except _StopRun:
        pass
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        exit_code = 130
    except Exception as exc:  # noqa: BLE001
        manifest["status"] = "runner_error"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        exit_code = 1
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"\nmanifest={manifest_path}", flush=True)
        print(f"status={manifest['status']}", flush=True)
        if exit_code:
            print("Do not destroy the instance until run.log and manifest.json are copied off.", flush=True)

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
