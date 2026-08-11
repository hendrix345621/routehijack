"""Fail-fast hardware/software preflight for the real DeepSeek-V4 mHC fixture.

This performs no model-weight download. Run it immediately after creating a Vast.ai
instance. Required failures mean the instance is the wrong shape for the pinned,
as-shipped FP4/FP8 checkpoint and should be destroyed before spending download time.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

_ADAPTER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ADAPTER_ROOT / "src"))

import torch
from huggingface_hub import model_info

from routeaudit import config as cfg_mod
from routeaudit_deepseek_v4.config import CONFIG_PATH

GIB = 1024**3
MIN_VRAM_HEADROOM = 20 * GIB
MIN_DISK_HEADROOM = 40 * GIB
MIN_CPU_RAM = 64 * GIB
RECOMMENDED_CPU_RAM = 128 * GIB
MIN_COMPUTE_CAPABILITY = (10, 0)  # native packed FP4 experts require Blackwell
MIN_CUDA = (12, 9)
MIN_TORCH = (2, 9)
MIN_TRANSFORMERS = (5, 14)
DEEPGEMM_KERNEL_API_VERSION = 2


def _version_tuple(value: str | None) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", value or "")[:3])


def _at_least(value: str | None, wanted: tuple[int, ...]) -> bool:
    got = _version_tuple(value)
    return got + (0,) * (len(wanted) - len(got)) >= wanted


def _bytes_label(n: int | None) -> str:
    return "unknown" if n is None else f"{n / GIB:.1f} GiB"


def _system_ram() -> int | None:
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None


def _command(args: list[str]) -> str:
    try:
        result = subprocess.run(args, check=False, capture_output=True, text=True, timeout=15)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    return (result.stdout or result.stderr).strip()


def _nvcc_version() -> str | None:
    text = _command(["nvcc", "--version"])
    match = re.search(r"release\s+(\d+\.\d+)", text)
    return match.group(1) if match else None


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _gpu_inventory() -> list[dict[str, Any]]:
    devices = []
    if not torch.cuda.is_available():
        return devices
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        free, total = torch.cuda.mem_get_info(index)
        devices.append(
            {
                "index": index,
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "compute_capability_number": props.major * 10 + props.minor,
                "total_vram_bytes": int(total),
                "free_vram_bytes": int(free),
            }
        )
    return devices


def _probe_deepgemm() -> tuple[bool, str]:
    if importlib.util.find_spec("kernels") is None:
        return False, "kernels package is not installed"
    try:
        from kernels import get_kernel

        kernel = get_kernel(
            "kernels-community/deep-gemm",
            version=DEEPGEMM_KERNEL_API_VERSION,
        )
        return True, (
            f"loaded API v{DEEPGEMM_KERNEL_API_VERSION} "
            f"{type(kernel).__module__}.{type(kernel).__name__}"
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def run_preflight(config: str, out: Path) -> dict[str, Any]:
    cfg = cfg_mod.load(config)
    model_id = str(cfg.model.hf_id)
    requested_revision = getattr(cfg.model, "revision", None)
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str, *, required: bool = True) -> None:
        checks.append({"name": name, "passed": bool(passed), "required": required, "detail": detail})

    repo_size = None
    resolved_revision = None
    hub_error = None
    try:
        info = model_info(model_id, revision=requested_revision, files_metadata=True)
        resolved_revision = info.sha
        repo_size = sum(int(s.size or 0) for s in info.siblings)
    except Exception as exc:  # noqa: BLE001
        hub_error = f"{type(exc).__name__}: {exc}"

    check("Linux host", sys.platform.startswith("linux"), platform.platform())
    check("CUDA available", torch.cuda.is_available(), f"torch.cuda.is_available={torch.cuda.is_available()}")

    gpus = _gpu_inventory()
    check("GPU visible", bool(gpus), f"{len(gpus)} CUDA device(s)")
    blackwell = bool(gpus) and all(
        gpu["compute_capability_number"] >= MIN_COMPUTE_CAPABILITY[0] * 10 for gpu in gpus
    )
    check(
        "Blackwell SM100+",
        blackwell,
        ", ".join(f"GPU {g['index']} {g['name']} (SM {g['compute_capability']})" for g in gpus) or "no GPUs",
    )

    free_vram = sum(g["free_vram_bytes"] for g in gpus)
    total_vram = sum(g["total_vram_bytes"] for g in gpus)
    required_vram = repo_size + MIN_VRAM_HEADROOM if repo_size is not None else None
    check(
        "free aggregate VRAM",
        required_vram is not None and free_vram >= required_vram,
        f"free {_bytes_label(free_vram)}, total {_bytes_label(total_vram)}, "
        f"required {_bytes_label(required_vram)} (checkpoint + 20 GiB)",
    )

    # Model weights are cached under HF_HOME when it is set. Checking cwd here can
    # reject a valid Vast instance whose small container root has a large attached
    # volume, or (worse) approve the wrong filesystem.
    storage_path = Path(os.environ.get("HF_HOME", Path.cwd())).expanduser().resolve()
    disk_probe_path = storage_path
    while not disk_probe_path.exists() and disk_probe_path != disk_probe_path.parent:
        disk_probe_path = disk_probe_path.parent
    disk = shutil.disk_usage(disk_probe_path)
    required_disk = repo_size + MIN_DISK_HEADROOM if repo_size is not None else None
    check(
        "free disk",
        required_disk is not None and disk.free >= required_disk,
        f"path {storage_path}, free {_bytes_label(disk.free)}, "
        f"required {_bytes_label(required_disk)} (checkpoint + 40 GiB)",
    )

    ram = _system_ram()
    check("CPU RAM minimum", ram is not None and ram >= MIN_CPU_RAM, f"RAM {_bytes_label(ram)}")
    check(
        "CPU RAM recommended",
        ram is not None and ram >= RECOMMENDED_CPU_RAM,
        f"RAM {_bytes_label(ram)}; 128 GiB recommended",
        required=False,
    )

    torch_version = _package_version("torch") or torch.__version__
    transformers_version = _package_version("transformers")
    kernels_version = _package_version("kernels")
    cuda_runtime = torch.version.cuda
    nvcc_version = _nvcc_version()
    check("PyTorch >=2.9", _at_least(torch_version, MIN_TORCH), f"torch={torch_version}")
    check(
        "Transformers >=5.14",
        _at_least(transformers_version, MIN_TRANSFORMERS),
        f"transformers={transformers_version}",
    )
    check("CUDA runtime >=12.9", _at_least(cuda_runtime, MIN_CUDA), f"torch CUDA={cuda_runtime}")
    check("CUDA toolkit nvcc >=12.9", _at_least(nvcc_version, MIN_CUDA), f"nvcc={nvcc_version}")
    check(
        "kernels package",
        importlib.util.find_spec("kernels") is not None,
        f"kernels={kernels_version}",
    )
    deepgemm_ok, deepgemm_detail = _probe_deepgemm()
    check("DeepGEMM kernel load", deepgemm_ok, deepgemm_detail)

    experts_backend = getattr(getattr(cfg.model, "load", None), "experts_implementation", None)
    check("native expert backend", experts_backend == "deepgemm", f"experts_implementation={experts_backend}")
    check("pinned revision", bool(requested_revision), f"requested={requested_revision}")
    check(
        "checkpoint metadata",
        hub_error is None and repo_size is not None,
        hub_error or f"{model_id}@{resolved_revision}, repository {_bytes_label(repo_size)}",
    )
    check(
        "revision resolved exactly",
        bool(requested_revision and resolved_revision == requested_revision),
        f"requested={requested_revision}, resolved={resolved_revision}",
    )

    topology = _command(["nvidia-smi", "topo", "-m"])
    if len(gpus) > 1:
        check(
            "fast multi-GPU link",
            "NV" in topology,
            "NVLink detected" if "NV" in topology else "no NVLink marker; PCIe sharding may be slower",
            required=False,
        )

    required_failures = [c for c in checks if c["required"] and not c["passed"]]
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "passed": not required_failures,
        "config": config,
        "model_id": model_id,
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
        "repository_bytes": repo_size,
        "python": sys.version,
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch_version,
        "transformers": transformers_version,
        "kernels": kernels_version,
        "torch_cuda": cuda_runtime,
        "nvcc": nvcc_version,
        "cpu_ram_bytes": ram,
        "storage_path": str(storage_path),
        "disk_probe_path": str(disk_probe_path),
        "disk_free_bytes": disk.free,
        "gpus": gpus,
        "nvidia_smi": _command(["nvidia-smi"]),
        "topology": topology,
        "checks": checks,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for item in checks:
        marker = "PASS" if item["passed"] else ("FAIL" if item["required"] else "WARN")
        print(f"[{marker}] {item['name']}: {item['detail']}")
    print(f"preflight_json={out}")
    print("PREFLIGHT PASSED" if payload["passed"] else "PREFLIGHT FAILED — do not download the model")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--out", type=Path, default=Path("artifacts/mhc_preflight.json"))
    args = ap.parse_args()
    result = run_preflight(args.config, args.out)
    raise SystemExit(0 if result["passed"] else 2)


if __name__ == "__main__":
    main()
