"""Colab / Kaggle runtime helpers.

Free-tier reality: sessions end after ~12 h (or earlier), the VM disk is wiped,
and GPU availability is not guaranteed. So every long job in this repo is
RESUMABLE: it writes results to Drive as it goes and skips files that already
exist. These helpers cover the common plumbing.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict


def in_colab() -> bool:
    return "google.colab" in sys.modules or os.path.exists("/content") and "COLAB_RELEASE_TAG" in os.environ


def in_kaggle() -> bool:
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def mount_drive(mount_point: str = "/content/drive") -> bool:
    """Mount Google Drive (no-op outside Colab). Returns True if mounted."""
    if not in_colab():
        print("Not running in Colab - skipping Drive mount.")
        return False
    from google.colab import drive  # type: ignore

    drive.mount(mount_point)
    return os.path.isdir(f"{mount_point}/MyDrive")


def gpu_info() -> Dict[str, object]:
    """Describe the GPU without importing torch (works before installs)."""
    info: Dict[str, object] = {"available": False}
    if shutil.which("nvidia-smi") is None:
        return info
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip().splitlines()
        if out:
            name, total, used, driver = [x.strip() for x in out[0].split(",")]
            info.update(available=True, name=name, memory_total_mb=int(total), memory_used_mb=int(used), driver=driver)
    except Exception as e:  # pragma: no cover
        info["error"] = str(e)
    return info


def disk_free_gb(path: str = "/") -> float:
    return round(shutil.disk_usage(path).free / 1e9, 1)


def ram_gb() -> float | None:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    return round(int(line.split()[1]) / 1e6, 1)
    except OSError:
        return None
    return None


def check_env() -> Dict[str, object]:
    """Print and return a runtime snapshot. Warns about known pitfalls."""
    report: Dict[str, object] = {
        "python": platform.python_version(),
        "platform": "colab" if in_colab() else "kaggle" if in_kaggle() else "other",
        "gpu": gpu_info(),
        "disk_free_gb": disk_free_gb("/"),
        "ram_gb": ram_gb(),
    }
    try:
        import numpy as np

        report["numpy"] = np.__version__
        ok, msg = numpy_torch_smoke_test()
        report["numpy_torch_ok"] = ok
        if not ok and "ModuleNotFoundError" not in msg:
            report["warning_numpy"] = (
                f"numpy<->torch conversion FAILED ({msg}). Fix: pip install 'numpy<2' then restart the runtime."
            )
    except ImportError:
        report["numpy"] = None
    try:
        import torch

        report["torch"] = torch.__version__
        report["cuda_available"] = torch.cuda.is_available()
    except Exception as e:
        report["torch"] = f"not importable ({type(e).__name__})"

    gpu = report["gpu"]
    print("=== Runtime check ===")
    for k, v in report.items():
        print(f"{k:>16}: {v}")
    if not gpu.get("available"):  # type: ignore[union-attr]
        print("\n!! No GPU. Colab: Runtime > Change runtime type > T4 GPU. "
              "If none is offered, try later or use the Kaggle fallback.")
    return report


def numpy_torch_smoke_test() -> tuple[bool, str]:
    """Functional check for the NumPy-2 / old-torch incompatibility we hit on the DGX.

    That bug did not always raise on import; it silently broke array->tensor
    conversions. So test the conversion itself instead of just comparing versions.
    """
    try:
        import numpy as np
        import torch

        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        t = torch.from_numpy(arr)
        back = t.numpy()
        t2 = torch.tensor(arr)
        if not (np.array_equal(back, arr) and t2.shape == (3, 4)):
            return False, "round-trip mismatch"
        return True, "ok"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def sh(cmd: str, check: bool = True) -> int:
    """Run a shell command, streaming output (used from notebooks)."""
    print(f"$ {cmd}")
    p = subprocess.run(cmd, shell=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {cmd}")
    return p.returncode


def seed_everything(seed: int = 42) -> None:
    """Fix RNGs. Stochastic stages (Stable Diffusion x10) must be reproducible."""
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)
