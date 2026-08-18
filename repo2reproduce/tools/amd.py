"""AMD/ROCm enablement tools (the "hands" for Track 2).

These are pure-Python helpers (no Claude) that DO real work and return evidence:
* discover the repo's torch/python needs to pick a base image,
* capture the base image's ROCm torch baseline,
* build/run Docker images with AMD GPU access,
* the integrity guard: verify torch/torchvision/torchaudio stayed ROCm and the GPU is
  visible inside a built image,
* detect custom CUDA kernels,
* persist Dockerfile / patch artifacts.

Execution tools shell out to the host `docker`; they are only exposed to agents when the
run is in --execute mode (see hooks.py / cli.py).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

_DEFAULT_DOCKER_RUN_TIMEOUT = 21600
_DEFAULT_DOCKER_BUILD_TIMEOUT = 7200


def _resolve_timeout(explicit: Optional[int], env_var: str, default: int) -> int:
    """Timeout precedence: explicit argument, environment, built-in default."""
    if explicit is not None:
        return int(explicit)
    value = os.environ.get(env_var)
    if value:
        try:
            return int(value)
        except ValueError:
            pass
    return default


# GPU device flags for `docker run` on an AMD/ROCm host.
ROCM_DOCKER_DEVICE_ARGS = [
    "--device=/dev/kfd",
    "--device=/dev/dri",
    "--group-add",
    "video",
    "--security-opt",
    "seccomp=unconfined",
]

# A small python probe that reports the torch stack + GPU visibility as JSON.
_TORCH_PROBE = (
    "import json,torch;"
    "import importlib;"
    "tv=getattr(importlib.import_module('torchvision'),'__version__',None) "
    "if importlib.util.find_spec('torchvision') else None;"
    "ta=getattr(importlib.import_module('torchaudio'),'__version__',None) "
    "if importlib.util.find_spec('torchaudio') else None;"
    "print(json.dumps({'torch':torch.__version__,'torchvision':tv,'torchaudio':ta,"
    "'hip':getattr(torch.version,'hip',None),'cuda_available':torch.cuda.is_available(),"
    "'device':(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)}))"
)


# --------------------------------------------------------------------------- #
# Base image selection (evidence: the repo's own dependency declarations)
# --------------------------------------------------------------------------- #
def discover_requirements(repo_root: Path) -> dict[str, Any]:
    """Read the repo's dependency declarations to inform base-image choice."""
    findings: dict[str, Any] = {"torch_spec": None, "python_spec": None, "files": {}}
    for name in ["requirements.txt", "setup.py", "setup.cfg", "pyproject.toml", "environment.yml"]:
        p = repo_root / name
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        findings["files"][name] = text[:4000]
        m = re.search(r"torch[=<>!~ ]*([0-9]+\.[0-9]+(?:\.[0-9]+)?)", text)
        if m and not findings["torch_spec"]:
            findings["torch_spec"] = m.group(1)
        pm = re.search(r"python_requires\s*=\s*['\"]([^'\"]+)['\"]", text)
        if pm and not findings["python_spec"]:
            findings["python_spec"] = pm.group(1)
    # README hints
    for name in ["README.md", "README.rst"]:
        p = repo_root / name
        if p.exists():
            text = p.read_text(encoding="utf-8", errors="replace")
            m = re.search(r"[Pp]y[Tt]orch[^\n]*?([0-9]+\.[0-9]+(?:\.[0-9]+)?)", text)
            if m:
                findings.setdefault("readme_torch_hint", m.group(1))
            break
    return findings


# Known-good ROCm PyTorch base images (extend as needed).
ROCM_BASE_IMAGES = [
    "rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.9.1",
    "rocm/pytorch:latest",
]


# --------------------------------------------------------------------------- #
# Custom CUDA kernel detection
# --------------------------------------------------------------------------- #
def detect_cuda_kernels(repo_root: Path) -> dict[str, Any]:
    """Find CUDA kernel sources and their build scripts (grounds kernel migration)."""
    skip = {".git", "node_modules", "__pycache__", ".venv", "venv"}
    cu_files: list[str] = []
    build_files: list[str] = []
    for p in repo_root.rglob("*"):
        if any(part in skip for part in p.parts) or not p.is_file():
            continue
        if p.suffix in {".cu", ".cuh"}:
            cu_files.append(str(p.relative_to(repo_root)))
        if p.name in {"build.py", "setup.py"} and (".cu" in p.read_text("utf-8", "replace")[:20000]
                                                    or "CUDAExtension" in p.read_text("utf-8", "replace")[:20000]
                                                    or "nvcc" in p.read_text("utf-8", "replace")[:20000]
                                                    or "gencode" in p.read_text("utf-8", "replace")[:20000]):
            build_files.append(str(p.relative_to(repo_root)))
    return {
        "has_custom_kernel": bool(cu_files),
        "cuda_sources": cu_files[:50],
        "build_scripts": build_files[:20],
    }


# --------------------------------------------------------------------------- #
# Docker build / run
# --------------------------------------------------------------------------- #
def docker_available() -> bool:
    return shutil.which("docker") is not None


def docker_build(
    context_dir: Path,
    tag: str,
    timeout: Optional[int] = None,
) -> dict[str, Any]:
    """Build an image. Returns exit code + tail of the build log (for error feedback)."""
    if not docker_available():
        return {"ok": False, "error": "docker not found on host"}
    resolved = _resolve_timeout(
        timeout,
        "R2R_DOCKER_BUILD_TIMEOUT",
        _DEFAULT_DOCKER_BUILD_TIMEOUT,
    )
    try:
        proc = subprocess.run(
            ["docker", "build", "-t", tag, "."],
            cwd=str(context_dir),
            capture_output=True,
            text=True,
            timeout=resolved,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "exit_code": 124,
            "timed_out": True,
            "timeout_used": resolved,
            "log_tail": (
                f"docker build exceeded {resolved}s. Raise "
                "R2R_DOCKER_BUILD_TIMEOUT for a legitimately longer build."
            ),
            "tag": tag,
        }
    log = (proc.stdout + "\n" + proc.stderr)
    return {
        "ok": proc.returncode == 0,
        "exit_code": proc.returncode,
        "timed_out": False,
        "timeout_used": resolved,
        "log_tail": log[-4000:],
        "tag": tag,
    }


def docker_run(
    tag: str,
    command: str,
    gpus: bool = True,
    timeout: Optional[int] = None,
    env: Optional[dict[str, str]] = None,
    workdir: Optional[str] = None,
) -> dict[str, Any]:
    """Run a command in the image (with AMD GPU devices by default).

    Forwards the host's GPU-pinning env (``HIP_VISIBLE_DEVICES`` /
    ``ROCR_VISIBLE_DEVICES``) into the container so a scheduler can pin each run to a
    single GPU without any change to the agent's command.
    """
    if not docker_available():
        return {"ok": False, "error": "docker not found on host"}
    resolved = _resolve_timeout(
        timeout,
        "R2R_DOCKER_RUN_TIMEOUT",
        _DEFAULT_DOCKER_RUN_TIMEOUT,
    )
    args = ["docker", "run", "--rm"]
    if gpus:
        args += ROCM_DOCKER_DEVICE_ARGS
    if workdir:
        args += ["-w", workdir]
    merged_env = dict(env or {})
    for pin in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        if pin in os.environ and pin not in merged_env:
            merged_env[pin] = os.environ[pin]
    for k, v in merged_env.items():
        args += ["-e", f"{k}={v}"]
    args += [tag, "bash", "-lc", command]
    run_dir = Path(os.environ.get("R2R_RUN_DIR", "runs/_default"))
    log_dir = run_dir / "amd_run" / "docker_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"docker_run_{time.strftime('%Y%m%d-%H%M%S')}.log"

    timed_out = False
    with log_path.open("w", encoding="utf-8") as log_file:
        try:
            proc = subprocess.run(
                args,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=resolved,
            )
            return_code = proc.returncode
        except subprocess.TimeoutExpired as exc:
            return_code = 124
            timed_out = True
            stdout = (
                exc.stdout.decode()
                if isinstance(exc.stdout, bytes)
                else (exc.stdout or "")
            )
            stderr = (
                exc.stderr.decode()
                if isinstance(exc.stderr, bytes)
                else (exc.stderr or "")
            )
            if stdout or stderr:
                log_file.write(stdout + "\n" + stderr + "\n")
            log_file.write(
                f"\n[r2r] docker_run timed out after {resolved}s. Increase the "
                "timeout after measuring a pilot ETA rather than reducing scope.\n"
            )

    output = log_path.read_text(encoding="utf-8", errors="replace")
    return {
        "ok": return_code == 0 and not timed_out,
        "timed_out": timed_out,
        "timeout_seconds": resolved,
        "timeout_used": resolved,
        "exit_code": return_code,
        "log_file": str(log_path),
        "output": output[-8000:],
        **(
            {
                "error": (
                    f"docker_run timed out after {resolved}s; full partial output "
                    f"is preserved at {log_path}. Increase the timeout after "
                    "measuring a pilot ETA rather than reducing scope."
                )
            }
            if timed_out
            else {}
        ),
    }


def probe_torch(image_or_base: str) -> dict[str, Any]:
    """Run the torch probe in an image and parse the JSON line."""
    res = docker_run(image_or_base, f"python -c \"{_TORCH_PROBE}\"", gpus=True, timeout=600)
    if not res.get("ok"):
        return {"ok": False, "error": res.get("output") or res.get("error"), "raw": res}
    # Extract the last JSON object from output.
    for line in reversed(res["output"].splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                data = json.loads(line)
                return {"ok": True, **data}
            except json.JSONDecodeError:
                continue
    return {"ok": False, "error": "could not parse torch probe output", "raw": res["output"]}


def verify_rocm_integrity(image: str, baseline: Optional[dict] = None) -> dict[str, Any]:
    """THE GUARD. Confirm the ROCm torch stack is intact and the GPU is visible.

    Fails if: GPU not available, torch/vision/audio not a ROCm build, or (when a baseline
    is provided) any of the three version strings drifted from the base image.
    """
    probe = probe_torch(image)
    if not probe.get("ok"):
        return {"verified": False, "reason": "torch probe failed", "detail": probe}

    problems: list[str] = []
    if not probe.get("cuda_available"):
        problems.append("torch.cuda.is_available() is False (AMD GPU not visible)")
    if not probe.get("hip"):
        problems.append("torch.version.hip is None (not a ROCm torch build)")
    for key in ("torch", "torchvision", "torchaudio"):
        v = probe.get(key)
        if v is not None and "rocm" not in str(v).lower():
            problems.append(f"{key} version '{v}' is not a ROCm build")
    if baseline:
        for key in ("torch", "torchvision", "torchaudio"):
            b, c = baseline.get(key), probe.get(key)
            if b and c and b != c:
                problems.append(f"{key} drifted from baseline: {b} -> {c}")

    return {
        "verified": not problems,
        "problems": problems,
        "observed": {k: probe.get(k) for k in ("torch", "torchvision", "torchaudio", "hip",
                                               "cuda_available", "device")},
    }


# --------------------------------------------------------------------------- #
# Dockerfile validation (guard against garbled base-image names)
# --------------------------------------------------------------------------- #
# A valid ROCm base image lives in the `rocm/` (or docker.io/rocm/) namespace.
_FROM_RE = re.compile(r"^\s*FROM\s+(?P<image>\S+)", re.IGNORECASE | re.MULTILINE)
_GARBLED_ROCM_RE = re.compile(r"[A-Za-z0-9_]+_rocm/pytorch", re.IGNORECASE)


def validate_dockerfile(content: str) -> dict[str, Any]:
    """Sanity-check a Dockerfile's FROM line and auto-repair known garble.

    Catches cases like ``FROM build_env_rocm/pytorch:...`` (a hallucinated prefix on the
    real ``rocm/pytorch`` image) and normalizes them. Returns the (possibly fixed) content
    plus warnings so the caller/agent knows what changed.
    """
    warnings: list[str] = []
    froms = _FROM_RE.findall(content)
    if not froms:
        return {"content": content, "warnings": ["no FROM line found"], "valid": False}

    fixed = content
    for image in froms:
        # Repair a garbled rocm/pytorch reference (e.g. build_env_rocm/pytorch:tag).
        if _GARBLED_ROCM_RE.search(image):
            repaired = _GARBLED_ROCM_RE.sub("rocm/pytorch", image)
            fixed = fixed.replace(image, repaired)
            warnings.append(f"repaired garbled base image '{image}' -> '{repaired}'")
            image = repaired
        # A ROCm reproduction should build on a rocm/* base; flag anything else.
        norm = image.split("/", 1)[-1] if image.startswith("docker.io/") else image
        if not norm.startswith("rocm/"):
            warnings.append(
                f"base image '{image}' is not in the rocm/ namespace - a ROCm run should "
                f"use a rocm/pytorch:* base (e.g. {ROCM_BASE_IMAGES[0]})"
            )
    return {"content": fixed, "warnings": warnings, "valid": True}


# --------------------------------------------------------------------------- #
# Artifact writers
# --------------------------------------------------------------------------- #
def write_text_artifact(path: Path, content: str) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return {"path": str(path), "bytes": len(content.encode("utf-8"))}
