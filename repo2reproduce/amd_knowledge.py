"""Curated AMD/ROCm porting knowledge.

This replaces Repo2ROCm's static ``plan.txt`` with durable, reusable rules. The rules are
seeded into the persistent memory (scope=global, tag=rocm) so every future run recalls
them, and they compound as new lessons are learned via ``mcp__r2r__remember``.

The prime invariant: **never break the ROCm PyTorch stack**. Any dependency install that
changes torch / torchvision / torchaudio away from the base image's ROCm build, or that
makes ``torch.cuda.is_available()`` False, is a failure.
"""

from __future__ import annotations

from . import memory as mem


# Curated ROCm porting rules. Each becomes a global, rocm-tagged memory entry.
AMD_RULES: list[str] = [
    # --- Prime invariants ---
    "ROCm invariant: after any pip install, torch/torchvision/torchaudio must STILL be "
    "the base image's ROCm build (version string contains 'rocm' and torch.version.hip "
    "is not None). If an install changed them, revert that install.",
    "ROCm invariant: torch.cuda.is_available() must remain True inside the container "
    "(ROCm exposes AMD GPUs through the CUDA API). If it flips to False, the environment "
    "is broken - do not proceed.",
    # --- Base image ---
    "Pick a base image from the rocm/pytorch family (e.g. "
    "rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.9.1) whose bundled torch "
    "and python satisfy the repo's requirements; do NOT pip-install torch on top of it.",
    "Write the Dockerfile FROM line as the EXACT rocm/pytorch:* tag (copy it verbatim); "
    "never add a prefix/rename like 'build_env_rocm/pytorch' - that image does not exist "
    "and the saved Dockerfile would be non-reproducible. Use write_dockerfile (which "
    "validates the FROM line) so the artifact matches the built image.",
    # --- Dependency porting ---
    "On a rocm/pytorch base, DROP pins for torch, torchvision, torchaudio, numpy, ninja, "
    "and packaging from requirements.txt - they are preinstalled; re-pinning them pulls "
    "CUDA/CPU wheels that clobber the ROCm stack.",
    "Prefer binary wheels (pip install --only-binary=:all:) for pandas/numpy on Python "
    "3.12 to avoid slow/failing source builds; keep setuptools<81 so pkg_resources exists.",
    # --- Flash attention ---
    "If the repo imports flash_attn, build the ROCm flash-attention from source with "
    "FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE (Triton-AMD backend covers CDNA and RDNA). "
    "SDPA/eager attention is an acceptable fallback for env verification.",
    "On ROCm, set FLASH_ATTENTION_TRITON_AMD_AUTOTUNE=0 during decoding to avoid the "
    "per-sequence-length Triton autotuning that stalls generation on the first run.",
    # --- Custom CUDA kernels ---
    "For repos with custom CUDA kernels (csrc/*.cu + a build.py), make the build HIP-aware: "
    "guard NVIDIA-only flags (-gencode arch=..., --ptxas-options, --use_fast_math, "
    "-lineinfo, --expt-relaxed-constexpr, --expt-extended-lambda) behind "
    "'if not torch.version.hip'. hipcc rejects those flags.",
    "PyTorch ships torch.utils.hipify (hipify_python) to translate CUDA source to HIP; use "
    "it for extensions that don't already have a ROCm path.",
    # --- Runtime env ---
    "Useful ROCm runtime env: PYTORCH_TUNABLEOP_ENABLED=1 (auto-select rocBLAS/hipBLASLt "
    "GEMMs), ROCM_HOME=/opt/rocm, and set LD_LIBRARY_PATH to include torch/lib and "
    "/opt/rocm*/lib so custom extensions link.",
    # --- Data / models ---
    "Prefer UNGATED HuggingFace models for smoke tests (e.g. Qwen/Qwen2.5-*B-Instruct) - "
    "gated meta-llama repos fail without an HF token.",
    "Newer HF dataset repos (e.g. rajpurkar/squad) use the 'List' feature type; if the "
    "pinned 'datasets' errors with \"Feature type 'List' not found\", upgrade datasets.",
    # --- Verification discipline ---
    "Verify the ROCm environment by (1) torch integrity + GPU visibility, (2) building any "
    "custom kernel and importing it, (3) running the repo's own smoke command to exit 0 - "
    "not by assuming the Dockerfile succeeded.",
    "Emit a MINIMAL Dockerfile (base -> clone -> deterministic patch -> deps -> kernel "
    "build -> optional flash-attn), not an exploratory apply-then-revert sequence.",
    "EXECUTION: each docker_run is a fresh --rm container. NEVER background training "
    "(&/nohup/run_in_background) in one docker_run and poll in another - the first "
    "container exits and kills training, and polls run in empty new containers forever. "
    "Run train+eval SYNCHRONOUSLY in ONE foreground docker_run with a large timeout; "
    "reduce epochs/samples so it fits one call.",
]


def seed_amd_memory() -> dict:
    """Seed the AMD rules into persistent memory. Idempotent (dedups on text)."""
    added = 0
    for rule in AMD_RULES:
        res = mem.add_memory(rule, tags=["rocm", "amd"], scope="global")
        if res.get("added"):
            added += 1
    return {"rules_total": len(AMD_RULES), "newly_added": added}


def amd_knowledge_text() -> str:
    """Return the curated rules as a prompt block (used by recall_amd_knowledge)."""
    lines = ["## AMD/ROCm porting rules (apply when enabling a repo on AMD GPUs)"]
    lines += [f"- {r}" for r in AMD_RULES]
    return "\n".join(lines)
