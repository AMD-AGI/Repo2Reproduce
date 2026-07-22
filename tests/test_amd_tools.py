"""Tests for AMD/ROCm helpers that don't require Docker or a GPU."""

from __future__ import annotations

from pathlib import Path

from repo2reproduce.tools import amd as amd_tools
from repo2reproduce import amd_knowledge, memory as mem


def test_detect_cuda_kernels(tmp_path: Path):
    (tmp_path / "csrc").mkdir()
    (tmp_path / "csrc" / "kernel.cu").write_text("__global__ void k(){}", encoding="utf-8")
    (tmp_path / "csrc" / "build.py").write_text(
        "from torch.utils.cpp_extension import CUDAExtension\n"
        "cc_flag=['-gencode','arch=compute_80,code=sm_80']\n",
        encoding="utf-8",
    )
    res = amd_tools.detect_cuda_kernels(tmp_path)
    assert res["has_custom_kernel"] is True
    assert "csrc/kernel.cu" in res["cuda_sources"]
    assert "csrc/build.py" in res["build_scripts"]


def test_detect_no_kernel(tmp_path: Path):
    (tmp_path / "train.py").write_text("import torch\n", encoding="utf-8")
    res = amd_tools.detect_cuda_kernels(tmp_path)
    assert res["has_custom_kernel"] is False


def test_discover_requirements(tmp_path: Path):
    (tmp_path / "requirements.txt").write_text(
        "torch==2.3.0\ntransformers==4.51.3\n", encoding="utf-8"
    )
    res = amd_tools.discover_requirements(tmp_path)
    assert res["torch_spec"] == "2.3.0"
    assert "requirements.txt" in res["files"]


def test_integrity_logic_clobbered(monkeypatch):
    """verify_rocm_integrity must FAIL when torch is no longer a ROCm build."""
    def fake_probe(image):
        return {
            "ok": True, "torch": "2.11.0", "torchvision": "0.26.0", "torchaudio": "2.11.0",
            "hip": None, "cuda_available": True, "device": "cpu",
        }

    monkeypatch.setattr(amd_tools, "probe_torch", fake_probe)
    res = amd_tools.verify_rocm_integrity("img")
    assert res["verified"] is False
    assert any("not a ROCm build" in p or "hip is None" in p for p in res["problems"])


def test_integrity_logic_ok(monkeypatch):
    def fake_probe(image):
        return {
            "ok": True, "torch": "2.9.1+rocm7.2.4", "torchvision": "0.24.0+rocm7.2.4",
            "torchaudio": "2.9.1+rocm7.2.4", "hip": "7.2", "cuda_available": True,
            "device": "AMD Radeon Graphics",
        }

    monkeypatch.setattr(amd_tools, "probe_torch", fake_probe)
    baseline = {"torch": "2.9.1+rocm7.2.4", "torchvision": "0.24.0+rocm7.2.4",
                "torchaudio": "2.9.1+rocm7.2.4"}
    res = amd_tools.verify_rocm_integrity("img", baseline=baseline)
    assert res["verified"] is True
    assert res["problems"] == []


def test_integrity_drift(monkeypatch):
    """Drift from baseline must fail even if still nominally rocm."""
    def fake_probe(image):
        return {
            "ok": True, "torch": "2.9.1+rocm7.2.4", "torchvision": "0.99+rocm7.2.4",
            "torchaudio": "2.9.1+rocm7.2.4", "hip": "7.2", "cuda_available": True, "device": "x",
        }

    monkeypatch.setattr(amd_tools, "probe_torch", fake_probe)
    baseline = {"torch": "2.9.1+rocm7.2.4", "torchvision": "0.24.0+rocm7.2.4",
                "torchaudio": "2.9.1+rocm7.2.4"}
    res = amd_tools.verify_rocm_integrity("img", baseline=baseline)
    assert res["verified"] is False
    assert any("drifted" in p for p in res["problems"])


def test_validate_dockerfile_repairs_garbled_base():
    bad = "FROM build_env_rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.9.1\nRUN echo hi\n"
    res = amd_tools.validate_dockerfile(bad)
    assert res["valid"] is True
    assert "rocm/pytorch:rocm7.2.4" in res["content"]
    assert "build_env_rocm" not in res["content"]
    assert any("repaired" in w for w in res["warnings"])


def test_validate_dockerfile_flags_non_rocm_base():
    df = "FROM python:3.12-slim\nRUN echo hi\n"
    res = amd_tools.validate_dockerfile(df)
    assert any("not in the rocm/ namespace" in w for w in res["warnings"])


def test_validate_dockerfile_accepts_good_base():
    good = "FROM rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.9.1\nRUN echo hi\n"
    res = amd_tools.validate_dockerfile(good)
    assert res["valid"] is True
    assert res["warnings"] == []
    assert res["content"] == good


def test_seed_amd_memory(tmp_path, monkeypatch):
    monkeypatch.setenv("R2R_MEMORY_PATH", str(tmp_path / "mem.jsonl"))
    res = amd_knowledge.seed_amd_memory()
    assert res["newly_added"] == res["rules_total"]
    # idempotent: second seed adds nothing
    res2 = amd_knowledge.seed_amd_memory()
    assert res2["newly_added"] == 0
    # rocm rules are recallable
    hits = mem.recall(tags=["rocm"])
    assert len(hits) >= 5
