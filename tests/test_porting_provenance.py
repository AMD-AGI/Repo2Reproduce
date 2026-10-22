"""A hipify_with_fixes port must derive from the pinned official HIPIFY translation."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from repo2reproduce.executor import ManifestExecutionError, validate_porting_files
from repo2reproduce.porting import official_hipify as hipify
from repo2reproduce.porting import provenance
from repo2reproduce.schemas import RocmPortingManifest, RocmPortReport
from tests.test_porting_official_hipify import VECTOR_ADD_WITH_HOST
from tests.test_porting_pipeline import _ported_ready_payload, _script

CUDA_SOURCE = """#include <cuda_runtime.h>
template <int N>
__global__ void __launch_bounds__(256) score_kernel(const int* a, const int* b, float* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int bits = __popc(a[i] ^ b[i]);
  __shared__ float partial[256];
  partial[threadIdx.x] = bits;
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = 0;
    for (int j = 0; j < 256; ++j) total += partial[j];
    out[blockIdx.x] = total;
  }
}
void launch(const int* a, const int* b, float* out, cudaStream_t stream) {
  score_kernel<4><<<1, 256, 0, stream>>>(a, b, out);
}
"""
FIXED_PORT = (
    CUDA_SOURCE.replace("#include <cuda_runtime.h>", "#include <hip/hip_runtime.h>")
    .replace("cudaStream_t", "hipStream_t")
    .replace("float total = 0;", "float total = 0.0f;")
)
REWRITE = """#include <hip/hip_runtime.h>
__global__ void score_kernel(const int* a, const int* b, float* out) {
  out[threadIdx.x] = static_cast<float>(__popc(a[threadIdx.x] ^ b[threadIdx.x]));
}
"""


def _translate_like_hipify(tool: hipify.HipifyTool, source: Path) -> str:
    return source.read_text(encoding="utf-8").replace(
        "#include <cuda_runtime.h>", "#include <hip/hip_runtime.h>"
    ).replace("cudaStream_t", "hipStream_t")


def _run(tmp_path: Path, port_source: str, *, attempts: list[str], strategy: str | None):
    run_dir = tmp_path / "run"
    script = _script(run_dir)
    workspace = run_dir / "workspace"
    for relative, text in (("repo/csrc/tp.cu", CUDA_SOURCE), ("repo-hip/csrc/tp.cu", port_source)):
        (workspace / relative).parent.mkdir(parents=True, exist_ok=True)
        (workspace / relative).write_text(text, encoding="utf-8")
    (workspace / "docker").mkdir(parents=True)
    (workspace / "docker" / "rocm_port.py").write_text("print('patched')\n", encoding="utf-8")
    parity = run_dir / "artifacts" / "parity" / "parity.log"
    parity.parent.mkdir(parents=True)
    parity.write_text("max error 0\n", encoding="utf-8")
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "hipify": {
                    "repository": hipify.HIPIFY_REPOSITORY,
                    "ref": hipify.HIPIFY_REF,
                    "commit": hipify.HIPIFY_COMMIT,
                    "script": "hipify-perl",
                }
            }
        ),
        encoding="utf-8",
    )
    payload = _ported_ready_payload(script)
    port = payload["rocm_port"]
    port["attempts"] = [
        {"strategy": name, "outcome": "succeeded", "evidence": f"{name}: built and verified"}
        for name in attempts
    ]
    port["ported_kernels"][0]["cuda_kernel"] = "score_kernel"
    port["ported_kernels"][0]["strategy"] = strategy
    return run_dir, RocmPortingManifest.model_validate(payload)


@pytest.fixture(autouse=True)
def _fake_translation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provenance, "translate_file", _translate_like_hipify)


def test_hipify_port_with_fixes_passes_and_records_evidence(tmp_path: Path) -> None:
    run_dir, manifest = _run(tmp_path, FIXED_PORT, attempts=["hipify_with_fixes"], strategy=None)

    validate_porting_files(run_dir, manifest)

    summary = json.loads((run_dir / "artifacts" / "hipify-provenance" / "summary.json").read_text())
    kernel = summary["kernels"][0]
    assert summary["hipify"]["commit"] == hipify.HIPIFY_COMMIT
    assert kernel["compared"] == "kernel body" and 0.85 < kernel["overlap"] < 1.0
    diff = (run_dir / "artifacts" / "hipify-provenance" / kernel["diff_path"]).read_text()
    assert "-    float total = 0;" in diff and "+    float total = 0.0f;" in diff


def test_rewrite_labeled_as_hipify_is_rejected(tmp_path: Path) -> None:
    run_dir, manifest = _run(tmp_path, REWRITE, attempts=["hipify_with_fixes"], strategy=None)
    with pytest.raises(ManifestExecutionError, match="record the kernel as targeted_rewrite"):
        validate_porting_files(run_dir, manifest)


def test_labeled_rewrite_skips_the_hipify_comparison(tmp_path: Path) -> None:
    run_dir, manifest = _run(
        tmp_path,
        REWRITE,
        attempts=["hipify_with_fixes", "targeted_rewrite"],
        strategy="targeted_rewrite",
    )
    validate_porting_files(run_dir, manifest)
    assert not (run_dir / "artifacts" / "hipify-provenance").exists()


def test_porting_record_requires_per_kernel_strategy_when_ambiguous(tmp_path: Path) -> None:
    payload = _ported_ready_payload(_script(tmp_path))["rocm_port"]
    payload["attempts"] = [
        {"strategy": name, "outcome": "succeeded", "evidence": f"{name}: built and verified"}
        for name in ("hipify_with_fixes", "targeted_rewrite")
    ]
    with pytest.raises(ValidationError, match="must name its strategy"):
        RocmPortReport.model_validate(payload)
    payload["attempts"] = payload["attempts"][:1]
    payload["ported_kernels"][0]["strategy"] = "targeted_rewrite"
    with pytest.raises(ValidationError, match="has no succeeded attempt"):
        RocmPortReport.model_validate(payload)


def test_function_body_finds_definitions_not_calls() -> None:
    body = provenance.function_body(CUDA_SOURCE, "score_kernel")
    assert body is not None and body.startswith("{") and "__popc" in body
    assert provenance.function_body(CUDA_SOURCE, "missing_kernel") is None


@pytest.mark.skipif(
    not os.environ.get("R2R_TEST_HIPIFY_PERL"),
    reason="set R2R_TEST_HIPIFY_PERL to a rocm-7.2.4 bin/hipify-perl to run the real tool",
)
def test_real_hipify_translation_is_its_own_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.undo()
    tool = hipify.HipifyTool(
        repository=hipify.HIPIFY_REPOSITORY,
        ref=hipify.HIPIFY_REF,
        commit=hipify.HIPIFY_COMMIT,
        script=os.environ["R2R_TEST_HIPIFY_PERL"],
    )
    workspace = tmp_path / "workspace"
    (workspace / "repo").mkdir(parents=True)
    (workspace / "repo" / "vector_add.cu").write_text(VECTOR_ADD_WITH_HOST, encoding="utf-8")
    translated = provenance.translate_file(tool, workspace / "repo" / "vector_add.cu")
    assert "hipMalloc" in translated and "cudaMalloc" not in translated
    (workspace / "port").mkdir()
    (workspace / "port" / "vector_add.hip").write_text(translated, encoding="utf-8")

    result = provenance.check_kernel(
        workspace=workspace,
        cuda_kernel="vectorAdd",
        cuda_source="repo/vector_add.cu",
        port_files=["port/vector_add.hip"],
        tool=tool,
        evidence_dir=tmp_path / "evidence",
    )

    assert result.compared == "kernel body" and result.overlap == 1.0
