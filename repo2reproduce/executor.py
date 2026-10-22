"""Deterministic, blocking execution of a validated reproduction manifest."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import time
from pathlib import Path

from .porting import provenance
from .porting.official_hipify import HipifyTool
from .schemas import (
    ExecutionManifest,
    ExecutionResult,
    ForcedKernelPortingManifest,
    PortedKernel,
    RocmPortingManifest,
    RocmPortReport,
)

_KERNEL_DEFINITION = {
    "hip": re.compile(r"__global__"),
    "triton": re.compile(r"@triton\.(?:jit|autotune)"),
    "composable_kernel": re.compile(r"\bck(?:_tile)?::|#\s*include\s*[<\"]ck(?:_tile)?/"),
    "rocwmma": re.compile(r"\brocwmma::"),
}
_ASSIGNMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
_VARIABLE_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")


class ManifestExecutionError(RuntimeError):
    """Raised when the manifest cannot safely be executed."""


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _inside(root: Path, relative: str, *, must_exist: bool = False) -> Path:
    if Path(relative).is_absolute():
        raise ManifestExecutionError(f"absolute path is not allowed: {relative}")
    root_resolved = root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise ManifestExecutionError(f"path escapes {root}: {relative}") from exc
    if must_exist and not candidate.exists():
        raise ManifestExecutionError(f"required path does not exist: {candidate}")
    return candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _stop_labeled_containers(run_id: str) -> None:
    """Best-effort cleanup when the attached Docker client is interrupted."""

    try:
        query = subprocess.run(
            [
                "/usr/bin/docker",
                "ps",
                "-q",
                "--filter",
                f"label=claude_repro_run={run_id}",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        container_ids = [line for line in query.stdout.splitlines() if line.strip()]
        if container_ids:
            subprocess.run(
                ["/usr/bin/docker", "stop", "--time", "30", *container_ids],
                capture_output=True,
                timeout=60,
                check=False,
            )
    except (OSError, subprocess.SubprocessError):
        pass


def validate_porting_files(run_dir: Path, manifest: RocmPortingManifest) -> None:
    """Check that the patches and parity log named by a porting manifest exist on the host."""

    port = manifest.rocm_port
    workspace = run_dir / "workspace"
    for relative in port.patches:
        patch = _inside(workspace, relative, must_exist=True)
        if not patch.is_file():
            raise ManifestExecutionError(f"porting patch must be a regular file: {relative}")
    for kernel in port.ported_kernels:
        _validate_ported_kernel(workspace, kernel)
    _validate_hipify_provenance(run_dir, port)
    if port.parity_check is not None:
        parity_log = _inside(run_dir / "artifacts", port.parity_check.log_path, must_exist=True)
        if not parity_log.is_file() or parity_log.stat().st_size == 0:
            raise ManifestExecutionError("parity-check log must be a non-empty regular file")
    if isinstance(manifest, ForcedKernelPortingManifest):
        for name in ("port-scan.json", "port-hipify.json"):
            report = _inside(run_dir / "artifacts", name, must_exist=True)
            if not report.is_file() or report.stat().st_size == 0:
                raise ManifestExecutionError(
                    f"forced kernel port requires a non-empty {name} report"
                )


def _validate_hipify_provenance(run_dir: Path, port: RocmPortReport) -> None:
    """Require each ``hipify_with_fixes`` kernel to derive from the pinned HIPIFY output."""

    kernels = [
        kernel for kernel in port.ported_kernels
        if port.kernel_strategy(kernel) == "hipify_with_fixes"
    ]
    if not kernels:
        return
    try:
        recorded = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))["hipify"]
        tool = HipifyTool(**recorded)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ManifestExecutionError(
            "hipify_with_fixes ports need the run's pinned HIPIFY record in metadata.json"
        ) from exc
    evidence_dir = run_dir / "artifacts" / "hipify-provenance"
    results = []
    for kernel in kernels:
        _inside(run_dir / "workspace", kernel.cuda_source, must_exist=True)
        try:
            result = provenance.check_kernel(
                workspace=run_dir / "workspace",
                cuda_kernel=kernel.cuda_kernel,
                cuda_source=kernel.cuda_source,
                port_files=kernel.port_files,
                tool=tool,
                evidence_dir=evidence_dir,
            )
        except (provenance.ProvenanceError, OSError, subprocess.SubprocessError) as exc:
            raise ManifestExecutionError(str(exc)) from exc
        results.append(result.to_dict())
    (evidence_dir / "summary.json").write_text(
        json.dumps(
            {
                "hipify": {"repository": tool.repository, "ref": tool.ref, "commit": tool.commit},
                "minimum_overlap": provenance.MIN_HIPIFY_OVERLAP,
                "kernels": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _read_regular_file(workspace: Path, relative: str, role: str) -> str:
    path = _inside(workspace, relative, must_exist=True)
    if not path.is_file():
        raise ManifestExecutionError(f"{role} must be a regular file: {relative}")
    return path.read_text(encoding="utf-8", errors="replace")


def _validate_ported_kernel(workspace: Path, kernel: PortedKernel) -> None:
    """Require the CUDA kernel to exist and its port to define a GPU kernel, not PyTorch ops."""

    source = _read_regular_file(workspace, kernel.cuda_source, "ported_kernels.cuda_source")
    if not re.search(rf"\b{re.escape(kernel.cuda_kernel)}\b", source):
        raise ManifestExecutionError(
            f"{kernel.cuda_source} does not mention CUDA kernel {kernel.cuda_kernel}"
        )
    definition = _KERNEL_DEFINITION[kernel.implementation]
    texts = [
        _read_regular_file(workspace, relative, "ported_kernels.port_files item")
        for relative in kernel.port_files
    ]
    if not any(definition.search(text) for text in texts):
        raise ManifestExecutionError(
            f"the port of {kernel.cuda_kernel} is declared as {kernel.implementation} but none "
            f"of {', '.join(kernel.port_files)} defines a {kernel.implementation} kernel; "
            "a PyTorch fallback is not a port"
        )


def _expand_variables(value: str, variables: dict[str, str]) -> str | None:
    """Substitute already-assigned shell variables, or return None if one is unknown."""

    expanded: list[str] = []
    position = 0
    for reference in _VARIABLE_REFERENCE.finditer(value):
        name = reference.group(1) or reference.group(2)
        if name not in variables:
            return None
        expanded.append(value[position : reference.start()])
        expanded.append(variables[name])
        position = reference.end()
    expanded.append(value[position:])
    return "".join(expanded)


def _shell_path_variables(lines: list[str]) -> dict[str, str]:
    """Resolve single-word shell assignments, including ones composed of earlier variables."""

    variables: dict[str, str] = {}
    for line in lines:
        assignment = _ASSIGNMENT.fullmatch(line.removeprefix("export "))
        if assignment is None:
            continue
        raw_value = assignment.group(2)
        if "'" in raw_value:
            # Single quotes suppress expansion; do not guess the runtime value.
            continue
        try:
            tokens = shlex.split(raw_value)
        except ValueError:
            continue
        if len(tokens) != 1:
            continue
        expanded = _expand_variables(tokens[0], variables)
        if expanded is not None:
            variables[assignment.group(1)] = expanded
    return variables


def validate_manifest_files(run_dir: Path, manifest: ExecutionManifest) -> tuple[Path, Path]:
    """Validate all host-side files before starting Docker."""

    if manifest.status != "ready":
        raise ManifestExecutionError(manifest.blocked_reason or "manifest is blocked")
    if isinstance(manifest, RocmPortingManifest):
        validate_porting_files(run_dir, manifest)

    assert manifest.launch_script is not None
    assert manifest.host_repository is not None
    assert manifest.launch_script_sha256 is not None
    assert manifest.smoke_test is not None

    script = _inside(run_dir, manifest.launch_script, must_exist=True)
    repository = _inside(run_dir, manifest.host_repository, must_exist=True)
    smoke_log = _inside(run_dir, manifest.smoke_test.log_path, must_exist=True)

    if script.is_symlink() or not script.is_file():
        raise ManifestExecutionError("launch script must be a regular, non-symlink file")
    if not repository.is_dir():
        raise ManifestExecutionError("host_repository must be a directory")
    if not smoke_log.is_file() or smoke_log.stat().st_size == 0:
        raise ManifestExecutionError("smoke-test log must be a non-empty regular file")
    if _sha256(script) != manifest.launch_script_sha256:
        raise ManifestExecutionError("launch script hash does not match the manifest")

    script_text = script.read_text(encoding="utf-8")
    lines = [line.strip() for line in script_text.splitlines() if line.strip()]
    if not lines or lines[0] != "#!/usr/bin/env bash":
        raise ManifestExecutionError("launch script must start with '#!/usr/bin/env bash'")
    if "set -euo pipefail" not in lines:
        raise ManifestExecutionError("launch script must enable 'set -euo pipefail'")
    assert manifest.container_repository is not None
    syntax_check = subprocess.run(
        ["bash", "-n", str(script)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if syntax_check.returncode != 0:
        detail = syntax_check.stderr.strip() or "unknown bash syntax error"
        raise ManifestExecutionError(f"launch script contains invalid shell syntax: {detail}")

    variables = _shell_path_variables(lines)
    valid_cd = False
    for line in lines:
        if not line.startswith("cd "):
            continue
        tokens = shlex.split(line)
        if len(tokens) != 2:
            continue
        if _expand_variables(tokens[1], variables) == manifest.container_repository:
            valid_cd = True
            break
    if not valid_cd:
        raise ManifestExecutionError(
            "launch script must change to the manifest's exact container_repository"
        )
    if "/run/artifacts/" not in script_text:
        raise ManifestExecutionError("launch script must persist output below /run/artifacts")
    if any(line == "nohup" or line.startswith("nohup ") for line in lines):
        raise ManifestExecutionError("launch script cannot use nohup")
    if any(line.endswith("&") and not line.endswith("&&") for line in lines):
        raise ManifestExecutionError("launch script cannot start background processes")

    wrapper = run_dir / "bin" / "docker"
    if wrapper.is_symlink() or not wrapper.is_file() or not os.access(wrapper, os.X_OK):
        raise ManifestExecutionError(f"Docker wrapper is missing or not executable: {wrapper}")
    return script, wrapper


def _extract_metric(run_dir: Path, manifest: ExecutionManifest) -> tuple[str | None, str | None]:
    """Extract the achieved value using the frozen machine-readable manifest rule."""

    extractor = manifest.metric_extractor
    if extractor is None:
        return None, None
    source = _inside(run_dir, extractor.source, must_exist=True)
    raw = source.read_text(encoding="utf-8", errors="replace")
    if extractor.format == "json":
        value: object = json.loads(raw)
        for component in extractor.selector.split("."):
            if isinstance(value, dict):
                if component not in value:
                    raise ManifestExecutionError(
                        f"metric JSON selector component is missing: {component}"
                    )
                value = value[component]
            elif isinstance(value, list) and component.isdigit():
                index = int(component)
                if index >= len(value):
                    raise ManifestExecutionError(
                        f"metric JSON selector index is out of range: {component}"
                    )
                value = value[index]
            else:
                raise ManifestExecutionError(
                    f"metric JSON selector cannot descend through: {component}"
                )
        if isinstance(value, (dict, list)):
            return json.dumps(value, sort_keys=True), extractor.source
        return str(value), extractor.source

    match = re.search(extractor.selector, raw, flags=re.MULTILINE)
    if match is None:
        raise ManifestExecutionError("metric regex did not match its source artifact")
    if "value" in match.groupdict():
        return match.group("value"), extractor.source
    if match.groups():
        return match.group(1), extractor.source
    return match.group(0), extractor.source


def execute_manifest(
    run_dir: Path,
    manifest: ExecutionManifest,
    *,
    assigned_gpu: str,
) -> ExecutionResult:
    """Run the container-internal launch script and wait for completion."""

    run_dir = run_dir.resolve()
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "executor.log"
    result_path = run_dir / "execution_result.json"

    try:
        _script, wrapper = validate_manifest_files(run_dir, manifest)
    except ManifestExecutionError as exc:
        result = ExecutionResult(
            status="blocked",
            command=[],
            exit_code=None,
            timed_out=False,
            duration_seconds=0,
            log_path=str(log_path),
            expected_outputs=manifest.expected_outputs,
            missing_outputs=manifest.expected_outputs,
            error=str(exc),
        )
        _atomic_json(result_path, result.model_dump(mode="json"))
        return result

    assert manifest.container_repository is not None
    assert manifest.image_tag is not None
    assert manifest.timeout_seconds is not None
    container_script = f"/run/{manifest.launch_script}"
    command = [
        str(wrapper),
        "run",
        "-w",
        manifest.container_repository,
        manifest.image_tag,
        "bash",
        container_script,
    ]
    run_id = run_dir.name
    env = os.environ.copy()
    env.update(
        {
            "RUN_DIR": str(run_dir),
            "RUN_ID": run_id,
            "ASSIGNED_GPU": str(assigned_gpu),
            "PATH": f"{run_dir / 'bin'}{os.pathsep}{env.get('PATH', '')}",
        }
    )

    started = time.monotonic()
    timed_out = False
    exit_code: int | None = None
    error: str | None = None
    with log_path.open("wb") as output:
        output.write(f"[two-stage] command: {json.dumps(command)}\n".encode())
        output.flush()
        try:
            process = subprocess.Popen(
                command,
                cwd=str(run_dir / "workspace"),
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                exit_code = process.wait(timeout=manifest.timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                error = f"execution exceeded {manifest.timeout_seconds} seconds"
                _terminate_process_group(process)
                _stop_labeled_containers(run_id)
                exit_code = -1
        except OSError as exc:
            error = f"{type(exc).__name__}: {exc}"

    duration = time.monotonic() - started
    expected = [_inside(run_dir, relative) for relative in manifest.expected_outputs]
    missing = [
        relative
        for relative, path in zip(manifest.expected_outputs, expected, strict=True)
        if not path.exists()
    ]
    achieved_metric: str | None = None
    metric_source: str | None = None
    metric_extraction_error: str | None = None
    if not timed_out and exit_code == 0 and not missing and manifest.metric_extractor:
        try:
            achieved_metric, metric_source = _extract_metric(run_dir, manifest)
        except (ManifestExecutionError, OSError, ValueError, json.JSONDecodeError) as exc:
            metric_extraction_error = str(exc)
    if timed_out:
        status = "timed_out"
    elif exit_code == 0 and not missing and metric_extraction_error is None:
        status = "succeeded"
    else:
        status = "failed"
        if error is None:
            if exit_code != 0:
                error = f"launch command exited with code {exit_code}"
            elif metric_extraction_error:
                error = f"metric extraction failed: {metric_extraction_error}"
            else:
                error = f"missing expected outputs: {', '.join(missing)}"

    result = ExecutionResult(
        status=status,
        command=command,
        exit_code=exit_code,
        timed_out=timed_out,
        duration_seconds=duration,
        log_path=str(log_path),
        expected_outputs=manifest.expected_outputs,
        missing_outputs=missing,
        error=error,
        achieved_metric=achieved_metric,
        metric_source=metric_source,
        metric_extraction_error=metric_extraction_error,
    )
    _atomic_json(result_path, result.model_dump(mode="json"))
    return result
