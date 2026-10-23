"""Host commands a ``--rocm-porting`` preparation agent is allowed to run.

``port-scan`` reads the cloned repository. ``port-hipify`` translates a copy of the clone with
AMD's official HIPIFY (``hipify-perl`` from https://github.com/ROCm/HIPIFY, pinned and cloned
into ``$RUN_DIR/tools``); it writes only below the run's workspace and never edits the clone.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from repo2reproduce.porting.official_hipify import hipify_repository
from repo2reproduce.porting.scanner import SEVERITIES, scan_path


def _workspace() -> Path:
    return (Path(os.environ["RUN_DIR"]).resolve() / "workspace").resolve()


def workspace_target(raw: str) -> Path:
    """Resolve ``raw`` and require it to stay inside this run's workspace."""

    workspace = _workspace()
    candidate = Path(raw).expanduser().resolve()
    if candidate != workspace and workspace not in candidate.parents:
        raise SystemExit(f"path must stay inside {workspace}")
    if not candidate.exists():
        raise SystemExit(f"path does not exist: {candidate}")
    return candidate


def _write_reports(stem: str, text: str, payload: dict) -> None:
    artifacts = Path(os.environ["RUN_DIR"]).resolve() / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / f"{stem}.txt").write_text(text + "\n", encoding="utf-8")
    (artifacts / f"{stem}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def scan_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Report CUDA-to-ROCm findings for a clone inside this run's workspace."
    )
    parser.add_argument("path", help="Host path of the cloned repository")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--min-severity", choices=SEVERITIES, default="high")
    parser.add_argument("--max-findings", type=int, default=80)
    args = parser.parse_args(argv)
    if args.max_findings <= 0:
        raise SystemExit("--max-findings must be positive")
    report = scan_path(workspace_target(args.path)).filtered(args.min_severity)
    text = report.to_text(max_findings=args.max_findings)
    _write_reports("port-scan", text, report.to_dict())
    print(json.dumps(report.to_dict(), indent=2) if args.json else text)
    return 0


def hipify_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Translate a copy of a clone inside this run's workspace with the official "
            "ROCm HIPIFY (hipify-perl)."
        )
    )
    parser.add_argument("path", help="Host path of the cloned repository (left unchanged)")
    parser.add_argument(
        "--output", help="Host path for the translated copy (default: <clone>-hip next to it)"
    )
    parser.add_argument("--overwrite", action="store_true", help="Regenerate an existing output")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")
    root = workspace_target(args.path)
    if not root.is_dir():
        raise SystemExit(f"path must be a directory: {root}")
    workspace = _workspace()
    if root == workspace:
        raise SystemExit("path must be the clone, not the whole workspace")
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else root.with_name(f"{root.name}-hip")
    )
    if workspace not in output.parents:
        raise SystemExit(f"--output must stay inside {workspace}")
    run_dir = Path(os.environ["RUN_DIR"]).resolve()
    report = hipify_repository(
        root,
        output,
        cache_dir=run_dir / "tools",
        overwrite=args.overwrite,
        timeout=args.timeout,
    )
    text = report.to_text()
    _write_reports("port-hipify", text, report.to_dict())
    print(json.dumps(report.to_dict(), indent=2) if args.json else text)
    return 1 if report.error else 0


if __name__ == "__main__":
    command = Path(sys.argv[0]).name
    raise SystemExit(hipify_main() if command == "port-hipify" else scan_main())
