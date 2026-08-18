"""The finalization gate: a deterministic completeness check over the final report.

This is intentionally a plain, side-effect-free function (no LLM, no I/O) so the Python
phase runner reaches a deterministic verdict and cannot be talked out of failing. The
runner calls it once after assembling the report from persisted phase artifacts.

The contract encodes the pipeline promise that today lives only as prose in
``prompts/orchestrator_amd.md``: a run is not "done" until the required outputs actually
exist. ``skipped``/honest-failure states are allowed; only silent omission is blocked.
"""

from __future__ import annotations

from typing import Any


def _s(value: Any) -> str:
    """Coerce to a stripped string ('' for None) so emptiness checks are uniform."""
    return str(value).strip() if value is not None else ""


def report_is_complete(
    report: dict[str, Any] | None,
    amd: bool = False,
    execute: bool = False,
) -> tuple[bool, list[str]]:
    """Return ``(ok, missing)`` for a candidate final report.

    ``missing`` is a list of human-readable ids of the required outputs that are absent or
    empty. ``ok`` is ``True`` iff ``missing`` is empty. The required set widens with the
    run mode:

    * always (Track 1): Output 1 (experiment + target), Output 3 (>=1 command).
    * ``amd``: Output 4 (an AMD environment with a ``build_status``).
    * ``execute``: Output 5, and it must be a real result (``executed``) UNLESS the AMD
      environment honestly failed to verify (in which case not running is acceptable).

    Output 2 is checked only for presence (a config object), kept lenient on purpose.
    """
    missing: list[str] = []

    if not isinstance(report, dict):
        # No report at all -> everything required is missing.
        return False, ["report (no structured report produced)"]

    # --- Output 1: experiment selection -------------------------------------------------
    out1 = report.get("output1_experiment_selection") or {}
    if not (_s(out1.get("selected_experiment")) and _s(out1.get("target_result"))):
        missing.append("output1_experiment_selection (selected_experiment + target_result)")

    # --- Output 2: configuration (lenient: just present) --------------------------------
    if not isinstance(report.get("output2_configuration"), dict):
        missing.append("output2_configuration")

    # --- Output 3: replication commands -------------------------------------------------
    out3 = report.get("output3_replication_plan") or {}
    if not (out3.get("commands") or []):
        missing.append("output3_replication_plan.commands (at least one command)")

    # --- Track 2 (--amd): Output 4 ------------------------------------------------------
    out4 = report.get("output4_amd_environment")
    if amd:
        if not isinstance(out4, dict) or not _s(out4.get("build_status")):
            missing.append("output4_amd_environment (with build_status)")

    # --- Execute mode: Output 5 ---------------------------------------------------------
    if execute:
        out5 = report.get("output5_reproduction_result")
        if not isinstance(out5, dict):
            missing.append("output5_reproduction_result")
        elif not bool(out5.get("executed")):
            # Not executed is acceptable ONLY when a real build/verify attempt was made
            # and honestly failed (``build_failed`` / ``integrity_failed``). Every other
            # state is the premature-skip signature we exist to catch:
            #   * ``verified``      -> env was fine but nothing ran,
            #   * ``not_attempted`` -> the build/execute phase was skipped entirely,
            #   * ``""``            -> no AMD environment at all.
            build_status = _s((out4 or {}).get("build_status")) if isinstance(out4, dict) else ""
            if build_status not in ("build_failed", "integrity_failed"):
                missing.append(
                    "output5_reproduction_result.executed (reproduction not run; "
                    f"build_status={build_status or 'none'} is not an honest build/verify "
                    "failure -> the build/execute phase was skipped)"
                )

    return (len(missing) == 0, missing)
