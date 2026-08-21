"""The finalization gate: a deterministic, *graded* completeness check over the report.

This is intentionally a plain, side-effect-free module (no LLM, no I/O) so the Python
phase runner reaches a deterministic verdict and cannot be talked out of it. The runner
calls it once after assembling the report from persisted phase artifacts.

The verdict is graded, not all-or-nothing:

* ``hard_missing`` — omissions that leave no usable reproduction plan (no experiment/
  target, no commands, or — in execute mode — a *silently* skipped run). These make the
  report ``incomplete`` and the run exit non-zero.
* ``soft_missing`` — gaps worth surfacing that do NOT invalidate the valid results
  already produced (a thin config object, or a missing AMD environment detail). These
  make the report ``partial`` but still fully written and usable.

This encodes the pipeline promise from ``prompts/orchestrator_amd.md``: a run is not
"done" until the required outputs exist and honest ``skipped``/failure states are
allowed — while ensuring valid work (e.g. a successful executor result) is never
discarded just because one advisory item is absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _s(value: Any) -> str:
    """Coerce to a stripped string ('' for None) so emptiness checks are uniform."""
    return str(value).strip() if value is not None else ""


@dataclass
class ReportAssessment:
    """Graded verdict over a candidate final report."""

    hard_missing: list[str] = field(default_factory=list)
    soft_missing: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        """``complete`` | ``partial`` | ``incomplete``."""
        if self.hard_missing:
            return "incomplete"
        if self.soft_missing:
            return "partial"
        return "complete"

    @property
    def ok(self) -> bool:
        """True when nothing *required* is missing (``partial`` still counts as usable)."""
        return not self.hard_missing


def assess_report(
    report: dict[str, Any] | None,
    amd: bool = False,
    execute: bool = False,
) -> ReportAssessment:
    """Grade a candidate report into hard (blocking) vs soft (advisory) omissions.

    Hard (a run without these has no usable reproduction plan):

    * Output 1 — ``selected_experiment`` + ``target_result``.
    * Output 3 — at least one command.
    * ``execute`` — Output 5 must be a real result UNLESS the AMD build honestly failed
      (``build_failed`` / ``integrity_failed``); a silent skip stays *hard* so premature
      finalization is still flagged.

    Soft (surfaced, but valid results are still written cleanly):

    * Output 2 — a configuration object.
    * ``amd`` — Output 4 with a ``build_status``.
    """
    if not isinstance(report, dict):
        return ReportAssessment(hard_missing=["report (no structured report produced)"])

    hard: list[str] = []
    soft: list[str] = []

    # --- Output 1: experiment selection (HARD) ------------------------------------------
    out1 = report.get("output1_experiment_selection") or {}
    if not (_s(out1.get("selected_experiment")) and _s(out1.get("target_result"))):
        hard.append("output1_experiment_selection (selected_experiment + target_result)")

    # --- Output 3: replication commands (HARD) ------------------------------------------
    out3 = report.get("output3_replication_plan") or {}
    if not (out3.get("commands") or []):
        hard.append("output3_replication_plan.commands (at least one command)")

    # --- Output 2: configuration (SOFT: just presence, kept lenient on purpose) ---------
    if not isinstance(report.get("output2_configuration"), dict):
        soft.append("output2_configuration")

    # --- Track 2 (--amd): Output 4 (SOFT) -----------------------------------------------
    out4 = report.get("output4_amd_environment")
    if amd and (not isinstance(out4, dict) or not _s(out4.get("build_status"))):
        soft.append("output4_amd_environment (with build_status)")

    # --- Execute mode: Output 5 (HARD unless an honest build/verify failure) ------------
    if execute:
        out5 = report.get("output5_reproduction_result")
        if not isinstance(out5, dict):
            hard.append("output5_reproduction_result")
        elif not bool(out5.get("executed")):
            # Not executed is acceptable ONLY when a real build/verify attempt was made
            # and honestly failed (``build_failed`` / ``integrity_failed``). Every other
            # state is the premature-skip signature we exist to catch.
            build_status = _s((out4 or {}).get("build_status")) if isinstance(out4, dict) else ""
            if build_status not in ("build_failed", "integrity_failed"):
                hard.append(
                    "output5_reproduction_result.executed (reproduction not run; "
                    f"build_status={build_status or 'none'} is not an honest build/verify "
                    "failure -> the build/execute phase was skipped)"
                )

    return ReportAssessment(hard_missing=hard, soft_missing=soft)


def report_is_complete(
    report: dict[str, Any] | None,
    amd: bool = False,
    execute: bool = False,
) -> tuple[bool, list[str]]:
    """Backward-compatible wrapper.

    ``ok`` is True iff nothing *required* (hard) is missing. ``missing`` combines hard +
    soft so existing callers keep seeing every gap.
    """
    assessment = assess_report(report, amd=amd, execute=execute)
    return assessment.ok, assessment.hard_missing + assessment.soft_missing
