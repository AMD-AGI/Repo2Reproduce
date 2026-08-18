"""Choose the best available report and classify whether a run completed.

This is intentionally separate from the two Markdown reviewer agents.  It is
small deterministic plumbing for issues #5 and #8:

* a complete report written by ``save_report`` must not be overwritten by a
  degenerate final structured-output stub;
* execute-mode runs that never reached build/execution must not exit as
  successful merely because a schema-valid placeholder report exists.
"""

from __future__ import annotations

from typing import Any, Optional


_PLACEHOLDERS = {
    "",
    "-",
    "n/a",
    "na",
    "none",
    "null",
    "unknown",
    "test one",
    "test two",
    "not attempted",
}


def _meaningful(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in _PLACEHOLDERS
    if isinstance(value, dict):
        return any(_meaningful(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_meaningful(v) for v in value)
    return True


def report_completeness(report: Optional[dict[str, Any]]) -> tuple[int, int]:
    """Return ``(populated pipeline outputs, meaningful leaf count)``."""
    if not isinstance(report, dict):
        return (0, 0)

    output_keys = (
        "output1_experiment_selection",
        "output2_configuration",
        "output3_replication_plan",
        "output4_amd_environment",
        "output5_reproduction_result",
    )
    populated = sum(1 for key in output_keys if _meaningful(report.get(key)))

    def leaves(value: Any) -> int:
        if isinstance(value, dict):
            return sum(leaves(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return sum(leaves(v) for v in value)
        return 1 if _meaningful(value) else 0

    return (populated, leaves(report))


def choose_best_report(
    structured: Optional[dict[str, Any]],
    saved: Optional[dict[str, Any]],
) -> tuple[Optional[dict[str, Any]], str]:
    """Prefer whichever report contains more completed work."""
    structured_score = report_completeness(structured)
    saved_score = report_completeness(saved)

    if saved is not None and saved_score > structured_score:
        return saved, "saved_report"
    if structured is not None:
        return structured, "structured_output"
    if saved is not None:
        return saved, "saved_report"
    return None, "none"


def classify_run(
    report: Optional[dict[str, Any]],
    *,
    is_error: bool,
    execute_requested: bool,
) -> tuple[str, list[str]]:
    """Classify the workflow result without judging whether the metric matched."""
    reasons: list[str] = []
    if is_error:
        reasons.append("agent SDK returned an error")
    if not isinstance(report, dict):
        reasons.append("no report was produced")
        return "incomplete", reasons

    for key in (
        "output1_experiment_selection",
        "output2_configuration",
        "output3_replication_plan",
    ):
        if not _meaningful(report.get(key)):
            reasons.append(f"required phase missing: {key}")

    if not execute_requested:
        return ("incomplete", reasons) if reasons else ("completed", [])

    amd = report.get("output4_amd_environment") or {}
    result = report.get("output5_reproduction_result") or {}
    build_status = str(amd.get("build_status") or "not_attempted")
    executed = bool(result.get("executed"))

    if executed:
        return ("incomplete", reasons) if reasons else ("completed", [])

    if build_status in {"build_failed", "integrity_failed"}:
        reasons.append(f"environment attempt failed: {build_status}")
        return (
            ("incomplete", reasons)
            if any(r.startswith("required phase missing") for r in reasons)
            else ("completed_with_failure", reasons)
        )

    if build_status == "verified" and _meaningful(result):
        reasons.append("environment verified but reproduction did not execute")
        return (
            ("incomplete", reasons)
            if any(r.startswith("required phase missing") for r in reasons)
            else ("completed_with_failure", reasons)
        )

    reasons.append("execute mode never reached a verified build or executor result")
    return "incomplete", reasons
