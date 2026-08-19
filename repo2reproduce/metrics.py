"""Exec-Score metrics for Repo2Reproduce runs: Exec Rate and Performance Gap.

These are the two "execution fidelity" metrics from AutoReproduce / ReproduceBench
(arXiv:2505.20662), translated to Repo2Reproduce's ``report.json`` shape so results
are comparable apple-to-apple:

- **Exec Rate**: fraction of runs whose reproduction actually executed and produced a
  numeric result.
- **Performance Gap (↓)**: the relative distance between the paper's reference number
  and the number our executed run achieved, using the paper's exact formula::

      Gap = (1/n) * Σ_i  |P_ref_i - P_agent_i| / max(P_ref_i, P_agent_i)

  Non-executable runs get ``P_agent = 0`` (→ a per-run gap of 1.0), matching the
  benchmark. The ``max`` normalization keeps every per-run gap in ``[0, 1]``.

The achieved/target values in ``report.json`` are free-text (e.g. ``"82.73 ± 4.62
(%)"`` or ``"Acc: 0.8272"``), so we parse the primary number heuristically and, when
present, prefer an explicit ``raw:`` capture (the un-scaled fraction).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# --------------------------------------------------------------------------- #
# Number parsing
# --------------------------------------------------------------------------- #
# A signed decimal, optionally in scientific notation.
_NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
# The "mean ± std" head of a metric string; we only need the mean.
_MEAN_RE = re.compile(rf"({_NUM})\s*(%?)\s*(?:±|\+/-|\+-)")
# An explicit primary metric like "Acc: 0.827" / "MSE = 0.0074" / "PSNR: 39.68".
_LABELLED_RE = re.compile(
    rf"(?:Acc|Accuracy|MSE|PSNR|SSIM|Recall|ADE|DSC|Dice|Score|PPL|Perplexity|"
    rf"delta|δ)\s*[:=]?\s*({_NUM})",
    re.IGNORECASE,
)
# A trailing "[raw: ... <num> ...]" often carries the un-scaled fraction we prefer.
_RAW_RE = re.compile(rf"raw:[^0-9\-+]*({_NUM})", re.IGNORECASE)
_ANY_NUM_RE = re.compile(_NUM)


def parse_metric_value(text: Optional[str]) -> Optional[float]:
    """Extract the primary scalar from a free-text metric string.

    Resolution order (first hit wins):
      1. an explicit ``raw: <num>`` capture (the canonical un-scaled value),
      2. a labelled metric (``Acc:``, ``MSE=`` ...),
      3. the mean in a ``mean ± std`` expression,
      4. the first number anywhere.

    A trailing ``%`` on the mean is *not* auto-divided; percent vs fraction is
    reconciled in :func:`_align_scales` against the reference value.
    """
    if not text:
        return None
    for rx in (_RAW_RE, _LABELLED_RE, _MEAN_RE):
        m = rx.search(text)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    m = _ANY_NUM_RE.search(text)
    if m:
        try:
            return float(m.group(0))
        except ValueError:
            return None
    return None


def _align_scales(ref: float, agent: float) -> tuple[float, float]:
    """Put ref/agent on the same scale when one is a percentage and the other a fraction.

    Heuristic: if the two differ by roughly ×100 (one looks like a percent, the other a
    [0,1] fraction), scale the fraction up to percent. This is only triggered when the
    ratio is large enough to be unambiguous, so e.g. MSE pairs like (0.0074, 0.0080)
    are left untouched.
    """
    if ref == 0 or agent == 0:
        return ref, agent
    hi, lo = (ref, agent) if abs(ref) >= abs(agent) else (agent, ref)
    ratio = abs(hi) / abs(lo)
    if 20 <= ratio <= 500 and abs(lo) <= 1.5 < abs(hi):
        # lo is a fraction, hi is a percent -> lift the fraction to percent.
        if lo is agent or agent == lo:
            return ref, agent * 100.0
        return ref * 100.0, agent
    return ref, agent


# --------------------------------------------------------------------------- #
# Perf-gap formula (identical to AutoReproduce cal_per_gap.calculate_perf_gap)
# --------------------------------------------------------------------------- #
def performance_gap(ref_val: float, agent_val: float) -> float:
    """Relative performance gap for a single run, normalized to ``[0, 1]``."""
    numerator = abs(ref_val - agent_val)
    denominator = max(ref_val, agent_val)
    if denominator == 0:
        return 0.0
    return numerator / denominator


# --------------------------------------------------------------------------- #
# Per-run + aggregate results
# --------------------------------------------------------------------------- #
@dataclass
class RunMetric:
    name: str
    executed: bool
    ref_value: Optional[float]
    agent_value: Optional[float]
    gap: float
    achieved_raw: Optional[str] = None
    target_raw: Optional[str] = None
    note: Optional[str] = None


@dataclass
class MetricsSummary:
    runs: list[RunMetric] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.runs)

    @property
    def n_executed(self) -> int:
        return sum(1 for r in self.runs if r.executed)

    @property
    def exec_rate(self) -> float:
        return self.n_executed / self.n if self.n else 0.0

    @property
    def perf_gap(self) -> float:
        """Mean per-run gap over ALL runs (non-executed count as 1.0)."""
        return sum(r.gap for r in self.runs) / self.n if self.n else 0.0

    @property
    def perf_gap_executed_only(self) -> float:
        """Mean gap over only the runs that executed and had both numbers."""
        vals = [r.gap for r in self.runs if r.executed and r.agent_value is not None]
        return sum(vals) / len(vals) if vals else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_runs": self.n,
            "n_executed": self.n_executed,
            "exec_rate": round(self.exec_rate, 6),
            "perf_gap": round(self.perf_gap, 6),
            "perf_gap_executed_only": round(self.perf_gap_executed_only, 6),
            "runs": [
                {
                    "name": r.name,
                    "executed": r.executed,
                    "ref_value": r.ref_value,
                    "agent_value": r.agent_value,
                    "gap": round(r.gap, 6),
                    "note": r.note,
                }
                for r in self.runs
            ],
        }


# --------------------------------------------------------------------------- #
# Loading runs
# --------------------------------------------------------------------------- #
def _load_report(report_path: Path) -> Optional[dict[str, Any]]:
    try:
        return json.loads(report_path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def metric_from_report(
    report: dict[str, Any],
    name: str,
    reference_override: Optional[float] = None,
) -> RunMetric:
    """Build a :class:`RunMetric` from one report dict.

    ``reference_override`` lets callers pin ``P_ref`` to the benchmark's own baseline
    number (recommended for apple-to-apple comparison) instead of the paper target
    string parsed from the report.
    """
    rr = report.get("output5_reproduction_result") or {}
    executed = bool(rr.get("executed")) and rr.get("achieved_metric") is not None

    achieved_raw = rr.get("achieved_metric")
    target_raw = rr.get("paper_target")

    agent_value = parse_metric_value(achieved_raw) if executed else None
    ref_value = (
        reference_override
        if reference_override is not None
        else parse_metric_value(target_raw)
    )

    note = None
    if not executed:
        # Non-executable run: P_agent = 0 -> max gap of 1.0 (matches the benchmark).
        gap = 1.0
        note = "not executed (P_agent=0)"
    elif ref_value is None:
        gap = 1.0
        note = "no reference value to compare"
    elif agent_value is None:
        gap = 1.0
        note = "executed but achieved metric unparseable"
    else:
        ref_value, agent_value = _align_scales(ref_value, agent_value)
        gap = performance_gap(ref_value, agent_value)

    return RunMetric(
        name=name,
        executed=executed,
        ref_value=ref_value,
        agent_value=agent_value,
        gap=gap,
        achieved_raw=achieved_raw,
        target_raw=target_raw,
        note=note,
    )


def evaluate_runs(
    runs_dir: Path,
    references: Optional[dict[str, float]] = None,
) -> MetricsSummary:
    """Scan ``runs_dir`` for ``*/report.json`` and compute Exec Rate + Perf Gap.

    ``references`` optionally maps a run's directory name to its benchmark reference
    value (``P_ref``); when a run isn't in the map, the paper target string from its
    report is parsed instead.
    """
    references = references or {}
    summary = MetricsSummary()
    for report_path in sorted(runs_dir.glob("*/report.json")):
        report = _load_report(report_path)
        name = report_path.parent.name
        if report is None:
            summary.runs.append(
                RunMetric(
                    name=name,
                    executed=False,
                    ref_value=None,
                    agent_value=None,
                    gap=1.0,
                    note="report.json unreadable",
                )
            )
            continue
        summary.runs.append(
            metric_from_report(report, name, reference_override=references.get(name))
        )
    return summary
