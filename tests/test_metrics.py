"""Tests for Exec-Score metrics: value parsing, perf-gap formula, and aggregation."""

from __future__ import annotations

import json
from pathlib import Path

from repo2reproduce.metrics import (
    evaluate_runs,
    metric_from_report,
    parse_metric_value,
    performance_gap,
)


def test_parse_prefers_raw_capture():
    text = "82.73 ± 4.62 (%)  [raw: Acc: 0.8272972972972973 Std: 0.0461]"
    assert parse_metric_value(text) == 0.8272972972972973


def test_parse_percent_mean():
    assert parse_metric_value("82.54% ± 4.33% mean test accuracy") == 82.54


def test_parse_labelled_and_target_with_locator():
    assert parse_metric_value("83.00% ± 4.50% (p.8, Table 3, Texas TDGNN-w)") == 83.00
    assert parse_metric_value("MSE = 0.0074 on Darcy") == 0.0074


def test_parse_perplexity_does_not_read_dataset_suffix():
    assert parse_metric_value("WikiText-2 PPL: 5.758") == 5.758
    assert parse_metric_value("C4 perplexity: 7.734") == 7.734


def test_parse_none():
    assert parse_metric_value(None) is None
    assert parse_metric_value("no numbers here") is None


def test_performance_gap_formula():
    assert performance_gap(83.0, 82.54) == abs(83.0 - 82.54) / 83.0
    # Normalized by the larger value, so always in [0, 1].
    assert performance_gap(0.1, 0.3) == (0.3 - 0.1) / 0.3
    assert performance_gap(0.0, 0.0) == 0.0


def test_non_executed_run_gap_is_one():
    m = metric_from_report({"output5_reproduction_result": {"executed": False}}, "x")
    assert m.executed is False
    assert m.gap == 1.0
    assert m.agent_value is None


def test_executed_run_scale_alignment():
    report = {
        "output5_reproduction_result": {
            "executed": True,
            "achieved_metric": "Acc: 0.8272 Std: 0.046",  # fraction
            "paper_target": "83.00 ± 4.50 (%)",  # percent
        }
    }
    m = metric_from_report(report, "tdgnn")
    assert m.executed is True
    # Fraction lifted to percent before comparison.
    assert abs(m.agent_value - 82.72) < 1e-6
    assert m.gap < 0.01


def test_reference_override_pins_baseline():
    report = {
        "output5_reproduction_result": {
            "executed": True,
            "achieved_metric": "82.73 ± 4.62 (%)",
            "paper_target": "83.00 ± 4.50 (%)",
        }
    }
    m = metric_from_report(report, "tdgnn", reference_override=76.51)
    assert m.ref_value == 76.51
    assert abs(m.gap - abs(76.51 - 82.73) / 82.73) < 1e-6


def test_evaluate_runs_aggregates(tmp_path: Path):
    runs = tmp_path / "runs"
    # one executed, near-perfect
    good = runs / "good"
    good.mkdir(parents=True)
    (good / "report.json").write_text(
        json.dumps(
            {
                "output5_reproduction_result": {
                    "executed": True,
                    "achieved_metric": "82.54% ± 4.33%",
                    "paper_target": "83.00% ± 4.50%",
                }
            }
        ),
        "utf-8",
    )
    # one plan-only (no output5)
    plan = runs / "planonly"
    plan.mkdir(parents=True)
    (plan / "report.json").write_text(json.dumps({"paper_url": "x"}), "utf-8")

    summary = evaluate_runs(runs)
    assert summary.n == 2
    assert summary.n_executed == 1
    assert summary.exec_rate == 0.5
    # all-runs gap = mean(good_gap, 1.0); executed-only gap = good_gap
    assert summary.perf_gap_executed_only < 0.01
    assert 0.5 < summary.perf_gap < 0.51


def test_evaluate_runs_excludes_incomplete_workflows(tmp_path: Path):
    runs = tmp_path / "runs"
    complete = runs / "complete"
    complete.mkdir(parents=True)
    (complete / "report.json").write_text(
        json.dumps(
            {
                "run_status": "completed",
                "output5_reproduction_result": {
                    "executed": True,
                    "achieved_metric": "PPL: 5.758",
                    "paper_target": "PPL: 5.79",
                },
            }
        ),
        "utf-8",
    )
    incomplete = runs / "incomplete"
    incomplete.mkdir()
    (incomplete / "report.json").write_text(
        json.dumps(
            {
                "run_status": "incomplete",
                "output5_reproduction_result": {"executed": False},
            }
        ),
        "utf-8",
    )

    summary = evaluate_runs(runs)

    assert summary.n == 1
    assert summary.runs[0].name == "complete"
