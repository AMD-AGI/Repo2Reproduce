from __future__ import annotations

from repo2reproduce.gate import assess_report, report_is_complete


def _base() -> dict:
    """A minimal report with the two hard-required Track-1 outputs present."""
    return {
        "output1_experiment_selection": {
            "selected_experiment": "Table 1",
            "target_result": "1.0 accuracy",
        },
        "output2_configuration": {"model": {}},
        "output3_replication_plan": {"commands": [{"step": "evaluate", "cmd": "python x.py"}]},
    }


def test_complete_when_core_outputs_present():
    a = assess_report(_base())
    assert a.ok
    assert a.status == "complete"
    assert a.hard_missing == []
    assert a.soft_missing == []


def test_missing_config_is_soft_partial_not_blocking():
    report = _base()
    del report["output2_configuration"]
    a = assess_report(report)
    assert a.ok  # still usable
    assert a.status == "partial"
    assert any("output2" in m for m in a.soft_missing)
    assert a.hard_missing == []


def test_missing_commands_is_hard():
    report = _base()
    report["output3_replication_plan"] = {"commands": []}
    a = assess_report(report)
    assert not a.ok
    assert a.status == "incomplete"
    assert any("output3" in m for m in a.hard_missing)


def test_missing_experiment_target_is_hard():
    report = _base()
    report["output1_experiment_selection"] = {"selected_experiment": "Table 1"}
    a = assess_report(report)
    assert not a.ok
    assert any("output1" in m for m in a.hard_missing)


def test_amd_missing_environment_is_soft():
    a = assess_report(_base(), amd=True, execute=False)
    assert a.ok
    assert a.status == "partial"
    assert any("output4" in m for m in a.soft_missing)


def test_execute_silent_skip_is_hard():
    report = _base()
    report["output4_amd_environment"] = {"build_status": "verified"}
    report["output5_reproduction_result"] = {"executed": False}
    a = assess_report(report, amd=True, execute=True)
    assert not a.ok
    assert any("output5" in m for m in a.hard_missing)


def test_execute_honest_build_failure_is_acceptable():
    report = _base()
    report["output4_amd_environment"] = {"build_status": "build_failed"}
    report["output5_reproduction_result"] = {"executed": False}
    a = assess_report(report, amd=True, execute=True)
    assert a.ok  # honest failure is not a silent skip
    assert not any("output5" in m for m in a.hard_missing)


def test_executed_result_with_soft_gap_stays_usable():
    # The scenario we care about: executor produced a valid result, but an unrelated
    # optional output (config) is missing -> the report must remain usable (partial).
    report = _base()
    del report["output2_configuration"]
    report["output4_amd_environment"] = {"build_status": "verified"}
    report["output5_reproduction_result"] = {"executed": True, "achieved_metric": "1.0"}
    a = assess_report(report, amd=True, execute=True)
    assert a.ok
    assert a.status == "partial"


def test_backward_compat_wrapper():
    ok, missing = report_is_complete(_base())
    assert ok
    assert missing == []


def test_none_report_is_incomplete():
    a = assess_report(None)
    assert not a.ok
    assert a.status == "incomplete"
