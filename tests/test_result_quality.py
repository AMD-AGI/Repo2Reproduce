from repo2reproduce.result_quality import (
    choose_best_report,
    classify_run,
    report_completeness,
)


def _complete_report(executed=True):
    return {
        "output1_experiment_selection": {
            "selected_experiment": "Table 1",
            "target_result": "F1 0.80",
        },
        "output2_configuration": {
            "model": {"value": "model"},
            "dataset": {"value": "data"},
        },
        "output3_replication_plan": {
            "commands": [{"step": "evaluate", "cmd": "python eval.py"}],
        },
        "output4_amd_environment": {"build_status": "verified"},
        "output5_reproduction_result": {
            "executed": executed,
            "achieved_metric": "F1 0.79" if executed else None,
            "notes": "evaluation attempted",
        },
    }


def test_saved_complete_report_beats_structured_stub():
    saved = _complete_report()
    stub = {
        "output1_experiment_selection": {},
        "output2_configuration": {},
        "output3_replication_plan": {},
        "gaps_and_assumptions": ["test one", "test two"],
    }

    report, source = choose_best_report(stub, saved)

    assert report == saved
    assert source == "saved_report"
    assert report_completeness(saved) > report_completeness(stub)


def test_execute_mode_placeholder_is_incomplete():
    report = _complete_report(executed=False)
    report["output4_amd_environment"] = {"build_status": "not_attempted"}

    status, reasons = classify_run(
        report,
        is_error=False,
        execute_requested=True,
    )

    assert status == "incomplete"
    assert any("never reached" in reason for reason in reasons)


def test_attempted_build_failure_is_completed_with_failure():
    report = _complete_report(executed=False)
    report["output4_amd_environment"] = {"build_status": "build_failed"}

    status, reasons = classify_run(
        report,
        is_error=False,
        execute_requested=True,
    )

    assert status == "completed_with_failure"
    assert any("build_failed" in reason for reason in reasons)


def test_plan_only_complete_report_is_completed():
    report = _complete_report(executed=False)

    status, reasons = classify_run(
        report,
        is_error=False,
        execute_requested=False,
    )

    assert status == "completed"
    assert reasons == []
