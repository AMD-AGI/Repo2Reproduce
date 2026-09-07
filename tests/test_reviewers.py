from __future__ import annotations

import asyncio
from pathlib import Path

from repo2reproduce.reviewers import run_final_review, run_preflight_review


PROMPTS = Path(__file__).parents[1] / "repo2reproduce" / "prompts"


def test_selected_result_prompts_have_no_multi_arm_requirements():
    prompt_names = [
        "preflight_verifier.md",
        "executor.md",
        "final_verifier.md",
    ]
    prompts = {
        name: " ".join((PROMPTS / name).read_text(encoding="utf-8").lower().split())
        for name in prompt_names
    }

    prohibited = [
        "comparison arm",
        "comparative claim",
        "required experiment and comparisons",
        "all required conditions",
        "cheap baseline",
        "second condition",
    ]
    for name, prompt in prompts.items():
        for phrase in prohibited:
            assert phrase not in prompt, f"{phrase!r} remains in {name}"

    executor = prompts["executor.md"]
    assert "single result in output 1" in executor
    assert "full dataset and split" in executor
    assert "epochs and steps" in executor
    assert "sequence lengths" in executor
    assert "complete evaluation coverage" in executor

    final_verifier = prompts["final_verifier.md"]
    assert "full selected-result workload" in final_verifier


def test_preflight_review_writes_markdown(tmp_path):
    calls = []

    async def fake_runner(task, run_dir, prompt, tools):
        calls.append((task, run_dir, prompt, tools))
        return "# Pre-run verification\n\n## Recommendation\nProceed with caveats."

    markdown = asyncio.run(
        run_preflight_review(
            "https://arxiv.org/abs/1234.5678",
            "https://github.com/org/repo",
            tmp_path,
            target="Table 2",
            runner=fake_runner,
        )
    )

    assert "Proceed with caveats" in markdown
    assert (tmp_path / "preflight.md").read_text().strip() == markdown
    task, _run_dir, prompt, tools = calls[0]
    assert "ingest_paper" in task
    assert "Table 2" in task
    assert "model or dataset gated" in prompt
    assert "mcp__r2r__clone_repo" in tools


def test_final_review_writes_candidate_and_markdown(tmp_path):
    calls = []

    async def fake_runner(task, run_dir, prompt, tools):
        calls.append((task, run_dir, prompt, tools))
        return "# Final verification\n\n## Final assessment\nIncomplete."

    report = {
        "output5_reproduction_result": {
            "executed": False,
            "achieved_metric": None,
        }
    }
    markdown = asyncio.run(
        run_final_review(
            tmp_path,
            report,
            "raw SDK result",
            runner=fake_runner,
        )
    )

    assert "Incomplete" in markdown
    assert (tmp_path / "final_review.md").read_text().strip() == markdown
    assert (tmp_path / "review_candidate_report.json").exists()
    assert (tmp_path / "review_raw_result.txt").read_text() == "raw SDK result"
    task, _run_dir, prompt, tools = calls[0]
    assert "preflight.md" in task
    assert "structured output" in prompt
    assert "human approval" in " ".join(prompt.split())
    assert "Read" in tools


def test_reviewer_failure_is_advisory(tmp_path):
    async def broken_runner(*_args, **_kwargs):
        raise RuntimeError("gateway unavailable")

    markdown = asyncio.run(
        run_preflight_review(
            "paper",
            "repo",
            tmp_path,
            runner=broken_runner,
        )
    )

    assert "allowed to continue" in markdown
    assert "gateway unavailable" in markdown
    assert (tmp_path / "preflight.md").exists()
