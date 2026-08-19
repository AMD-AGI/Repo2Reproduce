"""Independent Markdown reviewers around the main agent pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from . import config
from .provider import ensure_runtime_path, provider_env, reasoning_model


_PROMPTS = Path(__file__).parent / "prompts"
ReviewRunner = Callable[[str, Path, str, list[str]], Awaitable[str]]


def _prompt(name: str) -> str:
    return (_PROMPTS / name).read_text(encoding="utf-8")


def _review_budget() -> float:
    try:
        return float(os.environ.get("R2R_REVIEW_MAX_BUDGET_USD", "5"))
    except ValueError:
        return 5.0


def _review_turns() -> int:
    try:
        return int(os.environ.get("R2R_REVIEW_MAX_TURNS", "60"))
    except ValueError:
        return 60


def _review_options(run_dir: Path, prompt: str, tools: list[str]) -> Any:
    from claude_agent_sdk import ClaudeAgentOptions
    from .tools.mcp_server import build_server

    ensure_runtime_path()
    os.environ["R2R_RUN_DIR"] = str(run_dir)
    run_env = {"R2R_RUN_DIR": str(run_dir)}
    if os.environ.get("R2R_JAVA_HOME"):
        run_env["R2R_JAVA_HOME"] = os.environ["R2R_JAVA_HOME"]
    run_env.update(provider_env())

    return ClaudeAgentOptions(
        system_prompt={
            "type": "preset",
            "preset": "claude_code",
            "append": prompt,
        },
        cwd=str(run_dir),
        add_dirs=[str(run_dir)],
        setting_sources=["project"],
        mcp_servers={"r2r": build_server()},
        allowed_tools=tools,
        permission_mode="acceptEdits",
        max_turns=min(config.max_turns(), _review_turns()),
        max_budget_usd=min(config.max_budget_usd(), _review_budget()),
        model=reasoning_model(),
        env=run_env,
    )


async def _run_markdown_session(
    task: str,
    run_dir: Path,
    prompt: str,
    tools: list[str],
) -> str:
    """Run one reviewer session and return its final Markdown text."""
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeSDKClient,
        ResultMessage,
        TextBlock,
    )

    options = _review_options(run_dir, prompt, tools)
    text_blocks: list[str] = []
    result_text = ""

    async with ClaudeSDKClient(options=options) as client:
        await client.query(task)
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock) and block.text.strip():
                        text_blocks.append(block.text)
            elif isinstance(message, ResultMessage):
                result_text = message.result or ""

    text = result_text.strip() or "\n".join(text_blocks).strip()
    if not text:
        raise RuntimeError("reviewer returned no Markdown")
    return text


def _failure_artifact(title: str, error: BaseException) -> str:
    return (
        f"# {title}\n\n"
        "The independent reviewer could not complete. The main pipeline was "
        "allowed to continue because this review is advisory.\n\n"
        f"Error: `{type(error).__name__}: {error}`\n"
    )


async def run_preflight_review(
    paper_url: str,
    repo_url: str,
    run_dir: Path,
    target: Optional[str] = None,
    runner: ReviewRunner = _run_markdown_session,
) -> str:
    """Write ``preflight.md`` before the main orchestrator starts."""
    task = (
        "Independently review whether this paper result is locally reproducible "
        "from its official repository.\n\n"
        f"PAPER_URL: {paper_url}\n"
        f"REPO_URL: {repo_url}\n"
        f"REQUESTED_TARGET: {target or '(choose the main locally testable claim)'}\n\n"
        "The deterministic bootstrap normally prepared paper/ and repo/ already. "
        "Inspect paper.md, paper_parse.json, the repo map, configs, "
        "data/evaluation code, and README. Call ingest_paper or clone_repo only "
        "if those artifacts are missing. Return only the requested Markdown review."
    )
    tools = [
        "Read",
        "Grep",
        "Glob",
        "WebFetch",
        "mcp__r2r__ingest_paper",
        "mcp__r2r__verify_quote_in_paper",
        "mcp__r2r__clone_repo",
        "mcp__r2r__map_repo",
        "mcp__r2r__extract_argparse",
        "mcp__r2r__verify_flags",
    ]
    try:
        markdown = await runner(task, run_dir, _prompt("preflight_verifier.md"), tools)
    except Exception as exc:
        markdown = _failure_artifact("Pre-run verification", exc)
    (run_dir / "preflight.md").write_text(markdown.rstrip() + "\n", encoding="utf-8")
    return markdown


async def run_final_review(
    run_dir: Path,
    report: Optional[dict[str, Any]],
    raw_result_text: Optional[str],
    runner: ReviewRunner = _run_markdown_session,
) -> str:
    """Write ``final_review.md`` after the main orchestrator finishes."""
    if report is not None:
        (run_dir / "review_candidate_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if raw_result_text:
        (run_dir / "review_raw_result.txt").write_text(
            raw_result_text,
            encoding="utf-8",
        )

    task = (
        "Review the completed reproduction run in this directory. Read "
        "preflight.md, paper/paper.md, the cloned repo, report.json if present, "
        "review_candidate_report.json, review_raw_result.txt, amd_run logs, "
        "agent transcripts, and any execution artifacts. Return only the "
        "requested Markdown review."
    )
    tools = [
        "Read",
        "Grep",
        "Glob",
        "mcp__r2r__verify_quote_in_paper",
        "mcp__r2r__verify_flags",
    ]
    try:
        markdown = await runner(task, run_dir, _prompt("final_verifier.md"), tools)
    except Exception as exc:
        markdown = _failure_artifact("Final verification", exc)
    (run_dir / "final_review.md").write_text(markdown.rstrip() + "\n", encoding="utf-8")
    return markdown
