"""The orchestrator: wires the SDK options, agents, tools, and hooks, then runs a
single paper-reproduction job and returns the final structured report.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    SystemMessage,
    TextBlock,
)

from . import config, memory as mem
from .agents import build_agents
from .hooks import build_hooks
from .provider import ensure_runtime_path, provider_env, reasoning_model
from .schemas import final_report_json_schema
from .tools.mcp_server import (
    AMD_EXEC_TOOL_NAMES,
    AMD_PLAN_TOOL_NAMES,
    TOOL_NAMES,
    build_server,
)

_PROMPTS = Path(__file__).parent / "prompts"


@dataclass
class RunResult:
    run_dir: Path
    report: Optional[dict[str, Any]]
    raw_result_text: Optional[str]
    cost_usd: Optional[float]
    is_error: bool


def _orchestrator_prompt(amd: bool = False) -> str:
    name = "orchestrator_amd.md" if amd else "orchestrator.md"
    return (_PROMPTS / name).read_text(encoding="utf-8")


def build_options(
    run_dir: Path,
    resume: Optional[str] = None,
    stream: bool = False,
    amd: bool = False,
    execute: bool = False,
) -> ClaudeAgentOptions:
    """Assemble ClaudeAgentOptions for one reproduction run.

    ``resume`` continues a prior session by id (skips re-doing finished work).
    ``stream`` enables partial-message token streaming for live progress.
    ``amd`` adds Track-2 (ROCm enablement) agents/tools.
    ``execute`` grants docker/verify tools + relaxes plan-only hooks so the run can build
    and actually run the reproduction inside the container.
    """
    ensure_runtime_path()
    os.environ["R2R_RUN_DIR"] = str(run_dir)

    builtin = ["Read", "Grep", "Glob", "Bash", "WebFetch", "Agent", "Skill"]
    allowed = builtin + TOOL_NAMES
    if amd:
        allowed = allowed + AMD_PLAN_TOOL_NAMES
        if execute:
            allowed = allowed + AMD_EXEC_TOOL_NAMES

    max_budget = config.max_budget_usd()
    max_turns = config.max_turns()

    # Provider (AMD gateway by default) -> base URL + custom auth headers for the CLI.
    run_env = {"R2R_RUN_DIR": str(run_dir)}
    if os.environ.get("R2R_JAVA_HOME"):
        run_env["R2R_JAVA_HOME"] = os.environ["R2R_JAVA_HOME"]
    run_env.update(provider_env())

    # Persistent memory: prepend globally-applicable learnings from previous runs so
    # they are in context immediately (repo-scoped ones are pulled via recall_memory).
    system_append = _orchestrator_prompt(amd=amd)
    learned = mem.format_for_prompt(scope="global")
    if learned:
        system_append = f"{system_append}\n\n{learned}"

    return ClaudeAgentOptions(
        system_prompt={
            "type": "preset",
            "preset": "claude_code",
            "append": system_append,
        },
        cwd=str(run_dir),
        add_dirs=[str(run_dir)],
        setting_sources=["project"],  # load CLAUDE.md + .claude/skills from the project
        skills="all",  # discover the /reproduce skill (adds the Skill tool)
        mcp_servers={"r2r": build_server()},
        allowed_tools=allowed,
        agents=build_agents(amd=amd, execute=execute),
        hooks=build_hooks(execute=execute),
        permission_mode="acceptEdits",
        max_turns=max_turns,
        max_budget_usd=max_budget,
        model=reasoning_model(),
        output_format=final_report_json_schema(),
        include_partial_messages=stream,
        resume=resume,
        env=run_env,
    )


def _session_file(run_dir: Path) -> Path:
    return run_dir / "session.json"


def load_session_id(run_dir: Path) -> Optional[str]:
    """Return the saved session id for a run dir, if any."""
    f = _session_file(run_dir)
    if f.exists():
        try:
            return json.loads(f.read_text(encoding="utf-8")).get("session_id")
        except json.JSONDecodeError:
            return None
    return None


async def _run_session(
    task: str,
    run_dir: Path,
    resume: Optional[str] = None,
    stream: bool = False,
    amd: bool = False,
    execute: bool = False,
    on_text=None,
) -> RunResult:
    """Drive one ClaudeSDKClient session, capturing the report, cost, and session id."""
    options = build_options(run_dir, resume=resume, stream=stream, amd=amd, execute=execute)

    report: Optional[dict[str, Any]] = None
    raw_text: Optional[str] = None
    cost: Optional[float] = None
    is_error = False
    session_id: Optional[str] = None

    async with ClaudeSDKClient(options=options) as client:
        await client.query(task)
        async for message in client.receive_response():
            if isinstance(message, SystemMessage) and message.subtype == "init":
                session_id = message.data.get("session_id")
                if session_id:
                    _session_file(run_dir).write_text(
                        json.dumps({"session_id": session_id}, indent=2), "utf-8"
                    )
            elif isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock) and on_text:
                        on_text(block.text)
            elif isinstance(message, ResultMessage):
                cost = message.total_cost_usd
                is_error = bool(message.is_error)
                raw_text = message.result
                if getattr(message, "structured_output", None):
                    report = message.structured_output

    # Prefer the persisted report.json (the save_report tool writes it).
    saved = run_dir / "report.json"
    if report is None and saved.exists():
        try:
            report = json.loads(saved.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    if report is None and raw_text:
        report = _try_parse_json(raw_text)

    return RunResult(run_dir, report, raw_text, cost, is_error)


async def reproduce(
    paper_url: str,
    repo_url: str,
    run_dir: Path,
    target: Optional[str] = None,
    on_text=None,
    stream: bool = False,
    amd: bool = False,
    execute: bool = False,
) -> RunResult:
    """Run the full pipeline. ``on_text`` is an optional callback(str) for streaming.

    ``amd`` enables the ROCm-enablement track; ``execute`` lets it build and run on AMD.
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    # CLAUDE.md must be visible from cwd; copy the project rules into the run dir.
    project_claude = Path(__file__).parent.parent / "CLAUDE.md"
    if project_claude.exists():
        (run_dir / "CLAUDE.md").write_text(project_claude.read_text(encoding="utf-8"), "utf-8")

    if amd:
        task = (
            f"Make this repo run on AMD ROCm GPUs and reproduce one paper result on it.\n"
            f"PAPER_URL: {paper_url}\n"
            f"REPO_URL: {repo_url}\n"
            f"TARGET_EXPERIMENT: {target or '(choose the single most reproducible one)'}\n"
            f"EXECUTE_MODE: {execute} (if false, produce the AMD env plan + Dockerfile but "
            f"do not build/run).\n"
            f"Run Track 1 and Track 2 in parallel, then converge. End with the final JSON."
        )
    else:
        task = (
            f"Reproduce one result from this paper using its code.\n"
            f"PAPER_URL: {paper_url}\n"
            f"REPO_URL: {repo_url}\n"
            f"TARGET_EXPERIMENT: {target or '(choose the single most reproducible one)'}\n"
            f"Follow the orchestrator pipeline and end with the final JSON report."
        )
    return await _run_session(
        task, run_dir, stream=stream, amd=amd, execute=execute, on_text=on_text
    )


async def resume_run(
    run_dir: Path,
    instruction: str,
    on_text=None,
    stream: bool = False,
) -> RunResult:
    """Continue a previous run in its original session (no re-parsing/re-cloning).

    Useful to re-pick the experiment, re-extract config, or re-synthesize commands
    using the context already built up in the prior session.
    """
    session_id = load_session_id(run_dir)
    if not session_id:
        raise RuntimeError(
            f"No saved session found in {run_dir}. Run `r2r reproduce` first, or point "
            "--run-dir at a completed run."
        )
    task = (
        f"{instruction}\n"
        "Reuse the paper parse, repo clone, and prior analysis already in this session. "
        "Re-run only the necessary phase(s) and end with the updated final JSON report, "
        "then persist it with mcp__r2r__save_report."
    )
    return await _run_session(task, run_dir, resume=session_id, stream=stream, on_text=on_text)


def _try_parse_json(text: str) -> Optional[dict[str, Any]]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None
