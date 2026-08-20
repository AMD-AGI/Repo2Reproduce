"""Deterministic Repo2Reproduce phase runner.

Python owns the fixed pipeline order, dependencies, optional phases, persistence, and
completion. LLM workers remain autonomous *inside* one phase, but no LLM decides which
global phase runs next or whether the pipeline is finished.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Type

from claude_agent_sdk import (
    AgentDefinition,
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    SystemMessage,
    TextBlock,
)
from pydantic import BaseModel, ValidationError

from . import config
from .agents import build_agents
from .gate import report_is_complete
from .hooks import build_hooks
from .provider import ensure_runtime_path, provider_env
from .schemas import (
    AmdEnvironment,
    BaseImageSelection,
    CodeMap,
    Configuration,
    EnvironmentPort,
    ExperimentSelection,
    FinalReport,
    KernelMigration,
    PaperMap,
    ReconciliationResult,
    ReplicationPlan,
    ReproductionResult,
    VerificationResult,
)
from .reviewers import run_final_review, run_preflight_review
from .tools import amd as amd_tools
from .tools import code as code_intel
from .tools import paper as paper_io
from .tools.mcp_server import build_server


@dataclass
class RunResult:
    run_dir: Path
    report: Optional[dict[str, Any]]
    raw_result_text: Optional[str]
    cost_usd: Optional[float]
    is_error: bool


@dataclass
class WorkerOutcome:
    output: Optional[dict[str, Any]]
    raw_text: Optional[str]
    cost_usd: float
    is_error: bool
    session_id: Optional[str]
    error: Optional[str] = None


@dataclass
class PhaseRecord:
    phase: str
    status: str
    output: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    skip_reason: Optional[str] = None
    skip_kind: Optional[str] = None
    cost_usd: float = 0.0
    attempts: int = 0
    session_id: Optional[str] = None
    updated_at: str = ""

    def __post_init__(self) -> None:
        if not self.updated_at:
            self.updated_at = datetime.now(timezone.utc).isoformat()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PhaseRecord":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in data.items() if key in allowed})


def _atomic_json(path: Path, value: Any) -> None:
    """Atomically persist JSON so interruption cannot leave a truncated state file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


class PipelineStore:
    """Durable phase artifacts + one compact pipeline_state.json index."""

    def __init__(self, run_dir: Path, metadata: Optional[dict[str, Any]] = None):
        self.run_dir = run_dir
        self.phase_dir = run_dir / "phases"
        self.state_path = run_dir / "pipeline_state.json"
        self._lock = asyncio.Lock()
        if self.state_path.exists():
            try:
                self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self.state = {}
        else:
            self.state = {}
        self.state.setdefault("schema_version", 1)
        self.state.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        self.state.setdefault("phases", {})
        self.state.setdefault("status", "running")
        if metadata:
            self.state.setdefault("run", {}).update(metadata)
        self._write_state()

    def _write_state(self) -> None:
        self.state["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.state["cost_usd"] = self.total_cost()
        _atomic_json(self.state_path, self.state)

    def phase_path(self, phase: str) -> Path:
        return self.phase_dir / f"{phase}.json"

    def load(self, phase: str) -> Optional[PhaseRecord]:
        path = self.phase_path(phase)
        if not path.exists():
            return None
        try:
            return PhaseRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, TypeError):
            return None

    def reusable(self, phase: str) -> bool:
        record = self.load(phase)
        return bool(
            record
            and (
                record.status == "succeeded"
                or (record.status == "skipped" and record.skip_kind == "not_applicable")
            )
        )

    def output(self, phase: str) -> Optional[dict[str, Any]]:
        record = self.load(phase)
        return record.output if record else None

    def succeeded(self, phase: str) -> bool:
        record = self.load(phase)
        return bool(record and record.status == "succeeded")

    def terminal(self, phase: str) -> bool:
        record = self.load(phase)
        return bool(record and record.status in {"succeeded", "failed", "skipped"})

    def records(self) -> list[PhaseRecord]:
        records: list[PhaseRecord] = []
        if not self.phase_dir.exists():
            return records
        for path in sorted(self.phase_dir.glob("*.json")):
            record = self.load(path.stem)
            if record:
                records.append(record)
        return records

    def total_cost(self) -> float:
        return sum(record.cost_usd for record in self.records())

    async def save(self, record: PhaseRecord) -> PhaseRecord:
        async with self._lock:
            record.updated_at = datetime.now(timezone.utc).isoformat()
            _atomic_json(self.phase_path(record.phase), asdict(record))
            self.state["phases"][record.phase] = {
                "status": record.status,
                "artifact": str(self.phase_path(record.phase)),
                "error": record.error,
                "skip_reason": record.skip_reason,
                "cost_usd": record.cost_usd,
                "attempts": record.attempts,
                "updated_at": record.updated_at,
            }
            self._write_state()
        return record

    async def finish(self, status: str, missing: Optional[list[str]] = None) -> None:
        async with self._lock:
            self.state["status"] = status
            self.state["missing"] = missing or []
            self._write_state()


def _safe_label(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-") or "phase"


def capture_transcript(
    run_dir: Path, session_id: Optional[str], phase: str = "orchestrator"
) -> Optional[Path]:
    """Copy one SDK transcript under ``agent_transcript/<phase>/`` (best effort)."""
    if not session_id:
        return None
    try:
        projects = Path.home() / ".claude" / "projects"
        matches = list(projects.glob(f"*/{session_id}.jsonl"))
        if not matches:
            return None
        src_main = matches[0]
        dest = run_dir / "agent_transcript" / _safe_label(phase)
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_main, dest / "session.jsonl")
        sub_dir = src_main.parent / session_id / "subagents"
        if sub_dir.is_dir():
            shutil.copytree(sub_dir, dest / "subagents", dirs_exist_ok=True)
        return dest
    except Exception:
        return None


def _try_parse_json(text: str) -> Optional[dict[str, Any]]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if 0 <= start < end:
            try:
                value = json.loads(text[start : end + 1])
                return value if isinstance(value, dict) else None
            except json.JSONDecodeError:
                return None
    return None


def _worker_env(run_dir: Path) -> dict[str, str]:
    env = {"R2R_RUN_DIR": str(run_dir)}
    if os.environ.get("R2R_JAVA_HOME"):
        env["R2R_JAVA_HOME"] = os.environ["R2R_JAVA_HOME"]
    env.update(provider_env())
    return env


async def _run_worker_session(
    *,
    phase: str,
    definition: AgentDefinition,
    task: str,
    schema_model: Type[BaseModel],
    run_dir: Path,
    amd: bool,
    execute: bool,
    max_budget_usd: float,
    max_turns: int,
    on_text=None,
    stream: bool = False,
) -> WorkerOutcome:
    """Run one specialized worker as its own top-level SDK session."""
    ensure_runtime_path()
    options = ClaudeAgentOptions(
        system_prompt={
            "type": "preset",
            "preset": "claude_code",
            "append": definition.prompt,
        },
        cwd=str(run_dir),
        add_dirs=[str(run_dir)],
        setting_sources=["project"],
        mcp_servers={"r2r": build_server()},
        allowed_tools=list(definition.tools or []),
        hooks=build_hooks(execute=execute, amd=amd),
        permission_mode="acceptEdits",
        max_turns=max_turns,
        max_budget_usd=max_budget_usd,
        model=definition.model,
        output_format={"type": "json_schema", "schema": schema_model.model_json_schema()},
        include_partial_messages=stream,
        env=_worker_env(run_dir),
    )

    structured: Optional[dict[str, Any]] = None
    raw_text: Optional[str] = None
    cost = 0.0
    is_error = False
    session_id: Optional[str] = None

    async with ClaudeSDKClient(options=options) as client:
        await client.query(task)
        async for message in client.receive_response():
            if isinstance(message, SystemMessage) and message.subtype == "init":
                session_id = message.data.get("session_id")
            elif isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock) and on_text:
                        on_text(f"\n[{phase}] {block.text}")
            elif isinstance(message, ResultMessage):
                cost = float(message.total_cost_usd or 0.0)
                is_error = bool(message.is_error)
                raw_text = message.result
                candidate = getattr(message, "structured_output", None)
                if isinstance(candidate, dict):
                    structured = candidate

    capture_transcript(run_dir, session_id, phase)
    candidate = structured or (_try_parse_json(raw_text) if raw_text else None)
    if candidate is None:
        return WorkerOutcome(
            None,
            raw_text,
            cost,
            True,
            session_id,
            error="worker produced no valid structured JSON",
        )
    try:
        normalized = schema_model.model_validate(candidate).model_dump(mode="json")
    except ValidationError as exc:
        return WorkerOutcome(
            candidate,
            raw_text,
            cost,
            True,
            session_id,
            error=f"worker output failed schema validation: {exc}",
        )
    return WorkerOutcome(
        normalized,
        raw_text,
        cost,
        is_error,
        session_id,
        error="SDK marked the worker result as an error" if is_error else None,
    )


async def _bootstrap_operation(phase: str, run_dir: Path, value: str) -> dict[str, Any]:
    """Run deterministic non-LLM prerequisites."""
    if phase == "ingest-paper":
        parse_json = run_dir / "paper" / "paper_parse.json"
        paper_md = run_dir / "paper" / "paper.md"
        if parse_json.exists() and paper_md.exists():
            return {
                "parse_json_path": str(parse_json),
                "markdown_path": str(paper_md),
                "reused": True,
            }

        def _ingest() -> dict[str, Any]:
            pdf = paper_io.download_pdf(value, run_dir / "paper" / "paper.pdf")
            parsed = paper_io.parse_paper(pdf, run_dir / "paper")
            return {
                "pdf_path": parsed.pdf_path,
                "parse_json_path": parsed.parse_json_path,
                "markdown_path": parsed.merged_markdown_path,
                "reused": False,
            }

        return await asyncio.to_thread(_ingest)

    if phase == "clone-repo":
        dest = run_dir / "repo"
        if dest.is_dir() and any(dest.iterdir()):
            return {"repo_path": str(dest), "reused": True}
        result = await asyncio.to_thread(code_intel.clone_repo, value, dest)
        return {**result, "repo_path": str(dest), "reused": False}

    raise ValueError(f"unknown bootstrap phase: {phase}")


async def _run_bootstrap_phase(store: PipelineStore, phase: str, value: str) -> PhaseRecord:
    if store.reusable(phase):
        return store.load(phase)  # type: ignore[return-value]
    existing = store.load(phase)
    prior_cost = existing.cost_usd if existing else 0.0
    prior_attempts = existing.attempts if existing else 0
    await store.save(
        PhaseRecord(
            phase=phase,
            status="running",
            cost_usd=prior_cost,
            attempts=prior_attempts + 1,
        )
    )
    try:
        output = await _bootstrap_operation(phase, store.run_dir, value)
        record = PhaseRecord(
            phase=phase,
            status="succeeded",
            output=output,
            cost_usd=prior_cost,
            attempts=prior_attempts + 1,
        )
    except Exception as exc:
        record = PhaseRecord(
            phase=phase,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            cost_usd=prior_cost,
            attempts=prior_attempts + 1,
        )
    return await store.save(record)


async def _skip_phase(
    store: PipelineStore,
    phase: str,
    reason: str,
    *,
    kind: str = "blocked",
    output: Optional[dict[str, Any]] = None,
) -> PhaseRecord:
    if store.reusable(phase):
        return store.load(phase)  # type: ignore[return-value]
    existing = store.load(phase)
    return await store.save(
        PhaseRecord(
            phase=phase,
            status="skipped",
            output=output,
            skip_reason=reason,
            skip_kind=kind,
            cost_usd=existing.cost_usd if existing else 0.0,
            attempts=existing.attempts if existing else 0,
        )
    )


async def _run_worker_phase(
    *,
    store: PipelineStore,
    phase: str,
    definition: AgentDefinition,
    task: str,
    schema_model: Type[BaseModel],
    amd: bool,
    execute: bool,
    phase_budget: float,
    phase_turns: int,
    on_text=None,
    stream: bool = False,
    max_attempts: int = 2,
) -> PhaseRecord:
    """Run/retry one worker; convert every exception into a durable failed record."""
    if store.reusable(phase):
        return store.load(phase)  # type: ignore[return-value]

    existing = store.load(phase)
    historical_cost = existing.cost_usd if existing else 0.0
    historical_attempts = existing.attempts if existing else 0
    new_cost = 0.0
    last_error = "worker did not run"
    last_output: Optional[dict[str, Any]] = None
    session_id: Optional[str] = None

    await store.save(
        PhaseRecord(
            phase=phase,
            status="running",
            cost_usd=historical_cost,
            attempts=historical_attempts,
        )
    )

    for attempt in range(1, max_attempts + 1):
        remaining = phase_budget - new_cost
        if remaining <= 0:
            last_error = "phase budget exhausted before a valid output was produced"
            break
        try:
            outcome = await _run_worker_session(
                phase=phase,
                definition=definition,
                task=task,
                schema_model=schema_model,
                run_dir=store.run_dir,
                amd=amd,
                execute=execute,
                max_budget_usd=remaining,
                max_turns=phase_turns,
                on_text=on_text,
                stream=stream,
            )
            new_cost += outcome.cost_usd
            last_output = outcome.output
            session_id = outcome.session_id
            if outcome.output is not None and not outcome.is_error:
                semantic_error = _semantic_phase_error(
                    phase,
                    outcome.output,
                    execute=execute,
                )
                if semantic_error:
                    last_error = semantic_error
                    continue
                return await store.save(
                    PhaseRecord(
                        phase=phase,
                        status="succeeded",
                        output=outcome.output,
                        cost_usd=historical_cost + new_cost,
                        attempts=historical_attempts + attempt,
                        session_id=session_id,
                    )
                )
            last_error = outcome.error or "worker returned an error"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"

    return await store.save(
        PhaseRecord(
            phase=phase,
            status="failed",
            output=last_output,
            error=last_error,
            cost_usd=historical_cost + new_cost,
            attempts=historical_attempts + max_attempts,
            session_id=session_id,
        )
    )


def _semantic_phase_error(
    phase: str,
    output: dict[str, Any],
    *,
    execute: bool,
) -> Optional[str]:
    """Reject schema-valid outputs that contradict the active run mode."""
    if phase == "build-verifier" and execute:
        status = str(output.get("build_status") or "")
        if status == "not_attempted":
            return (
                "execute mode cannot accept build_status='not_attempted'; "
                "return verified, build_failed, or integrity_failed with evidence"
            )
    return None


def _phase_task(
    phase: str,
    *,
    target: Optional[str],
    execute: bool,
    extra_instruction: Optional[str],
) -> str:
    """Give a worker explicit durable inputs without embedding large maps in context."""
    inputs: dict[str, list[str]] = {
        "paper-analyst": ["paper/paper.md", "paper/paper_parse.json"],
        "code-cartographer": ["repo/"],
        "base-image-selector": ["repo/", "phases/code-cartographer.json"],
        "experiment-selector": [
            "preflight.md",
            "phases/paper-analyst.json",
            "phases/code-cartographer.json",
        ],
        "env-porter": [
            "repo/",
            "phases/code-cartographer.json",
            "phases/base-image-selector.json",
        ],
        "kernel-migrator": ["repo/", "phases/base-image-selector.json"],
        "config-extractor": [
            "preflight.md",
            "phases/experiment-selector.json",
            "phases/paper-analyst.json",
            "phases/code-cartographer.json",
        ],
        "command-synthesizer": [
            "preflight.md",
            "phases/experiment-selector.json",
            "phases/config-extractor.json",
            "phases/code-cartographer.json",
        ],
        "build-verifier": [
            "preflight.md",
            "phases/base-image-selector.json",
            "phases/env-porter.json",
            "phases/kernel-migrator.json",
            "phases/command-synthesizer.json",
        ],
        "reconciler": [
            "preflight.md",
            "phases/experiment-selector.json",
            "phases/config-extractor.json",
            "phases/command-synthesizer.json",
            "phases/build-verifier.json",
        ],
        "executor": [
            "preflight.md",
            "phases/experiment-selector.json",
            "phases/build-verifier.json",
            "phases/reconciler.json",
        ],
        "verifier": [
            "preflight.md",
            "phases/experiment-selector.json",
            "phases/config-extractor.json",
            "phases/command-synthesizer.json",
            "phases/build-verifier.json",
            "phases/reconciler.json",
            "phases/executor.json",
        ],
    }
    lines = [
        f"Execute ONLY the '{phase}' phase.",
        "Read each listed phase artifact and use the value under its `output` field.",
        "Required inputs:",
        *[f"- {path}" for path in inputs.get(phase, [])],
    ]
    if phase == "experiment-selector":
        lines.append(f"Requested target: {target or '(choose the most reproducible result)'}")
    if phase == "build-verifier" and not execute:
        lines.extend(
            [
                "PLAN-ONLY MODE: create the AMD Dockerfile/environment plan, but do not "
                "build or run it.",
                "Return Output 4 with build_status='not_attempted'.",
            ]
        )
    if extra_instruction:
        lines.append(f"Additional resume instruction: {extra_instruction}")
    lines.append("Return ONLY the JSON object required by your system prompt.")
    return "\n".join(lines)


_PHASE_MODELS: dict[str, Type[BaseModel]] = {
    "paper-analyst": PaperMap,
    "code-cartographer": CodeMap,
    "base-image-selector": BaseImageSelection,
    "experiment-selector": ExperimentSelection,
    "env-porter": EnvironmentPort,
    "kernel-migrator": KernelMigration,
    "config-extractor": Configuration,
    "command-synthesizer": ReplicationPlan,
    "build-verifier": AmdEnvironment,
    "reconciler": ReconciliationResult,
    "executor": ReproductionResult,
    "verifier": VerificationResult,
}


def _planned_worker_phases(amd: bool, execute: bool) -> list[str]:
    phases = [
        "paper-analyst",
        "code-cartographer",
        "experiment-selector",
        "config-extractor",
        "command-synthesizer",
        "verifier",
    ]
    if amd:
        phases.extend(
            [
                "base-image-selector",
                "env-porter",
                "kernel-migrator",
                "build-verifier",
                "reconciler",
            ]
        )
        if execute:
            phases.append("executor")
    return phases


def _phase_limits(
    store: PipelineStore, planned: list[str], total_budget: float, total_turns: int
) -> tuple[float, int]:
    remaining_phases = [phase for phase in planned if not store.reusable(phase)]
    remaining_budget = max(0.0, total_budget - store.total_cost())
    budget = remaining_budget / max(1, len(remaining_phases))
    turns = max(10, total_turns // max(1, len(planned)))
    return budget, turns


async def _run_worker_stage(
    *,
    store: PipelineStore,
    phases: list[str],
    agents: dict[str, AgentDefinition],
    planned: list[str],
    target: Optional[str],
    amd: bool,
    execute: bool,
    total_budget: float,
    total_turns: int,
    extra_instruction: Optional[str],
    on_text,
    stream: bool,
) -> None:
    """Run independent workers concurrently; each catches/persists its own failure."""
    if not phases:
        return
    phase_budget, phase_turns = _phase_limits(store, planned, total_budget, total_turns)
    tasks = [
        _run_worker_phase(
            store=store,
            phase=phase,
            definition=agents[phase],
            task=_phase_task(
                phase,
                target=target,
                execute=execute,
                extra_instruction=extra_instruction,
            ),
            schema_model=_PHASE_MODELS[phase],
            amd=amd,
            execute=execute,
            phase_budget=phase_budget,
            phase_turns=phase_turns,
            on_text=on_text,
            stream=stream,
        )
        for phase in phases
        if not store.reusable(phase)
    ]
    if tasks:
        # Python 3.10 compatibility. Each task converts its own exceptions into a durable
        # failed PhaseRecord, so one worker cannot cancel unrelated siblings.
        await asyncio.gather(*tasks)


async def _require_or_skip(
    store: PipelineStore,
    phase: str,
    prerequisites: list[str],
    reason_prefix: str,
) -> bool:
    missing = [name for name in prerequisites if not store.succeeded(name)]
    if not missing:
        return True
    await _skip_phase(
        store,
        phase,
        f"{reason_prefix}; prerequisites not successful: {', '.join(missing)}",
        kind="blocked",
    )
    return False


def _phase_failure_gaps(store: PipelineStore) -> list[str]:
    gaps: list[str] = []
    for record in store.records():
        if record.status == "failed":
            gaps.append(f"Phase {record.phase} failed: {record.error or 'unknown error'}")
        elif record.status == "skipped" and record.skip_reason:
            gaps.append(f"Phase {record.phase} skipped: {record.skip_reason}")
    return gaps


def _assemble_report(
    store: PipelineStore,
    *,
    paper_url: str,
    repo_url: str,
    amd: bool,
    execute: bool,
) -> tuple[dict[str, Any], bool, list[str]]:
    paper_map = store.output("paper-analyst") or {}
    verifier = store.output("verifier") or {}
    configuration = store.output("config-extractor")
    commands = store.output("command-synthesizer")

    reconciliation = store.output("reconciler") or {}
    if reconciliation:
        updated2 = reconciliation.get("updated_output2")
        if isinstance(updated2, dict):
            configuration = updated2
        updated3 = reconciliation.get("updated_output3")
        if isinstance(updated3, dict):
            commands = updated3

    report: dict[str, Any] = {
        "paper_url": paper_url,
        "repo_url": repo_url,
        "paper_title": paper_map.get("title"),
        "gaps_and_assumptions": list(verifier.get("gaps_and_assumptions") or [])
        + _phase_failure_gaps(store),
        "confidence": verifier.get("confidence", "low"),
        "confidence_rationale": verifier.get("confidence_rationale")
        or "One or more pipeline phases did not produce a verifier result.",
    }
    if store.output("experiment-selector") is not None:
        report["output1_experiment_selection"] = store.output("experiment-selector")
    if configuration is not None:
        report["output2_configuration"] = configuration
    if commands is not None:
        report["output3_replication_plan"] = commands
    if amd and store.output("build-verifier") is not None:
        report["output4_amd_environment"] = store.output("build-verifier")
    if execute and store.output("executor") is not None:
        report["output5_reproduction_result"] = store.output("executor")

    ok, missing = report_is_complete(report, amd=amd, execute=execute)
    if ok:
        try:
            report = FinalReport.model_validate(report).model_dump(mode="json")
        except ValidationError as exc:
            ok = False
            missing = [f"FinalReport schema validation failed: {exc}"]
    return report, ok, missing


async def _run_pipeline(
    *,
    paper_url: str,
    repo_url: str,
    run_dir: Path,
    target: Optional[str],
    amd: bool,
    execute: bool,
    allow_scope_reduction: bool = False,
    on_text,
    stream: bool,
    extra_instruction: Optional[str] = None,
    enable_reviewers: bool = True,
) -> RunResult:
    metadata = {
        "paper_url": paper_url,
        "repo_url": repo_url,
        "target": target,
        "amd": amd,
        "execute": execute,
        "allow_scope_reduction": allow_scope_reduction,
    }
    store = PipelineStore(run_dir, metadata)
    agents = build_agents(amd=amd, execute=execute)
    planned = _planned_worker_phases(amd, execute)
    total_budget = config.max_budget_usd()
    total_turns = config.max_turns()

    # Deterministic prerequisites.
    await asyncio.gather(
        _run_bootstrap_phase(store, "ingest-paper", paper_url),
        _run_bootstrap_phase(store, "clone-repo", repo_url),
    )

    if enable_reviewers and not (run_dir / "preflight.md").exists():
        await run_preflight_review(paper_url, repo_url, run_dir, target=target)

    scope_instruction = (
        f"SCOPE_REDUCTION_APPROVED: {allow_scope_reduction}. "
        + (
            "Any reduction must be disclosed and cannot be presented as full paper parity."
            if allow_scope_reduction
            else "Do not reduce samples, epochs, seeds, comparison arms, or evaluation "
            "scope. Report that the full protocol could not run instead."
        )
    )
    extra_instruction = "\n".join(
        item for item in (extra_instruction, scope_instruction) if item
    )

    # First independent analysis wave.
    first: list[str] = []
    if store.succeeded("ingest-paper"):
        first.append("paper-analyst")
    else:
        await _skip_phase(store, "paper-analyst", "paper ingestion failed", kind="blocked")
    if store.succeeded("clone-repo"):
        first.append("code-cartographer")
        if amd:
            first.append("base-image-selector")
    else:
        await _skip_phase(store, "code-cartographer", "repository clone failed", kind="blocked")
        if amd:
            await _skip_phase(
                store, "base-image-selector", "repository clone failed", kind="blocked"
            )
    await _run_worker_stage(
        store=store,
        phases=first,
        agents=agents,
        planned=planned,
        target=target,
        amd=amd,
        execute=execute,
        total_budget=total_budget,
        total_turns=total_turns,
        extra_instruction=extra_instruction,
        on_text=on_text,
        stream=stream,
    )

    # Experiment selection can run alongside AMD dependency/kernel planning.
    second: list[str] = []
    if await _require_or_skip(
        store,
        "experiment-selector",
        ["paper-analyst", "code-cartographer"],
        "cannot select a grounded experiment",
    ):
        second.append("experiment-selector")
    if amd:
        if await _require_or_skip(
            store,
            "env-porter",
            ["code-cartographer", "base-image-selector"],
            "cannot plan the ROCm environment",
        ):
            second.append("env-porter")
        if store.succeeded("code-cartographer") and store.succeeded("base-image-selector"):
            try:
                detection = await asyncio.to_thread(amd_tools.detect_cuda_kernels, run_dir / "repo")
            except Exception:
                detection = {"has_custom_kernel": True}
            if detection.get("has_custom_kernel"):
                second.append("kernel-migrator")
            else:
                await _skip_phase(
                    store,
                    "kernel-migrator",
                    "repository contains no custom CUDA kernel",
                    kind="not_applicable",
                    output={
                        "has_custom_kernel": False,
                        "sources": [],
                        "note": "No custom CUDA kernel was detected; HIP migration is unnecessary.",
                    },
                )
        else:
            await _skip_phase(
                store,
                "kernel-migrator",
                "repository analysis or base-image selection failed",
                kind="blocked",
            )
    await _run_worker_stage(
        store=store,
        phases=second,
        agents=agents,
        planned=planned,
        target=target,
        amd=amd,
        execute=execute,
        total_budget=total_budget,
        total_turns=total_turns,
        extra_instruction=extra_instruction,
        on_text=on_text,
        stream=stream,
    )

    # Track 1 is deliberately sequential where outputs are true dependencies.
    if await _require_or_skip(
        store,
        "config-extractor",
        ["experiment-selector", "paper-analyst", "code-cartographer"],
        "cannot extract the selected experiment configuration",
    ):
        await _run_worker_stage(
            store=store,
            phases=["config-extractor"],
            agents=agents,
            planned=planned,
            target=target,
            amd=amd,
            execute=execute,
            total_budget=total_budget,
            total_turns=total_turns,
            extra_instruction=extra_instruction,
            on_text=on_text,
            stream=stream,
        )
    if await _require_or_skip(
        store,
        "command-synthesizer",
        ["experiment-selector", "config-extractor", "code-cartographer"],
        "cannot synthesize grounded commands",
    ):
        await _run_worker_stage(
            store=store,
            phases=["command-synthesizer"],
            agents=agents,
            planned=planned,
            target=target,
            amd=amd,
            execute=execute,
            total_budget=total_budget,
            total_turns=total_turns,
            extra_instruction=extra_instruction,
            on_text=on_text,
            stream=stream,
        )

    if amd:
        build_prereqs = [
            "base-image-selector",
            "env-porter",
            "command-synthesizer",
        ]
        kernel = store.load("kernel-migrator")
        kernel_ready = bool(
            kernel
            and (
                kernel.status == "succeeded"
                or (kernel.status == "skipped" and kernel.skip_kind == "not_applicable")
            )
        )
        if all(store.succeeded(name) for name in build_prereqs) and kernel_ready:
            await _run_worker_stage(
                store=store,
                phases=["build-verifier"],
                agents=agents,
                planned=planned,
                target=target,
                amd=amd,
                execute=execute,
                total_budget=total_budget,
                total_turns=total_turns,
                extra_instruction=extra_instruction,
                on_text=on_text,
                stream=stream,
            )
        else:
            await _skip_phase(
                store,
                "build-verifier",
                "AMD environment prerequisites or replication commands failed",
                kind="blocked",
            )

        build_status = (store.output("build-verifier") or {}).get("build_status")
        reconciliation_ready = build_status in {"verified", "not_attempted"}
        prerequisites_ready = all(
            store.succeeded(name)
            for name in [
                "experiment-selector",
                "config-extractor",
                "command-synthesizer",
                "build-verifier",
            ]
        )
        if reconciliation_ready and prerequisites_ready:
            await _run_worker_stage(
                store=store,
                phases=["reconciler"],
                agents=agents,
                planned=planned,
                target=target,
                amd=amd,
                execute=execute,
                total_budget=total_budget,
                total_turns=total_turns,
                extra_instruction=extra_instruction,
                on_text=on_text,
                stream=stream,
            )
        else:
            await _skip_phase(
                store,
                "reconciler",
                "AMD environment did not reach a state that can be reconciled "
                f"(build_status={build_status or 'missing'})",
                kind="blocked",
            )

    if execute:
        build = store.output("build-verifier") or {}
        if build.get("build_status") == "verified" and store.succeeded("reconciler"):
            await _run_worker_stage(
                store=store,
                phases=["executor"],
                agents=agents,
                planned=planned,
                target=target,
                amd=amd,
                execute=execute,
                total_budget=total_budget,
                total_turns=total_turns,
                extra_instruction=extra_instruction,
                on_text=on_text,
                stream=stream,
            )
        elif build.get("build_status") in {"build_failed", "integrity_failed"}:
            await _skip_phase(
                store,
                "executor",
                f"AMD build did not verify ({build.get('build_status')})",
                kind="blocked",
                output={
                    "executed": False,
                    "paper_target": (store.output("experiment-selector") or {}).get(
                        "target_result"
                    ),
                    "notes": "Execution was not attempted because the AMD build/verify phase failed.",
                },
            )
        else:
            await _skip_phase(
                store,
                "executor",
                "AMD environment was not verified or reconciliation failed",
                kind="blocked",
            )

    verify_prereqs = [
        "experiment-selector",
        "config-extractor",
        "command-synthesizer",
    ]
    if all(store.succeeded(name) for name in verify_prereqs):
        await _run_worker_stage(
            store=store,
            phases=["verifier"],
            agents=agents,
            planned=planned,
            target=target,
            amd=amd,
            execute=execute,
            total_budget=total_budget,
            total_turns=total_turns,
            extra_instruction=extra_instruction,
            on_text=on_text,
            stream=stream,
        )
    else:
        await _skip_phase(
            store,
            "verifier",
            "one or more grounded Track-1 outputs are unavailable",
            kind="blocked",
        )

    report, ok, missing = _assemble_report(
        store,
        paper_url=paper_url,
        repo_url=repo_url,
        amd=amd,
        execute=execute,
    )
    _atomic_json(run_dir / "draft_report.json", report)
    if ok:
        _atomic_json(run_dir / "report.json", report)
    else:
        (run_dir / "report.json").unlink(missing_ok=True)
    if enable_reviewers:
        await run_final_review(
            run_dir,
            report,
            None if ok else "; ".join(missing),
        )
    await store.finish("completed" if ok else "failed", missing)
    return RunResult(
        run_dir=run_dir,
        report=report,
        raw_result_text=None if ok else "; ".join(missing),
        cost_usd=store.total_cost(),
        is_error=not ok,
    )


def _copy_project_rules(run_dir: Path) -> None:
    project_claude = Path(__file__).parent.parent / "CLAUDE.md"
    if project_claude.exists():
        (run_dir / "CLAUDE.md").write_text(
            project_claude.read_text(encoding="utf-8"), encoding="utf-8"
        )


async def reproduce(
    paper_url: str,
    repo_url: str,
    run_dir: Path,
    target: Optional[str] = None,
    on_text=None,
    stream: bool = False,
    amd: bool = False,
    execute: bool = False,
    allow_scope_reduction: bool = False,
) -> RunResult:
    """Run the deterministic phase graph."""
    run_dir.mkdir(parents=True, exist_ok=True)
    _copy_project_rules(run_dir)
    os.environ["R2R_RUN_DIR"] = str(run_dir)
    return await _run_pipeline(
        paper_url=paper_url,
        repo_url=repo_url,
        run_dir=run_dir,
        target=target,
        amd=amd,
        execute=execute,
        allow_scope_reduction=allow_scope_reduction,
        on_text=on_text,
        stream=stream,
    )


def load_pipeline_state(run_dir: Path) -> Optional[dict[str, Any]]:
    path = run_dir / "pipeline_state.json"
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def load_session_id(run_dir: Path) -> Optional[str]:
    """Backward-compatible legacy session lookup."""
    path = run_dir / "session.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("session_id")
    except (OSError, json.JSONDecodeError):
        return None


async def resume_run(
    run_dir: Path,
    instruction: str = "",
    on_text=None,
    stream: bool = False,
) -> RunResult:
    """Resume failed/missing phases from durable phase artifacts."""
    state = load_pipeline_state(run_dir)
    if not state:
        raise RuntimeError(
            f"No pipeline_state.json found in {run_dir}. Start a new `r2r reproduce` run."
        )
    metadata = state.get("run") or {}
    paper_url = metadata.get("paper_url")
    repo_url = metadata.get("repo_url")
    if not paper_url or not repo_url:
        raise RuntimeError(f"Run metadata is incomplete in {run_dir / 'pipeline_state.json'}")
    _copy_project_rules(run_dir)
    os.environ["R2R_RUN_DIR"] = str(run_dir)
    return await _run_pipeline(
        paper_url=paper_url,
        repo_url=repo_url,
        run_dir=run_dir,
        target=metadata.get("target"),
        amd=bool(metadata.get("amd")),
        execute=bool(metadata.get("execute")),
        allow_scope_reduction=bool(metadata.get("allow_scope_reduction")),
        on_text=on_text,
        stream=stream,
        extra_instruction=instruction or None,
    )
