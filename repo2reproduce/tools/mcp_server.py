"""In-process MCP server exposing Repo2Reproduce's custom tools to the agents.

Tool names become ``mcp__r2r__<tool>``. The orchestrator passes ``RUN_DIR`` via env so
tools know where the per-run artifacts live.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from claude_agent_sdk import create_sdk_mcp_server, tool

from . import amd as amd_tools
from . import code as code_intel
from . import paper as paper_io
from .. import amd_knowledge, memory as mem


def _run_dir() -> Path:
    return Path(os.environ.get("R2R_RUN_DIR", "runs/_default"))


def _ok(text: str, **extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"content": [{"type": "text", "text": text}]}
    out.update(extra)
    return out


def _err(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "is_error": True}


# --------------------------------------------------------------------------- #
# Ingestion + paper parsing (STRICT: OpenDataLoader PDF only)
# --------------------------------------------------------------------------- #
@tool(
    "ingest_paper",
    "Download a paper PDF from a URL (arXiv-aware) and parse it with OpenDataLoader PDF "
    "(the only permitted engine; deterministic, no torch). Uses tables (cluster), "
    "reading order, page separators, and figure extraction. Produces paper.md and "
    "paper_parse.json in the run dir. Returns the artifact paths and counts.",
    {"url": str},
)
async def ingest_paper(args: dict[str, Any]) -> dict[str, Any]:
    run = _run_dir()

    def _ingest():
        pdf = paper_io.download_pdf(args["url"], run / "paper" / "paper.pdf")
        return paper_io.parse_paper(pdf, run / "paper")

    try:
        # Offloaded to a thread: download + the Java OpenDataLoader parse are blocking.
        parsed = await asyncio.to_thread(_ingest)
    except Exception as e:  # keep the loop alive
        return _err(f"Paper ingestion failed: {e}")
    return _ok(
        json.dumps(
            {
                "pdf_path": parsed.pdf_path,
                "markdown_path": parsed.merged_markdown_path,
                "parse_json_path": parsed.parse_json_path,
                "num_blocks": len(parsed.blocks),
                "num_tables": len(parsed.tables),
                "num_headings": len(parsed.headings),
                "artifacts": parsed.opendataloader,
            },
            indent=2,
        )
    )


@tool(
    "verify_quote_in_paper",
    "Verify that a candidate quote appears VERBATIM (whitespace-insensitive) in the "
    "parsed paper, and return its page locator. Use this before putting any quote in a "
    "final output.",
    {"quote": str},
)
async def verify_quote_in_paper(args: dict[str, Any]) -> dict[str, Any]:
    parse_json = _run_dir() / "paper" / "paper_parse.json"
    if not parse_json.exists():
        return _err("paper_parse.json not found. Run ingest_paper first.")
    res = paper_io.find_quote(parse_json, args["quote"])
    return _ok(json.dumps(res, indent=2))


# --------------------------------------------------------------------------- #
# Code intelligence
# --------------------------------------------------------------------------- #
@tool(
    "clone_repo",
    "Shallow-clone a public GitHub repository into the run dir so it can be inspected.",
    {"repo_url": str},
)
async def clone_repo(args: dict[str, Any]) -> dict[str, Any]:
    dest = _run_dir() / "repo"
    try:
        # Offloaded to a thread: git clone is a blocking subprocess.
        meta = await asyncio.to_thread(code_intel.clone_repo, args["repo_url"], dest)
    except Exception as e:
        return _err(f"Clone failed: {e}")
    return _ok(json.dumps(meta, indent=2), structuredContent=meta)


@tool(
    "map_repo",
    "Map the cloned repo: likely entry-point scripts, config files, and README command "
    "blocks. Returns a structured overview to guide deeper reading.",
    {},
)
async def map_repo(args: dict[str, Any]) -> dict[str, Any]:
    root = _run_dir() / "repo"
    if not root.exists():
        return _err("Repo not cloned yet. Run clone_repo first.")
    overview = {
        **code_intel.find_entry_points(root),
        **code_intel.scan_configs(root),
        **code_intel.find_readme_commands(root),
    }
    return _ok(json.dumps(overview, indent=2))


@tool(
    "extract_argparse",
    "Statically extract the REAL argparse CLI flags (names, defaults, types, choices) "
    "from a Python file in the repo. Use this to ground every command flag. Path is "
    "relative to the cloned repo root.",
    {"file_path": str},
)
async def extract_argparse(args: dict[str, Any]) -> dict[str, Any]:
    p = (_run_dir() / "repo" / args["file_path"]).resolve()
    if not p.exists():
        return _err(f"File not found in repo: {args['file_path']}")
    return _ok(json.dumps(code_intel.extract_argparse(p), indent=2))


@tool(
    "verify_flags",
    "Verify that a list of CLI flags actually exist in a given repo Python file's "
    "argparse definition. Use before finalizing any training/eval command.",
    {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "repo-relative path"},
            "flags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["file_path", "flags"],
    },
)
async def verify_flags(args: dict[str, Any]) -> dict[str, Any]:
    p = (_run_dir() / "repo" / args["file_path"]).resolve()
    if not p.exists():
        return _err(f"File not found in repo: {args['file_path']}")
    return _ok(json.dumps(code_intel.verify_flags(p, args["flags"]), indent=2))


# --------------------------------------------------------------------------- #
# Report persistence
# --------------------------------------------------------------------------- #
@tool(
    "save_report",
    "Persist the final structured report (the three outputs) as report.json in the run "
    "dir. Pass the report object as a JSON string.",
    {"report_json": str},
)
async def save_report(args: dict[str, Any]) -> dict[str, Any]:
    run = _run_dir()
    run.mkdir(parents=True, exist_ok=True)
    try:
        obj = json.loads(args["report_json"])
    except json.JSONDecodeError as e:
        return _err(f"report_json is not valid JSON: {e}")
    (run / "report.json").write_text(json.dumps(obj, indent=2, ensure_ascii=False), "utf-8")
    return _ok(f"Saved report to {run / 'report.json'}")


# --------------------------------------------------------------------------- #
# Persistent cross-run memory
# --------------------------------------------------------------------------- #
@tool(
    "remember",
    "Store a DURABLE learning that should help future reproduction runs (persists "
    "across runs and papers). Use for reusable, non-obvious facts: environment gotchas, "
    "repo conventions, parser quirks, dataset/version fixes. Keep it one crisp sentence. "
    "Optionally set tags and a scope ('global' = always applies, or a repo URL to scope "
    "it to that project).",
    {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The learning, one crisp sentence."},
            "tags": {"type": "array", "items": {"type": "string"}},
            "scope": {"type": "string", "description": "'global' or a repo URL."},
        },
        "required": ["text"],
    },
)
async def remember(args: dict[str, Any]) -> dict[str, Any]:
    res = mem.add_memory(
        text=args["text"],
        tags=args.get("tags"),
        scope=args.get("scope", "global"),
        source_run=str(_run_dir()),
    )
    return _ok(json.dumps(res))


@tool(
    "recall_memory",
    "Recall durable learnings from previous runs. Call this EARLY so you can apply past "
    "lessons. Optionally filter by a query substring, tags, or scope (a repo URL; global "
    "learnings are always included).",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "scope": {"type": "string", "description": "repo URL to include its scoped learnings"},
        },
    },
)
async def recall_memory(args: dict[str, Any]) -> dict[str, Any]:
    hits = mem.recall(
        query=args.get("query"),
        tags=args.get("tags"),
        scope=args.get("scope"),
    )
    if not hits:
        return _ok("(no learnings stored yet)")
    return _ok(
        json.dumps(
            [{"text": h["text"], "tags": h.get("tags", []), "scope": h.get("scope")} for h in hits],
            indent=2,
        )
    )


# --------------------------------------------------------------------------- #
# AMD / ROCm enablement (Track 2)
# --------------------------------------------------------------------------- #
@tool(
    "recall_amd_knowledge",
    "Return the curated AMD/ROCm porting rules (base-image choice, dependency pins to "
    "drop, flash-attn Triton build, custom-kernel HIP guards, torch-integrity invariant). "
    "Call this before porting a repo to AMD.",
    {},
)
async def recall_amd_knowledge(args: dict[str, Any]) -> dict[str, Any]:
    return _ok(amd_knowledge.amd_knowledge_text())


@tool(
    "discover_requirements",
    "Read the cloned repo's dependency declarations (requirements/setup/pyproject/README) "
    "to inform ROCm base-image selection. Returns torch/python hints + file contents.",
    {},
)
async def discover_requirements(args: dict[str, Any]) -> dict[str, Any]:
    root = _run_dir() / "repo"
    if not root.exists():
        return _err("Repo not cloned yet.")
    out = amd_tools.discover_requirements(root)
    out["candidate_base_images"] = amd_tools.ROCM_BASE_IMAGES
    return _ok(json.dumps(out, indent=2))


@tool(
    "detect_cuda_kernels",
    "Find custom CUDA kernel sources (*.cu) and their build scripts in the repo, so kernel "
    "migration to HIP is grounded in real files.",
    {},
)
async def detect_cuda_kernels(args: dict[str, Any]) -> dict[str, Any]:
    root = _run_dir() / "repo"
    if not root.exists():
        return _err("Repo not cloned yet.")
    return _ok(json.dumps(amd_tools.detect_cuda_kernels(root), indent=2))


@tool(
    "capture_torch_baseline",
    "Run the torch probe in a base image (or built image) and record the ROCm torch / "
    "torchvision / torchaudio versions + GPU visibility. Use on the chosen base image "
    "BEFORE building, to get the integrity baseline.",
    {"image": str},
)
async def capture_torch_baseline(args: dict[str, Any]) -> dict[str, Any]:
    # Offloaded to a thread: probe_torch shells out to `docker run` (blocking). Running it
    # inline would freeze the single host event loop, starving hooks/other MCP tools.
    res = await asyncio.to_thread(amd_tools.probe_torch, args["image"])
    if not res.get("ok"):
        return _err(f"torch probe failed: {res.get('error')}")
    return _ok(json.dumps(res, indent=2))


@tool(
    "write_dockerfile",
    "Write a Dockerfile into the run dir's amd_run/ build context (the exact file that "
    "docker_build uses). The base image FROM line is validated and auto-repaired if a "
    "garbled rocm/pytorch reference is detected; warnings are returned. Pass full content.",
    {"content": str},
)
async def write_dockerfile(args: dict[str, Any]) -> dict[str, Any]:
    check = amd_tools.validate_dockerfile(args["content"])
    dest = _run_dir() / "amd_run" / "Dockerfile"
    meta = amd_tools.write_text_artifact(dest, check["content"])
    meta["warnings"] = check["warnings"]
    if check["warnings"]:
        meta["note"] = "Base-image FROM line was validated/repaired; review warnings."
    return _ok(json.dumps(meta, indent=2))


@tool(
    "write_amd_file",
    "Write an auxiliary file (patch, helper script) into the run dir's amd_run/ build "
    "context so the Dockerfile can COPY it. Provide filename and content.",
    {
        "type": "object",
        "properties": {"filename": {"type": "string"}, "content": {"type": "string"}},
        "required": ["filename", "content"],
    },
)
async def write_amd_file(args: dict[str, Any]) -> dict[str, Any]:
    dest = _run_dir() / "amd_run" / args["filename"]
    meta = amd_tools.write_text_artifact(dest, args["content"])
    return _ok(json.dumps(meta))


@tool(
    "docker_build",
    "Build the Docker image from the run dir's amd_run/ context. Returns exit code + the "
    "tail of the build log (feed errors back and retry). EXECUTE MODE ONLY.",
    {"tag": str},
)
async def docker_build(args: dict[str, Any]) -> dict[str, Any]:
    ctx = _run_dir() / "amd_run"
    if not (ctx / "Dockerfile").exists():
        return _err("No Dockerfile in amd_run/. Call write_dockerfile first.")
    # Offloaded to a thread: docker_build blocks (up to 2h). Keeping it off the event loop
    # lets the host keep servicing hooks and other tool calls during the build.
    res = await asyncio.to_thread(amd_tools.docker_build, ctx, args["tag"])
    return _ok(json.dumps(res, indent=2)) if res.get("ok") else _err(json.dumps(res, indent=2))


@tool(
    "docker_run",
    "Run a bash command inside a built image with AMD GPU devices attached. Runs "
    "SYNCHRONOUSLY in the foreground; full output streams to a log file and the result "
    "returns a tail + log_file path + timeout_used. Pass `timeout` (seconds) sized to the "
    "full workload for the selected result (defaults to R2R_DOCKER_RUN_TIMEOUT / 6h). Use "
    "for smoke tests and the reproduction command. EXECUTE MODE ONLY.",
    {
        "type": "object",
        "properties": {
            "tag": {"type": "string"},
            "command": {"type": "string"},
            "workdir": {"type": "string"},
            "timeout": {
                "type": "integer",
                "description": "Max seconds before the container is killed. Size this to "
                "the full workload for the selected result (e.g. 14400 for a multi-hour "
                "train+eval). Omit to use R2R_DOCKER_RUN_TIMEOUT or the 6h default.",
            },
        },
        "required": ["tag", "command"],
    },
)
async def docker_run(args: dict[str, Any]) -> dict[str, Any]:
    # CRITICAL: offload to a thread. docker_run blocks on subprocess.run for the ENTIRE
    # container lifetime (up to 6h). Running it inline froze the single host event loop, so
    # the SDK could not service the finalize save_report call or its hook — the "host client
    # unreachable" failure in FlatQuant exec-4. to_thread keeps the loop responsive.
    res = await asyncio.to_thread(
        amd_tools.docker_run,
        args["tag"],
        args["command"],
        workdir=args.get("workdir"),
        timeout=args.get("timeout"),
    )
    return _ok(json.dumps(res, indent=2))


@tool(
    "verify_rocm_integrity",
    "THE GUARD: verify a built image kept the ROCm torch stack intact (torch/torchvision/"
    "torchaudio still ROCm builds, unchanged vs baseline) and the AMD GPU is visible "
    "(torch.cuda.is_available()). Pass the image tag and the baseline JSON from "
    "capture_torch_baseline. EXECUTE MODE ONLY.",
    {
        "type": "object",
        "properties": {
            "image": {"type": "string"},
            "baseline_json": {"type": "string", "description": "baseline probe JSON (optional)"},
        },
        "required": ["image"],
    },
)
async def verify_rocm_integrity(args: dict[str, Any]) -> dict[str, Any]:
    baseline = None
    if args.get("baseline_json"):
        try:
            baseline = json.loads(args["baseline_json"])
        except json.JSONDecodeError:
            baseline = None
    # Offloaded to a thread: verify_rocm_integrity shells out to `docker run` (blocking).
    res = await asyncio.to_thread(amd_tools.verify_rocm_integrity, args["image"], baseline=baseline)
    return (
        _ok(json.dumps(res, indent=2)) if res.get("verified") else _err(json.dumps(res, indent=2))
    )


def build_server():
    """Create the in-process MCP server with all r2r tools."""
    return create_sdk_mcp_server(
        name="r2r",
        version="0.1.0",
        tools=[
            ingest_paper,
            verify_quote_in_paper,
            clone_repo,
            map_repo,
            extract_argparse,
            verify_flags,
            save_report,
            remember,
            recall_memory,
            recall_amd_knowledge,
            discover_requirements,
            detect_cuda_kernels,
            capture_torch_baseline,
            write_dockerfile,
            write_amd_file,
            docker_build,
            docker_run,
            verify_rocm_integrity,
        ],
    )


# Convenience: the fully-qualified names for allowed_tools wiring.
TOOL_NAMES = [
    "mcp__r2r__ingest_paper",
    "mcp__r2r__verify_quote_in_paper",
    "mcp__r2r__clone_repo",
    "mcp__r2r__map_repo",
    "mcp__r2r__extract_argparse",
    "mcp__r2r__verify_flags",
    "mcp__r2r__save_report",
    "mcp__r2r__remember",
    "mcp__r2r__recall_memory",
]

# Track-2 (AMD) tools. Split so the orchestrator can gate EXECUTE-only tools by mode.
AMD_PLAN_TOOL_NAMES = [
    "mcp__r2r__recall_amd_knowledge",
    "mcp__r2r__discover_requirements",
    "mcp__r2r__detect_cuda_kernels",
    "mcp__r2r__write_dockerfile",
    "mcp__r2r__write_amd_file",
]
AMD_EXEC_TOOL_NAMES = [
    "mcp__r2r__capture_torch_baseline",
    "mcp__r2r__docker_build",
    "mcp__r2r__docker_run",
    "mcp__r2r__verify_rocm_integrity",
]
