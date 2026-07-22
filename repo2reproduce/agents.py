"""Subagent definitions for the Repo2Reproduce pipeline.

Each agent has a focused role, a restricted tool set, and a model tier:
* reasoning tier (default ``opus``) for selection / extraction / verification,
* mapping tier (default ``sonnet``) for the mechanical paper/code mapping.
"""

from __future__ import annotations

from pathlib import Path

from claude_agent_sdk import AgentDefinition

from .provider import mapping_model, reasoning_model

_PROMPTS = Path(__file__).parent / "prompts"


def _prompt(name: str) -> str:
    return (_PROMPTS / name).read_text(encoding="utf-8")


def _models() -> tuple[str, str]:
    return reasoning_model(), mapping_model()


# r2r tool subsets (fully-qualified names) per role.
_PAPER_TOOLS = ["mcp__r2r__ingest_paper", "mcp__r2r__verify_quote_in_paper"]
_CODE_TOOLS = [
    "mcp__r2r__clone_repo",
    "mcp__r2r__map_repo",
    "mcp__r2r__extract_argparse",
    "mcp__r2r__verify_flags",
]
_VERIFY_TOOLS = ["mcp__r2r__verify_quote_in_paper", "mcp__r2r__verify_flags"]
# Persistent cross-run memory: recall past learnings, store new ones.
_MEM_TOOLS = ["mcp__r2r__recall_memory", "mcp__r2r__remember"]
# AMD Track-2 tool sets.
_AMD_PLAN_TOOLS = [
    "mcp__r2r__recall_amd_knowledge",
    "mcp__r2r__discover_requirements",
    "mcp__r2r__detect_cuda_kernels",
    "mcp__r2r__write_dockerfile",
    "mcp__r2r__write_amd_file",
]
_AMD_EXEC_TOOLS = [
    "mcp__r2r__capture_torch_baseline",
    "mcp__r2r__docker_build",
    "mcp__r2r__docker_run",
    "mcp__r2r__verify_rocm_integrity",
]


def build_agents(amd: bool = False, execute: bool = False) -> dict[str, AgentDefinition]:
    """Build the subagent set.

    ``amd`` adds the Track-2 (ROCm enablement) agents; ``execute`` grants those agents the
    docker/verify tools that actually build and run (otherwise Track 2 is plan-only).
    """
    reasoning, mapping = _models()

    agents = {
        "paper-analyst": AgentDefinition(
            description="Reads the ENTIRE parsed paper and emits a structured, "
            "evidence-grounded paper map (experiments, tables, hyperparameters).",
            prompt=_prompt("paper_analyst.md"),
            tools=["Read", "Grep", "Glob", *_PAPER_TOOLS],
            model=mapping,
        ),
        "code-cartographer": AgentDefinition(
            description="Maps the cloned repo: entry points, CLI flags, configs, data "
            "prep, environment, and README commands. Cites real file paths.",
            prompt=_prompt("code_cartographer.md"),
            tools=["Read", "Grep", "Glob", "Bash", *_CODE_TOOLS, *_MEM_TOOLS],
            model=mapping,
        ),
        "experiment-selector": AgentDefinition(
            description="Chooses the single most reproducible experiment and produces "
            "Output 1 (exact paper text + table justifying the choice).",
            prompt=_prompt("experiment_selector.md"),
            tools=["Read", "Grep", *_PAPER_TOOLS],
            model=reasoning,
        ),
        "config-extractor": AgentDefinition(
            description="Produces Output 2: all hyperparameters/data/model/etc., "
            "reconciling paper vs code and flagging conflicts and gaps.",
            prompt=_prompt("config_extractor.md"),
            tools=["Read", "Grep", "Glob", *_PAPER_TOOLS, *_CODE_TOOLS],
            model=reasoning,
        ),
        "command-synthesizer": AgentDefinition(
            description="Produces Output 3: exact ordered commands to replicate, with "
            "every flag verified against real argparse definitions.",
            prompt=_prompt("command_synthesizer.md"),
            tools=["Read", "Grep", *_CODE_TOOLS],
            model=reasoning,
        ),
        "verifier": AgentDefinition(
            description="Anti-hallucination gate: verifies every quote and flag, and "
            "compiles the final gaps/assumptions and confidence.",
            prompt=_prompt("verifier.md"),
            tools=["Read", "Grep", *_VERIFY_TOOLS, *_MEM_TOOLS],
            model=reasoning,
        ),
    }

    if not amd:
        return agents

    # --- Track 2: AMD/ROCm enablement + convergence agents ---
    amd_exec = _AMD_EXEC_TOOLS if execute else []
    agents.update(
        {
            "base-image-selector": AgentDefinition(
                description="Picks the ROCm PyTorch base image whose bundled torch/python "
                "fit the repo, so no torch reinstall is needed.",
                prompt=_prompt("base_image_selector.md"),
                tools=["Read", "Grep", *_AMD_PLAN_TOOLS, *_MEM_TOOLS],
                model=reasoning,
            ),
            "env-porter": AgentDefinition(
                description="Rewrites repo dependencies for ROCm, dropping torch/vision/"
                "audio/numpy/ninja pins so the ROCm stack is never clobbered.",
                prompt=_prompt("env_porter.md"),
                tools=["Read", "Grep", "Glob", *_AMD_PLAN_TOOLS, *_MEM_TOOLS],
                model=mapping,
            ),
            "kernel-migrator": AgentDefinition(
                description="Detects custom CUDA kernels and produces a HIP-aware build "
                "patch so they compile on ROCm.",
                prompt=_prompt("kernel_migrator.md"),
                tools=["Read", "Grep", *_AMD_PLAN_TOOLS, *_MEM_TOOLS],
                model=reasoning,
            ),
            "build-verifier": AgentDefinition(
                description="Runs the build-verify loop: assembles the Dockerfile, builds, "
                "enforces the torch-integrity + GPU guard, and smoke-tests until "
                "ROCM_ENV_VERIFIED. Produces Output 4.",
                prompt=_prompt("build_verifier.md"),
                tools=["Read", "Grep", *_AMD_PLAN_TOOLS, *amd_exec, *_MEM_TOOLS],
                model=reasoning,
            ),
            "reconciler": AgentDefinition(
                description="Re-inspects the ported repo and updates the config/commands "
                "(Output 2/3) if AMD porting changed flags or required env vars.",
                prompt=_prompt("reconciler.md"),
                tools=["Read", "Grep", *_CODE_TOOLS, *amd_exec],
                model=reasoning,
            ),
            "executor": AgentDefinition(
                description="Runs the reconciled reproduction command inside the verified "
                "AMD image and captures the achieved metric (Output 5).",
                prompt=_prompt("executor.md"),
                tools=["Read", "Grep", *amd_exec, *_MEM_TOOLS],
                model=reasoning,
            ),
        }
    )
    return agents
