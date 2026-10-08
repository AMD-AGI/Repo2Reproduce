"""Structured contracts between Claude and the deterministic executor."""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, Field, model_validator

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _relative_path(value: str, field_name: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value in {"", "."}:
        raise ValueError(f"{field_name} must be a non-empty relative path")
    return value


class SmokeTestResult(BaseModel):
    """Evidence that the prepared image and command shape passed a bounded pilot."""

    exit_code: int
    log_path: str
    description: str

    @model_validator(mode="after")
    def validate_log_path(self) -> SmokeTestResult:
        self.log_path = _relative_path(self.log_path, "smoke_test.log_path")
        return self


class ExecutionManifest(BaseModel):
    """Validated handoff produced by the preparation agent."""

    schema_version: Literal[1] = 1
    status: Literal["ready", "blocked"]
    selected_experiment: str

    image_tag: str | None = None
    launch_script: str | None = None
    launch_script_sha256: str | None = None
    host_repository: str | None = None
    container_repository: str | None = None
    timeout_seconds: int | None = Field(default=None, gt=0, le=604800)
    expected_outputs: list[str] = Field(default_factory=list)
    smoke_test: SmokeTestResult | None = None
    blocked_reason: str | None = None

    @model_validator(mode="after")
    def validate_state(self) -> ExecutionManifest:
        if not self.selected_experiment.strip():
            raise ValueError("selected_experiment cannot be empty")

        if self.status == "blocked":
            if not self.blocked_reason or not self.blocked_reason.strip():
                raise ValueError("blocked_reason is required when status is blocked")
            return self

        required = {
            "image_tag": self.image_tag,
            "launch_script": self.launch_script,
            "launch_script_sha256": self.launch_script_sha256,
            "host_repository": self.host_repository,
            "container_repository": self.container_repository,
            "timeout_seconds": self.timeout_seconds,
            "smoke_test": self.smoke_test,
        }
        missing = [name for name, value in required.items() if value is None or value == ""]
        if missing:
            raise ValueError(f"ready manifest is missing: {', '.join(missing)}")

        assert self.launch_script is not None
        assert self.host_repository is not None
        assert self.container_repository is not None
        assert self.launch_script_sha256 is not None
        assert self.smoke_test is not None

        self.launch_script = _relative_path(self.launch_script, "launch_script")
        self.host_repository = _relative_path(self.host_repository, "host_repository")
        expected_container_repository = str(
            PurePosixPath("/run/workspace") / self.host_repository
        )
        if self.container_repository != expected_container_repository:
            raise ValueError(
                "container_repository must map exactly to host_repository below /run/workspace"
            )
        if not _SHA256.fullmatch(self.launch_script_sha256):
            raise ValueError("launch_script_sha256 must be a lowercase SHA-256 digest")
        if self.smoke_test.exit_code != 0:
            raise ValueError("ready manifest requires a successful smoke test")
        if not self.expected_outputs:
            raise ValueError("ready manifest requires at least one expected output")

        for output in self.expected_outputs:
            _relative_path(output, "expected_outputs item")
        return self


class ExecutionResult(BaseModel):
    """Durable result emitted by the Python executor."""

    schema_version: Literal[1] = 1
    status: Literal["succeeded", "failed", "timed_out", "blocked"]
    command: list[str]
    exit_code: int | None
    timed_out: bool
    duration_seconds: float
    log_path: str
    expected_outputs: list[str]
    missing_outputs: list[str]
    error: str | None = None
