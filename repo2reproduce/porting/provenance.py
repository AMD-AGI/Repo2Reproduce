"""Check that a ``hipify_with_fixes`` port derives from the pinned official HIPIFY output.

The CUDA source of each such kernel is translated again with the run's pinned
``hipify-perl`` into a temporary directory. The kernel's body in the port file is compared
with its body in that pristine translation, and the complete translation-to-port diff is
saved as evidence. A port that no longer resembles the HIPIFY output is a rewrite and must
be recorded as ``targeted_rewrite``.
"""

from __future__ import annotations

import difflib
import re
import subprocess
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from .official_hipify import HipifyTool, build_hipify_argv

MIN_HIPIFY_OVERLAP = 0.5
TRANSLATE_TIMEOUT_SECONDS = 300


class ProvenanceError(RuntimeError):
    """Raised when a port cannot be traced to the pinned HIPIFY translation."""


@dataclass(frozen=True)
class KernelProvenance:
    cuda_kernel: str
    cuda_source: str
    port_file: str
    compared: str
    overlap: float
    hipify_lines: int
    diff_path: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def translate_file(tool: HipifyTool, source: Path) -> str:
    """Return the pinned ``hipify-perl`` translation of one CUDA source."""

    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / source.name
        completed = subprocess.run(
            build_hipify_argv(tool, source.name, output),
            cwd=source.parent,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=TRANSLATE_TIMEOUT_SECONDS,
            check=False,
        )
        if completed.returncode != 0 or not output.is_file():
            raise ProvenanceError(
                f"hipify-perl could not translate {source}: "
                f"{completed.stderr.strip()[-500:] or completed.returncode}"
            )
        return output.read_text(encoding="utf-8", errors="replace")


def function_body(text: str, name: str) -> str | None:
    """Return the brace-delimited body of the first definition of ``name`` in ``text``."""

    for match in re.finditer(rf"\b{re.escape(name)}\s*\(", text):
        index = _closing(text, match.end() - 1, "(", ")")
        if index is None:
            continue
        rest = text[index + 1 :]
        stripped = rest.lstrip()
        if not stripped.startswith("{"):
            continue
        start = index + 1 + (len(rest) - len(stripped))
        end = _closing(text, start, "{", "}")
        if end is not None:
            return text[start : end + 1]
    return None


def _closing(text: str, start: int, opening: str, closing: str) -> int | None:
    depth = 0
    for index in range(start, len(text)):
        if text[index] == opening:
            depth += 1
        elif text[index] == closing:
            depth -= 1
            if depth == 0:
                return index
    return None


def _significant_lines(text: str) -> Counter[str]:
    lines: Counter[str] = Counter()
    for raw in text.splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if not line or line.startswith("//") or line in {"{", "}", "};", "});"}:
            continue
        lines[line] += 1
    return lines


def line_overlap(reference: str, candidate: str) -> tuple[float, int]:
    """Return the share of ``reference``'s significant lines kept in ``candidate``."""

    expected = _significant_lines(reference)
    total = sum(expected.values())
    if total == 0:
        same = reference.split() == candidate.split()
        return (1.0 if same else 0.0), 0
    present = _significant_lines(candidate)
    kept = sum(min(count, present[line]) for line, count in expected.items())
    return kept / total, total


def check_kernel(
    *,
    workspace: Path,
    cuda_kernel: str,
    cuda_source: str,
    port_files: list[str],
    tool: HipifyTool,
    evidence_dir: Path,
) -> KernelProvenance:
    """Compare one ported kernel with the pristine HIPIFY translation of its CUDA source."""

    translated = translate_file(tool, workspace / cuda_source)
    reference = function_body(translated, cuda_kernel)
    best: KernelProvenance | None = None
    for relative in port_files:
        port_text = (workspace / relative).read_text(encoding="utf-8", errors="replace")
        port_body = function_body(port_text, cuda_kernel)
        if reference is not None and port_body is not None:
            overlap, lines = line_overlap(reference, port_body)
            compared = "kernel body"
        else:
            overlap, lines = line_overlap(translated, port_text)
            compared = "whole file"
        if best is None or overlap > best.overlap:
            evidence_dir.mkdir(parents=True, exist_ok=True)
            diff_path = evidence_dir / f"{_safe(cuda_kernel)}.diff"
            diff_path.write_text(
                "".join(
                    difflib.unified_diff(
                        translated.splitlines(keepends=True),
                        port_text.splitlines(keepends=True),
                        fromfile=f"hipify-perl {tool.ref}: {cuda_source}",
                        tofile=relative,
                    )
                ),
                encoding="utf-8",
            )
            best = KernelProvenance(
                cuda_kernel=cuda_kernel,
                cuda_source=cuda_source,
                port_file=relative,
                compared=compared,
                overlap=round(overlap, 4),
                hipify_lines=lines,
                diff_path=diff_path.name,
            )
    if best is None:
        raise ProvenanceError(f"{cuda_kernel} lists no port files")
    if best.overlap < MIN_HIPIFY_OVERLAP:
        raise ProvenanceError(
            f"{cuda_kernel} is recorded as hipify_with_fixes, but only "
            f"{best.overlap:.0%} of the pinned HIPIFY translation's {best.compared} survives in "
            f"{best.port_file} (minimum {MIN_HIPIFY_OVERLAP:.0%}); keep HIPIFY's output and "
            "apply fixes on top of it, or record the kernel as targeted_rewrite"
        )
    return best


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name) or "kernel"
