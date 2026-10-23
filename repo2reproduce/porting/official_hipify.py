"""Translate a repository's CUDA sources to HIP with AMD's official HIPIFY (``hipify-perl``).

``ensure_hipify`` clones https://github.com/ROCm/HIPIFY at a pinned release tag and checks
the commit, so every run uses the same translation tables. ``translate_tree`` copies the
repository to a new directory and runs ``hipify-perl -print-stats -o=<copy> -- <source>``
once per CUDA source: ``-o`` names a single output file, and ``-inplace`` copies its backup
through an unquoted shell command. The original clone is never modified. PyTorch's
``torch.utils.hipify`` is not used.

hipify-perl prints statistics and warnings on stderr. ``<file>:<line>: warning: ...`` names
an identifier HIP does not support, that CUDA deprecated or removed, or that HIPIFY only
translates with ``-experimental``. ``warning: <file>:#<line> : <text>`` echoes a translated
line that still calls an unknown ``cuda*`` API. HIPIFY does not know PyTorch's C++ CUDA API
(``ATen/cuda``, ``at::cuda::``, ``C10_CUDA_*``) or ``__CUDA_ARCH__``, and leaves them without a
warning, so the translated files are also scanned for those leftovers.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .scanner import iter_repo_files

HIPIFY_REPOSITORY = "https://github.com/ROCm/HIPIFY"
HIPIFY_REF = "rocm-7.2.4"
HIPIFY_COMMIT = "fd26301dda3164cf82ec956cd7ba8d5adfe59e59"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "repo2reproduce" / "hipify"
DEFAULT_TIMEOUT_SECONDS = 900
CLONE_TIMEOUT_SECONDS = 300

# The extensions HIPIFY's bin/findcode.sh selects, compared case-insensitively.
HIPIFY_SUFFIXES = frozenset(
    {
        ".cu",
        ".cuh",
        ".cpp",
        ".cxx",
        ".c",
        ".cc",
        ".h",
        ".hpp",
        ".hh",
        ".inc",
        ".inl",
        ".hxx",
        ".hdl",
    }
)
_ALWAYS_TRANSLATED = frozenset({".cu", ".cuh"})

_CUDA_TOKEN = re.compile(
    r"\bcuda[A-Z_]\w*|\bcu[A-Z]\w+|\bCU(?:DA|BLAS|FFT|RAND|SPARSE|SOLVER|DNN)_\w+"
    r"|__global__|__device__|__shared__|<<<|\b__nv_\w+|\bnv_bfloat\w*"
    r"|#\s*include\s*[<\"](?:cu|cub/|thrust/|ATen/cuda/|c10/cuda/)"
)
_PYTORCH_EXTENSION = re.compile(r"torch\.utils\.cpp_extension|\bCUDAExtension\b|\bBuildExtension\b")
_FILE_STATS = re.compile(r"^\[HIPIFY\] info: file '(?P<path>.+)' statistics:\s*$")
_SECTION = re.compile(r"^\[HIPIFY\] info: (?P<title>.+?):\s*$")
_STAT = re.compile(r"^\s+(?P<key>[A-Za-z][^:]*?):\s+(?P<value>-?\d+(?:\.\d+)?)\s*$")
_CONVERTED = re.compile(r"^\s+(?P<source>\S+) => (?P<target>\S+): (?P<count>\d+)\s*$")
_UNKNOWN_WARNING = re.compile(r"^\s*warning: (?P<path>.+?):#(?P<line>\d+) : (?P<text>.*)$")
_LOCATED_WARNING = re.compile(r"^\s*(?P<path>\S[^:]*):(?P<line>\d+): warning: (?P<text>.*)$")
_UNSUPPORTED = re.compile(
    r"unsupported [\w ]*?(?:function|identifier)\w*:?\s*\"?(?P<name>[A-Za-z_]\w*)\"?"
)
_DEPRECATED = re.compile(r"(?:deprecated|removed) CUDA identifier:\s*(?P<name>[A-Za-z_]\w*)")
_EXPERIMENTAL = re.compile(r"experimental ROCm HIP identifier:\s*(?P<name>[A-Za-z_]\w*)")
_UNCONVERTED = re.compile(
    r"\b(?:cuda|cublas|cufft|curand|cusparse|cusolver|cudnn|nccl|nvrtc|nvtx|nvml)\w*"
    r"|\bcu[A-Z]\w*|\bCU(?:DA|BLAS|FFT|RAND|SPARSE|SOLVER|DNN)?_[A-Z0-9_]+|\b__nv_\w+"
)
_LEFTOVER = re.compile(
    r"\b(?:ATen|c10|THC)/cuda/[\w/.]+"
    r"|\b(?:at|c10)::cuda::\w+"
    r"|\b(?:C10|AT|TORCH)_CUDA_\w+"
    r"|\bCUDA(?:Guard|StreamGuard|Stream|Event|CachingAllocator|Generator)\w*"
    r"|\bget(?:Current|Default)CUDA\w+"
    r"|\b__CUDA_ARCH__\b|\bnvcuda::\w+"
)
_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)
WARNING_KINDS = {
    "unsupported": "Identifiers HIP does not support (each needs a manual replacement)",
    "unknown": "Identifiers HIPIFY did not recognise (left as CUDA)",
    "experimental": "Identifiers HIPIFY translates only with -experimental (left as CUDA)",
    "deprecated": "Deprecated or removed CUDA identifiers (check their semantics)",
}


class HipifyToolError(RuntimeError):
    """The official HIPIFY checkout could not be fetched or verified."""


@dataclass(frozen=True)
class HipifyTool:
    repository: str
    ref: str
    commit: str
    script: str


@dataclass(frozen=True)
class HipifyWarning:
    path: str
    line: int
    kind: str
    text: str
    identifiers: tuple[str, ...]


@dataclass(frozen=True)
class Leftover:
    path: str
    line: int
    identifiers: tuple[str, ...]
    text: str


@dataclass
class HipifyFileStats:
    path: str
    stats: dict[str, float] = field(default_factory=dict)
    converted: dict[str, int] = field(default_factory=dict)
    changed: bool = False


@dataclass
class HipifyReport:
    root: str
    output: str
    tool: HipifyTool | None = None
    files: list[HipifyFileStats] = field(default_factory=list)
    warnings: list[HipifyWarning] = field(default_factory=list)
    leftovers: list[Leftover] = field(default_factory=list)
    failures: dict[str, str] = field(default_factory=dict)
    pytorch_extension: bool = False
    error: str | None = None

    def total(self, key: str) -> int:
        return int(sum(stats.stats.get(key, 0) for stats in self.files))

    def identifiers(self, kind: str) -> dict[str, int]:
        """Identifier counts for one warning kind, most frequent first."""

        counts: dict[str, int] = {}
        for warning in self.warnings:
            if warning.kind != kind:
                continue
            for identifier in warning.identifiers:
                counts[identifier] = counts.get(identifier, 0) + 1
        return _by_count(counts)

    def leftover_identifiers(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for leftover in self.leftovers:
            for identifier in leftover.identifiers:
                counts[identifier] = counts.get(identifier, 0) + 1
        return _by_count(counts)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["changed_files"] = [stats.path for stats in self.files if stats.changed]
        data["identifiers"] = {kind: self.identifiers(kind) for kind in WARNING_KINDS}
        data["identifiers"]["leftover"] = self.leftover_identifiers()
        return data

    def to_text(self) -> str:
        if self.error:
            return f"Official HIPIFY on {self.root} failed: {self.error}"
        tool = self.tool
        source = f"{tool.repository} {tool.ref} @ {tool.commit[:12]}" if tool else "HIPIFY"
        changed = sum(stats.changed for stats in self.files)
        lines = [
            (
                f"hipify-perl ({source}) translated {len(self.files)} CUDA sources from "
                f"{self.root} into {self.output}: {changed} files changed, "
                f"{self.total('CONVERTED refs count')} references converted, "
                f"{len(self.warnings)} warnings, {len(self.leftovers)} lines with leftovers"
            )
        ]
        if self.pytorch_extension:
            lines.append(
                "This repository builds PyTorch C++/CUDA extensions. Build the translated copy "
                "without torch.utils.hipify, and rewrite the PyTorch CUDA API leftovers below "
                "to the ATen/hip and c10/hip equivalents."
            )
        for kind, title in WARNING_KINDS.items():
            counts = self.identifiers(kind)
            if counts:
                lines.append(f"{title}:")
                lines += self._identifier_lines(
                    counts,
                    [(w.path, w.line, w.identifiers) for w in self.warnings if w.kind == kind],
                )
        counts = self.leftover_identifiers()
        if counts:
            lines.append(
                "CUDA references HIPIFY left in the output without a warning (rewrite these):"
            )
            lines += self._identifier_lines(
                counts, [(item.path, item.line, item.identifiers) for item in self.leftovers]
            )
        unexplained = [w for w in self.warnings if not w.identifiers]
        if unexplained:
            lines.append("Warnings without a recognisable CUDA identifier:")
            lines += [f"  {w.path}:{w.line}: {w.text[:140]}" for w in unexplained[:20]]
        if self.failures:
            lines.append("Files hipify-perl could not translate (copied unchanged):")
            lines += [f"  {path}: {message[:300]}" for path, message in self.failures.items()]
        if self.files:
            lines.append("Per file (converted references / lines / warnings, * = changed):")
            for stats in self.files:
                values = stats.stats
                lines.append(
                    f"  {'*' if stats.changed else ' '} {stats.path}: "
                    f"{int(values.get('CONVERTED refs count', 0))} / "
                    f"{int(values.get('TOTAL lines of code', 0))} / "
                    f"{int(values.get('WARNINGS', 0))}"
                )
        return "\n".join(lines)

    @staticmethod
    def _identifier_lines(
        counts: dict[str, int], places: list[tuple[str, int, tuple[str, ...]]]
    ) -> list[str]:
        lines = []
        for identifier, count in counts.items():
            where = [f"{path}:{line}" for path, line, names in places if identifier in names]
            shown = ", ".join(where[:5]) + (" ..." if len(where) > 5 else "")
            lines.append(f"  {identifier} x{count} ({shown})")
        return lines


def _by_count(counts: dict[str, int]) -> dict[str, int]:
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def ensure_hipify(
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    *,
    ref: str = HIPIFY_REF,
    commit: str = HIPIFY_COMMIT,
    git: str = "git",
) -> HipifyTool:
    """Return the pinned official HIPIFY checkout, cloning it into ``cache_dir`` if needed."""

    cache_dir = Path(cache_dir).expanduser().resolve()
    checkout = cache_dir / f"HIPIFY-{ref}"
    script = checkout / "bin" / "hipify-perl"
    if not script.is_file():
        if shutil.which(git) is None:
            raise HipifyToolError(f"{git} not found on PATH")
        cache_dir.mkdir(parents=True, exist_ok=True)
        staging = cache_dir / f".HIPIFY-{ref}.{os.getpid()}"
        shutil.rmtree(staging, ignore_errors=True)
        argv = [
            git,
            "-c",
            "advice.detachedHead=false",
            "clone",
            "--quiet",
            "--depth",
            "1",
            "--branch",
            ref,
            HIPIFY_REPOSITORY,
            str(staging),
        ]
        try:
            proc = subprocess.run(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=CLONE_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            shutil.rmtree(staging, ignore_errors=True)
            raise HipifyToolError(f"cloning {HIPIFY_REPOSITORY} timed out") from exc
        if proc.returncode != 0:
            shutil.rmtree(staging, ignore_errors=True)
            raise HipifyToolError(
                f"cloning {HIPIFY_REPOSITORY} at {ref} failed: {proc.stdout.strip()[-1000:]}"
            )
        try:
            staging.rename(checkout)
        except OSError:
            shutil.rmtree(staging, ignore_errors=True)
            if not script.is_file():
                raise
    head = subprocess.run(
        [git, "-C", str(checkout), "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    ).stdout.strip()
    if head != commit:
        raise HipifyToolError(
            f"{checkout} is at {head or 'an unknown commit'}, expected {ref} = {commit}"
        )
    if not script.is_file():
        raise HipifyToolError(f"{script} is missing from the HIPIFY checkout")
    return HipifyTool(repository=HIPIFY_REPOSITORY, ref=ref, commit=commit, script=str(script))


def cuda_source_files(root: Path) -> list[str]:
    """Relative paths of the files HIPIFY should translate, in a stable order."""

    files: list[str] = []
    for path in iter_repo_files(root):
        suffix = path.suffix.lower()
        if suffix not in HIPIFY_SUFFIXES or path.is_symlink():
            continue
        if suffix not in _ALWAYS_TRANSLATED:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if not _CUDA_TOKEN.search(text):
                continue
        files.append(path.relative_to(root).as_posix())
    return files


def uses_pytorch_extension(root: Path) -> bool:
    for path in iter_repo_files(root):
        if path.suffix != ".py":
            continue
        try:
            if _PYTORCH_EXTENSION.search(path.read_text(encoding="utf-8", errors="replace")):
                return True
        except OSError:
            continue
    return False


def parse_hipify_output(output: str) -> tuple[list[HipifyFileStats], list[HipifyWarning]]:
    """Parse hipify-perl ``-print-stats`` stderr into per-file stats and warnings."""

    files: list[HipifyFileStats] = []
    warnings: list[HipifyWarning] = []
    current: HipifyFileStats | None = None
    section = ""
    for line in output.splitlines():
        match = _UNKNOWN_WARNING.match(line) or _LOCATED_WARNING.match(line)
        if match:
            text = match.group("text").strip()
            if match.re is _UNKNOWN_WARNING:
                kind, names = "unknown", _UNCONVERTED.findall(text)
            elif deprecated := _DEPRECATED.search(text):
                kind, names = "deprecated", [deprecated.group("name")]
            elif experimental := _EXPERIMENTAL.search(text):
                kind, names = "experimental", [experimental.group("name")]
            elif unsupported := _UNSUPPORTED.search(text):
                kind, names = "unsupported", [unsupported.group("name")]
            else:
                kind, names = "unsupported", []
            warnings.append(
                HipifyWarning(
                    path=match.group("path"),
                    line=int(match.group("line")),
                    kind=kind,
                    text=text,
                    identifiers=tuple(dict.fromkeys(names)),
                )
            )
            continue
        file_stats = _FILE_STATS.match(line)
        if file_stats:
            current = None
            if file_stats.group("path") != "GLOBAL":
                current = HipifyFileStats(path=file_stats.group("path"))
                files.append(current)
            section = "statistics"
            continue
        heading = _SECTION.match(line)
        if heading:
            section = heading.group("title")
            continue
        if current is None:
            continue
        converted = _CONVERTED.match(line)
        if converted and section.startswith("CONVERTED refs by names"):
            key = f"{converted.group('source')} => {converted.group('target')}"
            current.converted[key] = int(converted.group("count"))
            continue
        stat = _STAT.match(line)
        if stat and section == "statistics":
            current.stats[stat.group("key")] = float(stat.group("value"))
    return files, warnings


def find_leftovers(path: str, text: str) -> list[Leftover]:
    """CUDA references outside comments that HIPIFY leaves unchanged and does not warn about."""

    code = _COMMENT.sub(lambda match: re.sub(r"[^\n]", " ", match.group()), text)
    leftovers = []
    for number, (line, original) in enumerate(
        zip(code.split("\n"), text.split("\n"), strict=True), start=1
    ):
        names = _LEFTOVER.findall(line)
        if names:
            leftovers.append(
                Leftover(
                    path=path,
                    line=number,
                    identifiers=tuple(dict.fromkeys(names)),
                    text=original.strip()[:200],
                )
            )
    return leftovers


def build_hipify_argv(tool: HipifyTool, source: str, output: Path) -> list[str]:
    return ["perl", tool.script, "-print-stats", f"-o={output}", "--", source]


def translate_tree(
    root: Path | str,
    output: Path | str,
    *,
    tool: HipifyTool,
    files: Sequence[str] | None = None,
    overwrite: bool = False,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> HipifyReport:
    """Copy ``root`` to ``output`` and replace each CUDA source in the copy with HIPIFY's output."""

    root = Path(root).resolve()
    output = Path(os.path.abspath(output))
    report = HipifyReport(root=str(root), output=str(output), tool=tool)
    if not root.is_dir():
        report.error = f"not a directory: {root}"
        return report
    if output == root or root in output.parents or output in root.parents:
        report.error = "the output directory must be outside the repository and not contain it"
        return report
    report.pytorch_extension = uses_pytorch_extension(root)
    selected = list(files) if files is not None else cuda_source_files(root)
    if not selected:
        report.error = "no CUDA sources found"
        return report
    if shutil.which("perl") is None:
        report.error = "perl not found on PATH"
        return report
    if output.is_symlink() or (output.exists() and not overwrite):
        report.error = f"{output} already exists; pass overwrite to regenerate it"
        return report
    if output.exists():
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(root, output, symlinks=True)

    deadline = time.monotonic() + timeout
    stderr: list[str] = []
    for relative in selected:
        source = root / relative
        target = output / relative
        if source.is_symlink() or not source.is_file():
            report.failures[relative] = "not a regular file"
            continue
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            report.failures[relative] = f"not translated: {timeout} second budget exhausted"
            continue
        staging = target.with_name(f".{target.name}.hipify-tmp")
        try:
            proc = subprocess.run(
                build_hipify_argv(tool, relative, staging),
                cwd=root,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=remaining,
                check=False,
            )
        except subprocess.TimeoutExpired:
            staging.unlink(missing_ok=True)
            report.failures[relative] = "hipify-perl timed out"
            continue
        stderr.append(proc.stderr)
        if proc.returncode != 0 or not staging.is_file():
            staging.unlink(missing_ok=True)
            report.failures[relative] = (
                f"exit code {proc.returncode}: {proc.stderr.strip()[-1000:]}"
            )
            continue
        os.replace(staging, target)

    stats_by_path: dict[str, HipifyFileStats] = {}
    parsed, report.warnings = parse_hipify_output("\n".join(stderr))
    for stats in parsed:
        stats_by_path[stats.path] = stats
    for relative in selected:
        if relative in report.failures:
            continue
        stats = stats_by_path.get(relative) or HipifyFileStats(path=relative)
        translated = (output / relative).read_bytes()
        stats.changed = translated != (root / relative).read_bytes()
        report.files.append(stats)
        report.leftovers += find_leftovers(relative, translated.decode("utf-8", errors="replace"))
    return report


def hipify_repository(
    root: Path | str,
    output: Path | str,
    *,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    overwrite: bool = False,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> HipifyReport:
    """Fetch the pinned official HIPIFY and translate ``root`` into ``output``."""

    try:
        tool = ensure_hipify(cache_dir)
    except (HipifyToolError, OSError) as exc:
        return HipifyReport(root=str(root), output=str(output), error=str(exc))
    return translate_tree(root, output, tool=tool, overwrite=overwrite, timeout=timeout)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("root", type=Path, help="Repository directory (left unchanged)")
    parser.add_argument("output", type=Path, help="Directory for the translated copy")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output")
    parser.add_argument("--json", action="store_true", help="Print the report as JSON")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args(argv)
    report = hipify_repository(
        args.root,
        args.output,
        cache_dir=args.cache_dir,
        overwrite=args.overwrite,
        timeout=args.timeout,
    )
    print(json.dumps(report.to_dict(), indent=2) if args.json else report.to_text())
    return 1 if report.error else 0


if __name__ == "__main__":
    sys.exit(main())
