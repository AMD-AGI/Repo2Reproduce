"""Profile GPU kernels with rocprofv3 inside a ROCm container.

The command runs ``warmup + runs`` times in one container, one after another. Warm-up runs
are discarded and each kernel is summarised by its median over the measured runs. Running
everything in a single container keeps the JIT and autotuning caches that warm-up fills
(Triton, MIOpen, hipBLASLt), so the measured runs see what a benchmark would see.

rocprofv3 1.1.0 (ROCm 7.2) aborts and then hangs while flushing its buffers when its
working directory is ``/`` (ROCm/rocm-systems#3505), so the container never runs from
``/`` and every run has a hard timeout.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath
from typing import Any

# Device flags for a ROCm container. The multi-step package does not ship the old AMD
# tool module, and the profiler is not installed on the agent's PATH.
ROCM_DOCKER_DEVICE_ARGS = [
    "--device=/dev/kfd",
    "--device=/dev/dri",
    "--group-add",
    "video",
    "--security-opt",
    "seccomp=unconfined",
]

DEFAULT_RUNS = 5
DEFAULT_WARMUP = 1
DEFAULT_WORKDIR = "/tmp"
DEFAULT_RUN_TIMEOUT_SECONDS = 1800
DEFAULT_EXCLUDE = r"^__amd_rocclr_"
_MARK = "@@r2r-rocprof"
_BEGIN = re.compile(rf"^{_MARK} begin (\d+) (-?\d+)$")


@dataclass(frozen=True)
class KernelStat:
    """One row of rocprofv3's ``kernel_stats.csv`` (durations in nanoseconds)."""

    name: str
    calls: int
    total_ns: float
    average_ns: float
    min_ns: float | None = None
    max_ns: float | None = None


@dataclass(frozen=True)
class RunCapture:
    index: int
    exit_code: int
    stats: tuple[KernelStat, ...]
    log_tail: str = ""


@dataclass(frozen=True)
class KernelSummary:
    name: str
    runs_present: int
    calls_per_run: float
    median_total_ns: float
    median_average_ns: float
    min_total_ns: float
    max_total_ns: float
    cv: float | None


@dataclass
class ProfileReport:
    image: str
    command: list[str]
    workdir: str
    warmup: int
    runs: int
    exit_codes: list[int] = field(default_factory=list)
    kernels: list[KernelSummary] = field(default_factory=list)
    median_total_ns: float | None = None
    cv_total: float | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def measured_ok(self) -> int:
        return sum(1 for code in self.exit_codes[self.warmup :] if code == 0)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["measured_ok"] = self.measured_ok
        return data

    def to_text(self, top: int = 15) -> str:
        command = shlex.join(self.command)
        lines = [
            (
                f"rocprofv3 kernel profile: {self.warmup} warm-up + {self.runs} measured runs "
                f"of `{command}` in {self.image}"
            ),
            f"Measured runs that succeeded: {self.measured_ok}/{self.runs}",
        ]
        if self.median_total_ns is not None:
            spread = f" (CV {self.cv_total:.1%})" if self.cv_total is not None else ""
            lines.append(
                f"Median GPU kernel time per run: {_format_ns(self.median_total_ns)}{spread}"
            )
        if self.kernels:
            lines.append(f"Top kernels by median time per run (of {len(self.kernels)}):")
            total = self.median_total_ns or 0.0
            for rank, kernel in enumerate(self.kernels[:top], 1):
                share = f"{kernel.median_total_ns / total:6.1%}" if total else "     -"
                spread = f"CV {kernel.cv:.1%}" if kernel.cv is not None else "CV -"
                average = _format_ns(kernel.median_average_ns)
                lines.append(
                    f"  {rank:>2}. {_format_ns(kernel.median_total_ns):>10} {share}  "
                    f"calls/run {kernel.calls_per_run:g}  avg {average}  {spread}  "
                    f"{short_kernel_name(kernel.name)}"
                )
        lines += [f"Error: {error}" for error in self.errors]
        return "\n".join(lines)


def _format_ns(value: float) -> str:
    for unit, scale in (("s", 1e9), ("ms", 1e6), ("us", 1e3)):
        if value >= scale:
            return f"{value / scale:.3g} {unit}"
    return f"{value:.0f} ns"


def short_kernel_name(name: str, width: int = 100) -> str:
    """Drop the parameter list and collapse template arguments of a demangled name."""

    name = name.removeprefix("void ").replace("(anonymous namespace)", "{anonymous}")
    out: list[str] = []
    depth = 0
    for char in name:
        if char == "(" and depth == 0:
            break
        if char == "<":
            if depth == 0:
                out.append("<...>")
            depth += 1
        elif char == ">" and depth:
            depth -= 1
        elif depth == 0:
            out.append(char)
    short = "".join(out).strip() or name
    return short if len(short) <= width else short[: width - 3] + "..."


def parse_kernel_stats(csv_text: str) -> list[KernelStat]:
    """Parse one or more concatenated ``kernel_stats.csv`` files (one per process)."""

    merged: dict[str, KernelStat] = {}
    header: list[str] | None = None
    for row in csv.reader(io.StringIO(csv_text)):
        if not row:
            continue
        if row[0] == "Name":
            header = row
            continue
        if header is None or len(row) != len(header):
            continue
        values = dict(zip(header, row))
        try:
            stat = KernelStat(
                name=values["Name"],
                calls=int(values["Calls"]),
                total_ns=float(values["TotalDurationNs"]),
                average_ns=float(values["AverageNs"]),
                min_ns=float(values["MinNs"]) if values.get("MinNs") else None,
                max_ns=float(values["MaxNs"]) if values.get("MaxNs") else None,
            )
        except (KeyError, ValueError):
            continue
        previous = merged.get(stat.name)
        if previous is not None:
            calls = previous.calls + stat.calls
            total = previous.total_ns + stat.total_ns
            stat = KernelStat(
                name=stat.name,
                calls=calls,
                total_ns=total,
                average_ns=total / calls if calls else 0.0,
                min_ns=_combine(min, previous.min_ns, stat.min_ns),
                max_ns=_combine(max, previous.max_ns, stat.max_ns),
            )
        merged[stat.name] = stat
    return list(merged.values())


def _combine(pick: Any, first: float | None, second: float | None) -> float | None:
    present = [value for value in (first, second) if value is not None]
    return pick(present) if present else None


def build_profile_script(
    command: Sequence[str],
    *,
    runs: int,
    warmup: int,
    workdir: str,
    kernel_regex: str | None = None,
    run_timeout: int = DEFAULT_RUN_TIMEOUT_SECONDS,
) -> str:
    """Shell script, run inside the container, that profiles each run and prints its stats."""

    include = f" --kernel-include-regex {shlex.quote(kernel_regex)}" if kernel_regex else ""
    return "\n".join(
        [
            "set -u",
            f"cd {shlex.quote(workdir)}",
            'out=$(mktemp -d "${TMPDIR:-/tmp}/r2r-rocprof.XXXXXX")',
            f"for i in $(seq 1 {warmup + runs}); do",
            (
                f"  timeout -s KILL {run_timeout} rocprofv3 --kernel-trace --stats -f csv"
                f'{include} -d "$out/$i" -o "run_%pid%" -- {shlex.join(command)}'
                ' > "$out/$i.log" 2>&1'
            ),
            "  rc=$?",
            f'  echo "{_MARK} begin $i $rc"',
            '  cat "$out/$i"/*kernel_stats.csv 2>/dev/null',
            f'  echo "{_MARK} end $i"',
            '  if [ "$rc" -ne 0 ]; then',
            f'    echo "{_MARK} log $i"; tail -n 40 "$out/$i.log"; echo "{_MARK} endlog $i"',
            "  fi",
            '  rm -rf "$out/$i"',
            "done",
        ]
    )


def parse_profile_output(output: str) -> list[RunCapture]:
    """Split the container's stdout into one capture per run."""

    captures: list[RunCapture] = []
    index = exit_code = None
    stats_lines: list[str] = []
    log_lines: list[str] = []
    mode = ""
    for line in output.splitlines():
        begin = _BEGIN.match(line)
        if begin:
            index, exit_code = int(begin.group(1)), int(begin.group(2))
            stats_lines, log_lines, mode = [], [], "stats"
            continue
        if index is None:
            continue
        if line == f"{_MARK} end {index}":
            captures.append(
                RunCapture(index, exit_code or 0, tuple(parse_kernel_stats("\n".join(stats_lines))))
            )
            mode = ""
        elif line == f"{_MARK} log {index}":
            mode = "log"
        elif line == f"{_MARK} endlog {index}" and captures:
            last = captures[-1]
            captures[-1] = RunCapture(last.index, last.exit_code, last.stats, "\n".join(log_lines))
            mode = ""
        elif mode == "stats":
            stats_lines.append(line)
        elif mode == "log":
            log_lines.append(line)
    return captures


def summarize(
    captures: Sequence[RunCapture], *, warmup: int, exclude: str | None = DEFAULT_EXCLUDE
) -> tuple[list[KernelSummary], float | None, float | None]:
    """Median-based per-kernel summary over the successful measured runs."""

    excluded = re.compile(exclude) if exclude else None
    measured = [c for c in captures if c.index > warmup and c.exit_code == 0]
    per_kernel: dict[str, list[KernelStat]] = {}
    run_totals: list[float] = []
    for capture in measured:
        kept = [s for s in capture.stats if not (excluded and excluded.search(s.name))]
        run_totals.append(sum(stat.total_ns for stat in kept))
        for stat in kept:
            per_kernel.setdefault(stat.name, []).append(stat)
    kernels = [
        KernelSummary(
            name=name,
            runs_present=len(stats),
            calls_per_run=statistics.median(stat.calls for stat in stats),
            median_total_ns=statistics.median(stat.total_ns for stat in stats),
            median_average_ns=statistics.median(stat.average_ns for stat in stats),
            min_total_ns=min(stat.total_ns for stat in stats),
            max_total_ns=max(stat.total_ns for stat in stats),
            cv=_cv([stat.total_ns for stat in stats]),
        )
        for name, stats in per_kernel.items()
    ]
    kernels.sort(key=lambda kernel: kernel.median_total_ns, reverse=True)
    median_total = statistics.median(run_totals) if run_totals else None
    return kernels, median_total, _cv(run_totals)


def _cv(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = statistics.fmean(values)
    return statistics.stdev(values) / mean if mean else None


@dataclass(frozen=True)
class Mount:
    source: str
    target: str
    read_only: bool = True

    def argument(self) -> str:
        suffix = ",readonly" if self.read_only else ""
        return f"type=bind,source={self.source},target={self.target}{suffix}"

    @classmethod
    def parse(cls, spec: str) -> Mount:
        parts = spec.split(":")
        if len(parts) not in (2, 3) or (len(parts) == 3 and parts[2] not in {"ro", "rw"}):
            raise ValueError(f"mount must be SOURCE:TARGET[:ro|rw], got {spec!r}")
        return cls(os.path.abspath(parts[0]), parts[1], len(parts) == 2 or parts[2] == "ro")


def build_docker_argv(
    image: str,
    script: str,
    *,
    workdir: str,
    name: str,
    gpu: str | None = None,
    env: Mapping[str, str] | None = None,
    mounts: Sequence[Mount] = (),
    docker: str = "docker",
) -> list[str]:
    argv = [docker, "run", "--rm", "--name", name, "--ipc=host", "-w", workdir]
    argv += ROCM_DOCKER_DEVICE_ARGS
    variables = dict(env or {})
    if gpu is not None:
        variables.setdefault("ROCR_VISIBLE_DEVICES", gpu)
    else:
        for pin in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
            if pin in os.environ:
                variables.setdefault(pin, os.environ[pin])
    for key, value in variables.items():
        argv += ["-e", f"{key}={value}"]
    for mount in mounts:
        argv += ["--mount", mount.argument()]
    return argv + [image, "bash", "-lc", script]


def profile_in_docker(
    image: str,
    command: Sequence[str],
    *,
    runs: int = DEFAULT_RUNS,
    warmup: int = DEFAULT_WARMUP,
    workdir: str = DEFAULT_WORKDIR,
    gpu: str | None = None,
    env: Mapping[str, str] | None = None,
    mounts: Sequence[Mount] = (),
    kernel_regex: str | None = None,
    exclude: str | None = DEFAULT_EXCLUDE,
    run_timeout: int = DEFAULT_RUN_TIMEOUT_SECONDS,
    docker: str = "docker",
) -> ProfileReport:
    """Profile ``command`` with rocprofv3 in ``image`` and summarise the measured runs."""

    if not command:
        raise ValueError("command must not be empty")
    if runs < 1 or warmup < 0:
        raise ValueError("runs must be at least 1 and warmup at least 0")
    if not workdir.startswith("/") or PurePosixPath(workdir) == PurePosixPath("/"):
        raise ValueError(
            "workdir must be an absolute directory other than '/': rocprofv3 1.1.0 crashes "
            "when its working directory is '/'"
        )
    report = ProfileReport(
        image=image, command=list(command), workdir=workdir, warmup=warmup, runs=runs
    )
    if shutil.which(docker) is None:
        report.errors.append(f"{docker} not found on PATH")
        return report

    name = f"r2r-rocprof-{uuid.uuid4().hex[:12]}"
    script = build_profile_script(
        command,
        runs=runs,
        warmup=warmup,
        workdir=workdir,
        kernel_regex=kernel_regex,
        run_timeout=run_timeout,
    )
    argv = build_docker_argv(
        image, script, workdir=workdir, name=name, gpu=gpu, env=env, mounts=mounts, docker=docker
    )
    deadline = run_timeout * (warmup + runs) + 300
    try:
        proc = subprocess.run(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=deadline,
            check=False,
        )
        output = proc.stdout
    except subprocess.TimeoutExpired as exc:
        subprocess.run([docker, "kill", name], capture_output=True, check=False)
        output = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode()
        report.errors.append(f"profiling did not finish within {deadline} seconds")
        proc = None

    captures = parse_profile_output(output)
    report.exit_codes = [capture.exit_code for capture in captures]
    for capture in captures:
        if capture.exit_code != 0:
            label = "warm-up" if capture.index <= warmup else "measured"
            report.errors.append(
                f"{label} run {capture.index} exited with {capture.exit_code}:\n{capture.log_tail}"
            )
    if proc is not None and proc.returncode != 0 and not captures:
        report.errors.append(f"docker exited with {proc.returncode}:\n{output.strip()[-2000:]}")
    report.kernels, report.median_total_ns, report.cv_total = summarize(
        captures, warmup=warmup, exclude=exclude
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        usage="%(prog)s --image IMAGE [options] -- COMMAND [ARGS...]",
    )
    parser.add_argument("--image", required=True, help="ROCm image with rocprofv3")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--workdir", default=DEFAULT_WORKDIR, help="Container working directory")
    parser.add_argument("--gpu", help="Physical GPU index, passed as ROCR_VISIBLE_DEVICES")
    parser.add_argument(
        "--mount", action="append", default=[], help="SOURCE:TARGET[:ro|rw]; read-only by default"
    )
    parser.add_argument("--env", action="append", default=[], help="KEY=VALUE for the container")
    parser.add_argument("--kernel-regex", help="Only trace kernels whose names match")
    parser.add_argument("--run-timeout", type=int, default=DEFAULT_RUN_TIMEOUT_SECONDS)
    parser.add_argument("--json", action="store_true", help="Print the report as JSON")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("give the command to profile after --")
    if any("=" not in item for item in args.env):
        parser.error("--env takes KEY=VALUE")
    env = dict(item.split("=", 1) for item in args.env)
    report = profile_in_docker(
        args.image,
        command,
        runs=args.runs,
        warmup=args.warmup,
        workdir=args.workdir,
        gpu=args.gpu,
        env=env,
        mounts=[Mount.parse(spec) for spec in args.mount],
        kernel_regex=args.kernel_regex,
        run_timeout=args.run_timeout,
    )
    print(json.dumps(report.to_dict(), indent=2) if args.json else report.to_text())
    return 0 if report.measured_ok == report.runs and not report.errors else 1


if __name__ == "__main__":
    sys.exit(main())
