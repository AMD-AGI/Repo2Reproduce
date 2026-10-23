---
name: execution-repair
description: Diagnose and minimally repair a failed deterministic execution of a frozen research-paper reproduction without changing its science. Use when a prepared experiment failed and you must classify the failure from its logs, choose the smallest legitimate repair to the environment, dependencies, launch delivery, generated configs, metric extraction, or repository source, prove it with a bounded smoke test on the same code path, and preserve the failure and repair evidence.
---

# Execution repair

A deterministic executor ran a frozen experiment and it failed. Your job is to make the
same experiment executable with the smallest change that fixes the real root cause, while
the science stays exactly as frozen. The stage prompt supplies the run's artifact paths,
the attempt number, the frozen values, and where smoke-test evidence must be written.

## Evidence handling

Read before you change anything, and never destroy what you read.

1. **Execution result:** status, exit code, error text, missing outputs, and any metric
   extraction error. This tells you which phase failed: launch, run, output, or extraction.
2. **Complete failed log:** read the end first to find the terminal error, then scroll up
   to the **first** error or traceback. Cascading errors after the first one are symptoms.
   Note the exact failing file, line, and message.
3. **Previous repairs:** read every earlier repair record and patch. Do not repeat a
   repair that already failed, and do not undo one that fixed an earlier failure.
4. **Manifests and delivery files:** the environment manifest, execution manifest,
   Dockerfile and build log, launch script, and generated configs. Check how each frozen
   value reaches the code.
5. **Repository code on the failing path:** open the cited source lines and confirm the
   mechanism of failure before choosing a repair.

Preservation rules:

- Never delete, truncate, or overwrite earlier logs, archived attempts, repair records, or
  patches. Write new evidence to new paths.
- Save complete command output (stdout and stderr) for builds and smoke tests.
- Quote the decisive log line verbatim in the diagnosis and cite the source location.
- Leave repository edits as uncommitted working-tree changes against the pinned base
  commit. Never commit, stash, reset, rebase, or check out another revision; the
  orchestrator captures the diff.

## Classify the failure

Assign exactly one primary class from the first real error. See
[failure-signatures.md](failure-signatures.md) for common signatures.

| Class | Typical evidence | Legitimate repair surface |
| --- | --- | --- |
| Environment or build | missing module, import error, ABI or binary mismatch, GPU not visible, kernel built for another architecture | Dockerfile, system packages, wheel source, build flags |
| Dependency API drift | `AttributeError` or `TypeError` from a library API removed or renamed in the installed version | pin the version the code expects; otherwise a minimal compatibility shim |
| Launch or delivery | unrecognized argument, config-override syntax, wrong working directory, wrong path, quoting, missing generated config | launch script, generated config files |
| Data or checkpoint access | file not found, download or authentication failure, wrong layout | data placement or download step for the **same** dataset and checkpoint |
| Resource | out of memory, shared-memory or file-descriptor exhaustion | non-scientific resource knobs only (allocator settings, worker counts, shared memory size) |
| Platform code defect | code path assumes another vendor or library build (hard-coded device kernels, unsupported dtype ops) | minimal source patch that preserves the computation |
| Output or metric extraction | run completed but expected output missing or misnamed, extractor selector wrong | output paths, extractor source or selector, matched against the real artifact |
| Scientific failure | divergence, NaN loss, metric assertion, numerically wrong result | usually none; repair only if the root cause is platform or delivery |

If the evidence supports more than one class, fix the earliest failure in execution order
and note the others in the diagnosis.

## Immutable science

These never change during repair:

- the selected experiment, dataset and split, method, model, checkpoint, metric and its
  definition, evaluation protocol, and aggregation;
- every frozen scientific input value (each audit item id and its value in the execution
  manifest), seeds, sample counts, training duration, and batch sizes that are audited;
- the paper target and the pinned repository base commit;
- the frozen target and configuration-audit artifacts named by the stage prompt.

You may change **how** a frozen value is delivered (for example, from a CLI flag to a
generated config file), never the value itself.

Forbidden, even if it makes the run pass:

- reducing the workload: fewer samples, steps, epochs, seeds, runs, or a smaller split;
- skipping or stubbing a pipeline step, an evaluation component, or a metric term;
- swallowing failures (`|| true`, broad `try/except: pass`, ignoring exit codes);
- writing, copying, or hard-coding an expected or precomputed result;
- substituting a different metric implementation, dataset, checkpoint, or model;
- changing an audited value to fit memory or time.

Implementation substitutions (for example, replacing a vendor-specific fused kernel with
another implementation of the same math) change what is measured. They are acceptable
only when the target metric is not a property of that implementation (accuracy-type
metrics, not throughput, latency, or memory), the replacement is numerically equivalent,
and the diagnosis states the substitution explicitly so the final report can list it as a
deviation. For performance targets, never replace the implementation under test.

When the stage prompt includes a CUDA-to-ROCm porting policy, the `rocm-kernel-porting`
skill governs every ported or CUDA-specific kernel, and its no-fallback rule overrides the
substitution allowance above: never replace a ported kernel with PyTorch, a reference path,
or a disabled extension, never drop a kernel from the porting record, and re-run the parity
check whenever kernel code, build flags, or the image change.

If no legitimate repair exists, do not fabricate one. Explain the root cause and why every
available fix would alter frozen science, and return the manifests as blocked with that
reason so the failure is reported honestly.

## Minimal-repair method

1. State one root-cause hypothesis and the evidence for it.
2. Choose the least invasive surface that fixes it, in this order of preference:
   launch delivery or generated config, then environment or version pin, then a
   compatibility shim outside the repository, then a source patch.
3. Make the smallest diff: no refactors, renames, reformatting, or opportunistic fixes.
   A source patch touches only the lines on the failing path and keeps the computation.
4. When the Dockerfile or dependencies change, rebuild the image, keep the complete build
   log at a new path, and make the manifests reference the new image and log.
5. Keep every frozen value's delivery verifiable: values rendered in the launch script must
   still appear there exactly.

## Smoke test

Prove the repair on the same entry point and code path that failed:

- Run inside the same container image the executor will use, through the repository's
  real entry point. Do not import a different module or call a hand-written substitute.
- Bound it with the repository's own limits (maximum steps, batch limits, subset or
  debug flags) passed only to the smoke invocation. Never add those limits to the launch
  script used for the full run.
- It must advance past the previous failure point. Identify the log line that shows it
  did (for example, the first completed iteration after the failing import or call).
- It must exit with status 0 on its own terms. Do not mask a non-zero exit or rely on a
  timeout kill as success.
- Write its complete, non-empty log below the attempt's repair directory named by the
  stage prompt.
- Do not run the complete experiment unless the experiment itself is a bounded
  measurement.

## Diagnosis record

Write the diagnosis so a reviewer can audit it without rereading the log:

- **Symptom:** the verbatim decisive error line and the phase that failed.
- **Root cause:** the mechanism, with the source file and line or package version.
- **Class:** one class from the table.
- **Repair:** what changed in each file and why it is the minimal fix.
- **Science unchanged:** why no frozen value, protocol, or measured implementation changed,
  or the explicit substitution and its scope.
- **Smoke evidence:** the command shape, its log path, and the line proving progress past
  the old failure.
- **Not tried or rejected:** alternatives you rejected because they would alter science.

Every changed file gets its own change entry with a one-line reason.
