This is the environment-preparation stage of a research-paper reproduction.

Frozen target: {{TARGET_PATH}}
Target SHA-256: {{TARGET_SHA256}}
Validated configuration audit: {{AUDIT_PATH}}
Audit SHA-256: {{AUDIT_SHA256}}
Official repository: {{REPOSITORY}}
Repository commit: {{REPOSITORY_COMMIT}}
Working directory on the host: {{WORKSPACE}}
Working directory inside every container: /run/workspace
Available wall-clock time for full execution: {{TIMEOUT_SECONDS}} seconds

You are responsible for translating the canonical scientific configuration into the
repository's actual executable command. Read the target, configuration audit, repository,
README/configs, and the complete selected execution path. Understand repository-specific
CLI, Hydra, YAML, shell, and hard-coded delivery mechanisms. Do not concatenate prose into
an argument: translate each atomic chosen value into the syntax consumed by the code.
Preserve the selected experiment, metric, workload, and all implementable chosen values.

Environment choices such as ROCm image, Python, dependencies, paths, mounts, local helper
packages, generated markers such as .sweep, and data placement are yours to resolve here.
They do not require paper quotations and must not be reported as scientific gaps.

Required preparation:

1. Create and preserve a Dockerfile under the workspace, build a reusable ROCm image, and
   save the complete build output to a non-empty path relative to the run directory.
2. Install or create every environment prerequisite required by the traced path.
   Compatibility patches must not change the scientific method or metric.
3. Run a small bounded smoke test in the image. It must verify imports, GPU visibility when
   required, required data/checkpoint access, one representative operation on the selected
   execution path, and writing output under /run/artifacts. Save its complete log.
4. Create execution/launch.sh in the host workspace. It must:
   - start with #!/usr/bin/env bash and set -euo pipefail;
   - cd to the exact repository directory under /run/workspace;
   - execute every required step, seed, and aggregation for the complete target;
   - apply every implementable chosen value from configuration_audit.json using the
     repository's real delivery mechanism;
   - run the complete selected experiment synchronously, without &, nohup, detaching, or polling;
   - persist the metric and expected outputs under /run/artifacts/full-run;
   - exit non-zero on failure.
5. Compute SHA-256 hashes for the Dockerfile and launch script.
6. Return an EnvironmentManifest plus a version-3 ExecutionManifest. In
   scientific_configuration add one ResolvedInput for every implementable audit item with a
   chosen_value that is consumed as an execution input. Do not add the paper's reported
   result value or other expected outputs as command inputs. Each record must use the
   canonical audit item id, the applied value, its delivery mechanism, and (for CLI
   delivery) the exact substring present in launch.sh.
   For non-CLI delivery, set rendered only when it is also an exact launch.sh substring.
   Include the frozen paper target value and a machine-readable JSON-path or
   regular-expression metric extractor. Select the manifest entrypoint by copying one exact
   entry_points object from the audit. Set environment_sha256 to null; the orchestrator fills
   it after persisting the environment manifest.

Path contract for the returned JSON:

- Every host-side file path must be relative to the run root.
- Paths to prepared files must start with `workspace/`. For example, if you create the
  Dockerfile at `{{WORKSPACE}}/Dockerfile`, return `workspace/Dockerfile`; return the launch
  script as `workspace/execution/launch.sh`; and return the repository as `workspace/repo`.
- Paths to generated logs, metrics, and other outputs must start with `artifacts/`, for
  example `artifacts/env/build.log`, `artifacts/env/smoke.log`, and
  `artifacts/full-run/metric.json`.
- `dockerfile`, `launch_script`, `host_repository`, `build_log_path`, smoke-test `log_path`,
  `expected_outputs`, `metric_artifact`, and metric-extractor `source` are path fields. Never
  place file contents in these fields.
- Never return absolute host paths or `/run/...` container paths in host-side path fields.
  `container_repository` is the sole exception and must be the exact container equivalent
  of `host_repository` (for `workspace/repo`, use `/run/workspace/repo`).

All installation, downloads, Python execution, smoke tests, and evaluation must occur through
Docker. Do not run the complete selected experiment in this stage.

Return only the structured PreparationStageOutput requested by the output schema.
