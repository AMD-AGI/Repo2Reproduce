This is the environment-preparation stage of a research-paper reproduction.

Frozen target: {{TARGET_PATH}}
Target SHA-256: {{TARGET_SHA256}}
Validated configuration audit: {{AUDIT_PATH}}
Audit SHA-256: {{AUDIT_SHA256}}
Validated execution specification: {{SPEC_PATH}}
Execution-specification SHA-256: {{SPEC_SHA256}}
Official repository: {{REPOSITORY}}
Working directory on the host: {{WORKSPACE}}
Working directory inside every container: /run/workspace
Available wall-clock time for full execution: {{TIMEOUT_SECONDS}} seconds

The scientific target and resolved configuration are frozen. Do not change the selected
experiment, metric, dataset, model, checkpoint, workload, or any resolved input. Environment
choices such as ROCm image, Python, dependencies, paths, and mounts do not require paper
quotations, but they must preserve the validated scientific execution.

Required preparation:

1. Create and preserve a Dockerfile under the workspace, build a reusable ROCm image, and
   save the complete build output to a non-empty path relative to the run directory.
2. Apply only scientific patches explicitly approved by execution_spec.json. Environment
   compatibility patches must not change the method or metric.
3. Run a small bounded smoke test in the image. It must verify imports, GPU visibility when
   required, required data/checkpoint access, one representative operation on the selected
   execution path, and writing output under /run/artifacts. Save its complete log.
4. Create execution/launch.sh in the host workspace. It must:
   - start with #!/usr/bin/env bash and set -euo pipefail;
   - cd to the exact repository directory under /run/workspace;
   - contain every resolved scientific input from execution_spec.json;
   - run the complete selected experiment synchronously, without &, nohup, detaching, or polling;
   - persist the metric and expected outputs under /run/artifacts/full-run;
   - exit non-zero on failure.
5. Compute SHA-256 hashes for the Dockerfile and launch script.
6. Return an EnvironmentManifest plus a version-2 ExecutionManifest. Copy
   scientific_configuration exactly from execution_spec.json. Include the frozen paper target
   value and a machine-readable JSON-path or regular-expression metric extractor whose source
   is execution_spec.json's result_artifact. Set environment_sha256 to null; the orchestrator
   fills it after persisting the environment manifest.

All installation, downloads, Python execution, smoke tests, and evaluation must occur through
Docker. Do not run the complete selected experiment in this stage.

Return only the structured PreparationStageOutput requested by the output schema.
