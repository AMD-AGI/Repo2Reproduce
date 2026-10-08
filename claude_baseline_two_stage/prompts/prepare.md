This is stage 1 of a two-stage research-paper reproduction task.

Paper: {{PAPER_URL}}
Official code repository: {{REPO_URL}}
Working directory on the host: {{WORKSPACE}}
Working directory inside every container: /run/workspace
Available wall-clock time for the final execution: {{TIMEOUT_SECONDS}} seconds
Requested target: {{TARGET}}

Study the paper and official repository, then select the least computationally expensive
experiment that produces one representative numerical result. Prepare everything needed to
run it, but DO NOT run the complete selected experiment in this stage.

Required preparation:

1. Clone the official repository below /run/workspace. The clone persists on the host because
   the benchmark Docker wrapper mounts the whole run directory at /run.
2. Prepare a reusable Docker image containing every dependency needed by the selected run.
3. Run a small, bounded smoke test in that image. It must verify imports, GPU visibility when
   required, data access, at least one representative train/eval operation, and writing output
   under /run/artifacts. Save its complete log below /run/artifacts.
4. Create `execution/launch.sh` in the host workspace using a file-writing tool. This script is
   executed INSIDE the prepared container, not on the host. It must:
   - start with `#!/usr/bin/env bash` and `set -euo pipefail`;
   - `cd` to the exact repository directory under /run/workspace;
   - set every required environment variable;
   - contain the complete selected experiment command and all arguments;
   - run synchronously in the foreground, with no `&`, `nohup`, detached process, or polling;
   - write metrics, checkpoints, or other produced result files below /run/artifacts/full-run
     whenever the official program supports file outputs;
   - leave the numerical result visible in stdout so the Python executor log captures it;
   - exit non-zero when training or evaluation fails.
5. Compute the SHA-256 digest of `execution/launch.sh`.
6. List at least one `expected_outputs` file produced below /run/artifacts/full-run. If the
   official program only prints a metric, make the script persist that stdout with `tee`.
7. Return a structured execution manifest only after the smoke test exits successfully.

Docker isolation rules:

- All installation, downloads, cloning, Python, training, and evaluation happen through Docker.
- Do not pass `--rm`, `-v`, `--volume`, `--mount`, `--device`, `--gpus`, `--privileged`,
  `ROCR_VISIBLE_DEVICES`, or `HIP_VISIBLE_DEVICES`; the benchmark wrapper owns those settings.
- Do not depend on a container remaining alive after this stage. The image, repository, data,
  launch script, smoke log, and any required host artifacts must be sufficient for a fresh run.
- Paths in `launch_script` and `host_repository` are relative to the host workspace.
- Paths in `expected_outputs` and `smoke_test.log_path` are relative to /run/artifacts.
- `container_repository` must be the exact absolute directory below /run/workspace.

If preparation or smoke testing cannot succeed, return `status: "blocked"` with a precise
reason. Never claim `status: "ready"` without a real exit-code-zero smoke test and persisted
smoke log.
