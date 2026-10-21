This is execution-repair attempt {{REPAIR_ATTEMPT}} of {{MAX_REPAIRS}} for a failed
research-paper reproduction.

Immutable paper target: {{TARGET_PATH}}
Immutable configuration audit: {{AUDIT_PATH}}
Current environment manifest: {{ENVIRONMENT_PATH}}
Current execution manifest: {{MANIFEST_PATH}}
Failed execution result: {{RESULT_PATH}}
Complete failed execution log: {{LOG_PATH}}
Official repository: {{REPOSITORY}}
Pinned repository base commit: {{REPOSITORY_COMMIT}}
Previous repairs directory: {{REPAIRS_PATH}}

Read all of those files and diagnose the concrete execution failure. Apply the smallest
repair that makes the frozen experiment executable. You may modify the Dockerfile,
dependencies, launch script, generated execution configs, metric extraction, and—when
necessary—minimal repository source code. Do not restart target selection or configuration
audit.

The following scientific state is immutable:

- selected experiment, dataset, method, model, metric, paper target, and repository base
  commit;
- every `(audit_item_id, value)` pair in the current execution manifest;
- seeds, sample count, checkpoints, training duration, evaluation protocol, aggregation,
  and metric definition.

You may change how an immutable value is delivered (for example CLI to generated config),
but never change the value itself. Do not weaken the workload, suppress a real failure,
substitute a different metric, use a precomputed expected answer, or edit target.json or
configuration_audit.json.

After repairing, run a bounded smoke test through the same repository entrypoint and code
path that failed. It must advance beyond the previous failure and write a non-empty log
below `artifacts/execution-repairs/repair-{{REPAIR_ATTEMPT}}/`. Do not run the complete
experiment unless the selected experiment is itself a bounded measurement.

Return:

1. the updated EnvironmentManifest and version-3 ExecutionManifest;
2. a concise diagnosis and one RepairChange for every changed file;
3. successful smoke-test evidence.

Use run-root-relative paths. The orchestrator computes the authoritative failure
fingerprint, file hashes, repository diff, changed-file list, source hashes, and manifest
source hashes. Return only the structured ExecutionRepairOutput requested by the output
schema.
