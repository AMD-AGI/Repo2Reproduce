This is execution-repair attempt {{REPAIR_ATTEMPT}} of {{MAX_REPAIRS}} for a failed
research-paper reproduction.

Before diagnosing, invoke the `{{SKILL}}` skill with the Skill tool and follow its
evidence handling, failure classification, immutable-science rules, minimal-repair method,
and smoke-test requirements for this entire stage.

Immutable paper target: {{TARGET_PATH}}
Immutable configuration audit: {{AUDIT_PATH}}
Current environment manifest: {{ENVIRONMENT_PATH}}
Current execution manifest: {{MANIFEST_PATH}}
Failed execution result: {{RESULT_PATH}}
Complete failed execution log: {{LOG_PATH}}
Official repository: {{REPOSITORY}}
Pinned repository base commit: {{REPOSITORY_COMMIT}}
Previous repairs directory: {{REPAIRS_PATH}}

Frozen for this run:

- {{TARGET_PATH}} and {{AUDIT_PATH}} are never edited;
- every `(audit_item_id, value)` pair in `scientific_configuration` of
  {{MANIFEST_PATH}} keeps its value;
- the repository stays on base commit {{REPOSITORY_COMMIT}}.

Do not restart target selection or configuration audit. The repair surface is the
Dockerfile, dependencies, launch script, generated execution configs, metric extraction,
and, when necessary, minimal repository source code. All installation, Python execution,
and smoke tests run through Docker.

Write the smoke-test log below `artifacts/execution-repairs/repair-{{REPAIR_ATTEMPT}}/`.

Output contract:

1. the updated EnvironmentManifest and version-3 ExecutionManifest;
2. the diagnosis and one RepairChange for every changed file;
3. successful smoke-test evidence.

Use run-root-relative paths, following the same path rules as environment preparation.
The orchestrator computes the authoritative failure fingerprint, file hashes, repository
diff, changed-file list, source hashes, and manifest source hashes. Return only the
structured ExecutionRepairOutput requested by the output schema.
