This is the scientific configuration-audit stage of a reproduction.

Before reading any evidence, invoke the `{{SKILL}}` skill with the Skill tool and follow
its procedure, evidence rules, and finding definitions for this entire stage.

Frozen target: {{TARGET_PATH}}
Target SHA-256: {{TARGET_SHA256}}
Paper Markdown: {{PAPER_MD}}
Structured paper index: {{PAPER_PARSE}}
Official repository: {{REPOSITORY}}
Repository path relative to the workspace: {{REPOSITORY_RELATIVE}}
Repository commit: {{REPOSITORY_COMMIT}}

Stage boundaries:

- Do not build an environment, run the experiment, or create an execution specification
  or command.
- Do not inspect or report dependency installation, Docker/ROCm/Python versions,
  common_utils, .sweep markers, filesystem mounts, package availability, or any other
  environment concern; those belong exclusively to environment preparation.
- Scientific discrepancies never block this stage. Record them in the ledger.

Output contract:

- Return one AuditStageOutput whose single `audit` object holds the canonical ledger in
  its one `configuration` array, plus `entry_points` and the ordered `execution_path`.
- Set `target_sha256` to {{TARGET_SHA256}}, `repository_path` to {{REPOSITORY_RELATIVE}},
  and `repository_commit` to {{REPOSITORY_COMMIT}}.
- Code evidence paths are relative to {{REPOSITORY_RELATIVE}} and include short exact
  snippets.
- Every requirement id in {{TARGET_PATH}} appears in at least one item's
  `requirement_ids`.

Return only the structured AuditStageOutput requested by the output schema.
