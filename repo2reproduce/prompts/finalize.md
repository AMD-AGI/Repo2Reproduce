The deterministic Python executor has finished the selected experiment. This is a fresh,
evidence-only finalization stage; no prior Claude session state is authoritative.

Before reading evidence, invoke the `{{SKILL}}` skill with the Skill tool and follow its
evidence handling, comparison rules, verdict order, and assumption rules for this entire
stage.

Frozen paper target: {{TARGET_PATH}}
Canonical scientific-configuration audit: {{AUDIT_PATH}}
Environment manifest: {{ENVIRONMENT_PATH}}
Execution manifest: {{MANIFEST_PATH}}
Execution result: {{RESULT_PATH}}
Parsed paper Markdown: {{PAPER_MD}}
Structured paper index: {{PAPER_PARSE}}
Complete execution log: {{LOG_PATH}}
Artifacts directory: {{ARTIFACTS_PATH}}
Execution-repair evidence directory: {{REPAIRS_PATH}}

Run-specific comparison parameters:

- Scalar match tolerance: relative error of at most {{MATCH_TOLERANCE_PERCENT}} percent.
- The selected experiment, metric name, paper target, and result location come from
  {{TARGET_PATH}}; the execution status and achieved value come from {{RESULT_PATH}}.
  Copy them exactly. Python will not reinterpret your scientific decision.

Do not launch or repeat the experiment. Base the final status only on the persisted
manifests, execution result, log, repository, repair evidence, and output artifacts.

Output contract:

- `result_location`: table or figure, row, column, and cell when applicable.
- `baseline`: an applicable paper baseline with method, value, metric direction, and
  verbatim paper evidence, or explicitly not applicable with the reason.
- `comparison`: the calculation, relative error only when meaningful, the baseline
  advantage when a baseline applies, and the verdict.
- `assumptions`: one record per audit item whose finding is `paper_missing`, with its exact
  id in `audit_item_ids`, plus necessary scientific environment or evaluation choices.
- `deviations` and `gaps`: every conflict, partial or unimplemented paper detail, protocol
  or environment deviation, execution repair, failure, missing output, and unverified
  evidence.
- `artifacts`: the full applied configuration is already stored in
  {{MANIFEST_PATH}} under `scientific_configuration`; reference files rather than
  reproducing them. The orchestrator canonicalizes artifact paths from the frozen
  manifests.

Return only the structured FinalReport requested by the output schema.
