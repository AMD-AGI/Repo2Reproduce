The deterministic Python executor has finished the selected experiment. This is a fresh,
evidence-only finalization stage; no prior Claude session state is authoritative.

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

Read those files. Inspect any expected result files named by the manifest. Then provide the
final reproduction status:

- exact paper result location: table or figure, row, column, and cell when applicable;
- paper-reported value;
- achieved value;
- an applicable paper baseline only when it uses the same dataset, metric, configuration,
  and evaluation protocol; include its method, value, metric direction, and verbatim paper
  evidence. If none exists, explicitly mark the baseline not applicable;
- a transparent metric calculation, relative error when meaningful, and verdict;
- every paper/code conflict, paper-missing code fallback, partially implemented or
  unimplemented paper detail, unverified evidence, and protocol/environment deviation;
- any execution failure, missing output, limitation, or assumption.
- any execution repairs, including their diagnosis, changed files, smoke evidence, and
  repository patch; do not hide a repaired failure.

Apply this verdict policy:

- `match`: the selected experiment completed and produced a comparable primary value. For
  scalar values, relative error must be at most 5 percent. For intervals or vectors, use
  the paper-defined comparison or the direct element-wise/interval equivalent and state it
  explicitly. The result must also remain better than the applicable paper baseline in the
  correct metric direction;
- `miss`: a comparable completed result violates the applicable scalar or non-scalar
  comparison condition, or fails to preserve the applicable baseline advantage;
- when no applicable paper baseline exists, apply only the 5 percent condition;
- `run_incomplete`: execution or the required protocol did not complete;
- `needs_review`: the paper leaves interval, vector, aggregation, metric direction, or
  another comparison rule genuinely ambiguous. Explain the ambiguity rather than inventing
  a scalar or choosing a favorable interpretation.

For intervals and vectors, state the exact element-wise, interval-membership, or reduction
calculation used. Do not report relative error when it is not mathematically meaningful.
Make the scientific decision yourself from the frozen evidence; Python will not reinterpret
it.

Create one assumption record for every configuration-audit item whose finding is
`paper_missing`, and include its exact audit item id in `audit_item_ids`. Also record
scientifically relevant environment or evaluation choices that were necessary but not
specified by the paper. Distinguish code defaults, experiment configs, README guidance,
environment choices, and agent inference. Do not call a README or code value
paper-reported. Keep routine package details out unless they changed or constrained the
protocol.

The full applied configuration is already stored in execution_manifest.json under
`scientific_configuration`; reference that file rather than reproducing its lengthy
contents. Reference the environment manifest, Dockerfile, build log, launch script and
hash, metric artifact, execution result, executor log, and assumptions file. The
orchestrator will canonicalize these paths from the frozen manifests.

Do not launch or repeat the experiment. Base the final status only on the persisted manifest,
execution result, log, repository, and output artifacts.

The selected experiment, metric name, paper target, execution status, and achieved value must
exactly match the frozen target and deterministic execution result. Return only the structured
FinalReport requested by the output schema.
