This is the scientific configuration-audit stage of a reproduction.

Frozen target: {{TARGET_PATH}}
Target SHA-256: {{TARGET_SHA256}}
Paper Markdown: {{PAPER_MD}}
Structured paper index: {{PAPER_PARSE}}
Official repository: {{REPOSITORY}}
Repository path relative to the workspace: {{REPOSITORY_RELATIVE}}
Repository commit: {{REPOSITORY_COMMIT}}

Do not build an environment or run the experiment.

Find the selected experiment's real entry point. Trace the reproduction execution path
from that entry point through every computation that produces the reported metric or
artifact. This path may be training, inference, simulation, optimization, data analysis,
benchmarking, theorem checking, hardware measurement, or another experiment type.

Perform both directions of the audit:

1. Code to paper: inventory every scientific input consumed along the execution path.
   Include CLI arguments, config values, defaults, hard-coded constants, datasets,
   preprocessing, models, checkpoints, seeds, sample counts, thresholds, metric settings,
   repetitions, and aggregation. For each input, find the paper value and a verbatim
   quotation. When the paper does not state it, mark it code_only_assumption.
2. Paper to code: map every requirement in target.json to the exact code that consumes it
   and record the runtime value. If a paper detail is not implemented, record
   paper_detail_not_implemented.

For a conflict, resolve to the paper value only if the code can consume that value.
Otherwise mark the audit blocked and describe the minimal scientific patch that would be
required. Do not silently prefer a code default.

Code evidence paths must be relative to the repository and include short exact snippets.
Set both completeness declarations true only after every discovered code input and every
target requirement is represented. The execution specification must contain every
resolved code-to-paper audit item exactly once, a repository-relative result artifact, and
enough command detail for the environment stage to create the final launch script.

Return only the structured AuditStageOutput requested by the output schema.
