This is the scientific configuration-audit stage of a reproduction.

Frozen target: {{TARGET_PATH}}
Target SHA-256: {{TARGET_SHA256}}
Paper Markdown: {{PAPER_MD}}
Structured paper index: {{PAPER_PARSE}}
Official repository: {{REPOSITORY}}
Repository path relative to the workspace: {{REPOSITORY_RELATIVE}}
Repository commit: {{REPOSITORY_COMMIT}}

Do not build an environment or run the experiment.

Build one canonical scientific-configuration ledger in this order:

1. Discover all repository entry points and steps needed for this target, including
   training, validation, analysis, metric calculation, and multi-run aggregation when
   applicable. Record the selected execution path in order.
2. Trace that path and inventory every consumed SCIENTIFIC input: dataset and
   preprocessing, model, optimizer, learning rate, loss, epochs or steps, seeds,
   checkpoints, thresholds, metrics, aggregation, and relevant hard-coded values.
3. For each code input, find what the paper states. Preserve separate code defaults and
   recorded/documented experiment values. Attach verbatim paper evidence when a paper
   value exists and exact repository-relative code evidence.
4. Iterate every requirement in target.json and map its id into the same ledger through
   requirement_ids. Add an item when no existing item represents it.

Use exactly one configuration array; do not produce separate paper-to-code and
code-to-paper arrays. Keep each item atomic: for example, optimizer name and learning rate
are separate items when they are separate executable inputs.

Choose the author-reported paper value whenever the implementation can accept it. Use only
these findings:

- match: paper and experiment code agree;
- paper_preferred: paper differs from a default/documented value and the paper value is
  selected;
- paper_missing: the code consumes an input the paper does not state; select the
  experiment-code value as the explicit fallback;
- paper_partially_implemented: some, but not all, of the paper requirement is implemented;
- paper_not_implemented: the paper requirement has no implementation.

Scientific discrepancies never block this stage. Record them for the final report.
Do not inspect or report dependency installation, Docker/ROCm/Python versions,
common_utils, .sweep markers, filesystem mounts, package availability, or any other
environment concern; those belong exclusively to environment preparation.

Set repository_path to {{REPOSITORY_RELATIVE}} and repository_commit to
{{REPOSITORY_COMMIT}}. Code evidence paths are relative to that repository and include
short exact snippets. Return only AuditStageOutput with its single `audit` object. Do not
create an execution specification or command in this stage.

Return only the structured AuditStageOutput requested by the output schema.
