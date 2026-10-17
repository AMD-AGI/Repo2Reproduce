This is the target-selection stage of a research-paper reproduction.

Paper Markdown: {{PAPER_MD}}
Structured paper index: {{PAPER_PARSE}}
Requested target: {{TARGET}}

Read the ENTIRE paper Markdown, including appendices. Inspect the repository only to
Do not inspect the repository in this stage. Do not build an environment, install
dependencies, download datasets, or execute the experiment.

Follow the requested target exactly. It may name one scalar, a complete row, or a
multi-seed aggregate. If no target was requested, select one numerical result representing
a central contribution. Define the complete author-stated protocol: dataset and split,
model, checkpoint, seeds, sample count, steps/epochs, evaluation coverage, metric
definition, and aggregation whenever the paper states them.

Create one scientific requirement for every author-stated detail needed to define that
selected result. Each requirement must contain a verbatim paper quotation and locator.
Do not include repository defaults, README values, implementation observations,
environment requirements, or invented values in target.json. If the paper does not state
a needed value, leave it out of the requirements and mention the omission in warnings.
Return blocked only when the requested result cannot be identified in the paper.

Return only the structured TargetContract requested by the output schema.
