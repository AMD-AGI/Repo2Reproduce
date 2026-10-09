This is the target-selection stage of a research-paper reproduction.

Paper Markdown: {{PAPER_MD}}
Structured paper index: {{PAPER_PARSE}}
Official repository: {{REPOSITORY}}
Requested target: {{TARGET}}

Read the ENTIRE paper Markdown, including appendices. Inspect the repository only to
confirm that a candidate result has a plausible local implementation. Do not build an
environment, install dependencies, download datasets, or execute the experiment.

Select one numerical result that best represents a central contribution while remaining
locally reproducible from the official repository. Select one exact table cell or scalar
result, not an entire table. The target still includes the complete protocol behind that
cell: dataset and split, model, checkpoint, seeds, sample count, steps/epochs, evaluation
coverage, metric definition, and aggregation whenever the paper states them.

Create one scientific requirement for every author-stated detail needed to define that
selected result. Each requirement must contain a verbatim paper quotation and locator.
Do not invent missing values. If no central numerical result has plausible code support,
return a blocked target with a precise reason.

Return only the structured TargetContract requested by the output schema.
