---
name: configuration-audit
description: Bidirectional paper-to-code scientific-configuration audit for reproducing one research-paper result from its official repository. Use when discovering the repository entry points and execution path for a frozen paper target, tracing every scientific input the code actually consumes, finding the paper's stated value for each input, preferring implementable paper values over code defaults, and reporting conflicts, paper-missing fallbacks, and partially or unimplemented paper requirements.
---

# Bidirectional configuration audit

Build one canonical ledger of the scientific inputs that determine a single paper result.
The audit runs in two directions that merge into the same ledger:

- **Code to paper:** every scientific input the selected code path consumes is looked up
  in the paper.
- **Paper to code:** every paper requirement for the target is located in the code.

The audit is read-only. It never builds environments, installs packages, or runs the
experiment. The stage prompt names the frozen target, the paper artifacts, the repository,
and the exact output contract; this skill describes how to reason.

## Evidence rules

Every claim in the ledger is grounded or it is not made.

- **Paper evidence** is a verbatim, contiguous quote copied from the parsed paper Markdown,
  with a locator (page, section, table, or figure). Quote the shortest span that contains
  the value and enough context to identify the setting (at least a few words). Do not
  stitch quotes across paragraphs or table cells, do not paraphrase, and avoid spans with
  mangled math or ligature characters. Prefer prose or caption text when a table cell is
  hard to quote exactly.
- **Code evidence** is a real repository-relative file path, an optional locator (line,
  function, flag, or config key), and a short snippet that appears in the file exactly
  (whitespace differences are tolerated; anything else is not). Never invent a flag,
  config key, script, or default.
- A README or code comment is never paper evidence. A paper value stated for a different
  dataset, model, or table is not the paper value for this target; mention it in notes.
- When a fact cannot be grounded, leave the field empty and say so in notes or warnings.

## Procedure

Work through these steps in order. Read [tracing-patterns.md](tracing-patterns.md) when
the repository uses a configuration framework, launcher, or layout you need to trace.

### 1. Discover entry points and the execution path

1. Collect candidate entry points: README or docs "reproduce"/"usage" commands, shell
   scripts and Makefiles, `if __name__ == "__main__"` modules, console scripts in
   `setup.py`/`pyproject.toml`, framework mains (Hydra, Lightning CLI, `accelerate`,
   `torchrun`), notebooks, and experiment-config directories named after paper tables.
2. Select the candidate that produces the frozen target: match its dataset, model,
   method variant, and metric names to the target. Prefer scripts or configs the authors
   tie to the paper result (README sections, file names such as `table2_*.sh`, comments).
   Record rejected candidates and why in warnings when the choice is not obvious.
3. Cover the whole path that yields the number, not just the training script: data
   preparation, training or fine-tuning, checkpoint selection, inference, metric
   computation, and multi-seed or multi-run aggregation. If any step is missing from the
   repository (for example, aggregation exists only in the paper), record it.
4. Record the path in execution order: each step's role, the code that implements it,
   what it consumes, and what it produces (files handed to the next step).

### 2. Trace every consumed scientific input

Follow the selected path from its entry point through argument parsing, config loading,
and the functions that use each value. For every input that can change the result:

- Resolve precedence to find the **effective** value. Typically: function default <
  parser default < base config < experiment config < shell/CLI override < runtime
  reassignment in code. A later layer silently wins; look for `args.x = ...`,
  `cfg.merge_from_list`, `setattr`, and environment-variable overrides.
- Record the lowest-level code default and the value the documented experiment path
  actually uses as separate facts. Do not collapse them.
- Record where the value is consumed, not only where it is declared. Options that are
  parsed but never read on the selected path are not inputs; leave them out.
- Expand derived values into their parts: global batch = per-device batch x devices x
  gradient accumulation; steps = epochs x dataset size / batch; warmup ratio versus
  warmup steps; learning-rate schedule and its endpoints.
- Include hard-coded constants that act as hyperparameters: thresholds, temperatures,
  loss weights, sample counts, image sizes, sequence lengths, beam widths, top-k, number
  of evaluation samples, and seeds.
- Capture the metric exactly as implemented: library function, averaging (micro, macro,
  weighted, per-sample then mean), filtering, normalization of predictions, rounding,
  percent scaling, and how runs or seeds are aggregated (mean, median, best, std).
- Classify only scientific inputs. Numerical precision, quantization, and determinism
  settings are scientific when they change computed values or the paper states them.
  Worker counts, logging, output directories, cache paths, device selection, and package
  versions are environment concerns and stay out of this audit.

### 3. Find the paper value for each code input (code to paper)

- Read the experimental-setup, implementation-details, and appendix sections in full,
  plus captions and footnotes of the target table or figure. Search synonyms: learning
  rate / step size / lr; epochs / iterations / steps; batch size / mini-batch; warmup;
  seeds / runs / trials.
- Keep only values that apply to the selected experiment. When the paper states
  different values per dataset or model, choose the one for the target setting.
- Normalize before comparing: scientific notation, units, percent versus fraction,
  per-device versus global batch, epochs versus steps, inclusive versus exclusive ranges.
  Two spellings of the same value are a match, not a conflict.
- A hyperparameter search range ("we searched over {1e-4, 3e-4}") without the selected
  value is not a single paper value. Treat it as unstated, and quote the range in notes.

### 4. Map every paper requirement (paper to code)

Iterate every requirement in the frozen target. Attach each requirement id to the ledger
item that implements it. When no traced input represents a requirement, search the code
specifically for it; if it is still absent, add an item for it with no code value. A
single requirement may map to several atomic items, and an item may serve several
requirements.

### 5. Classify each item and choose the value

Prefer the author-reported paper value whenever the implementation can accept it, meaning
the value can be delivered through an existing flag, config key, or constant without
writing new method code.

| Situation | Finding | Chosen value |
| --- | --- | --- |
| Paper and effective experiment value agree (after normalization) | `match` | the shared value |
| Paper states a value; code default or experiment config differs; code accepts the paper value | `paper_preferred` | the paper value |
| Code consumes an input the paper never states | `paper_missing` | the experiment-code value, as an explicit fallback |
| Paper requirement is only partly implemented (for example, a schedule without its warmup, or one of two loss terms) | `paper_partially_implemented` | the implementable part; describe the missing part in notes |
| Paper requirement has no implementation at all | `paper_not_implemented` | none |

Rules for choosing:

- A paper value that needs new method code (a new loss, augmentation, architecture
  change, or metric) is not implementable; classify as partially or not implemented
  rather than inventing a delivery.
- When paper text is internally inconsistent (prose versus table), choose the value
  closest to the target table, quote it, and state the contradiction in notes and in a
  warning.
- When the effective code value cannot be determined statically (computed at runtime from
  data), record the expression and its inputs, explain it in notes, and add a warning.
- Never let a discrepancy block the audit. Discrepancies are findings for the final report.

### 6. Report conflicts

For every item other than `match`, the notes state: the paper value and its source, the
code default and the experiment value with their sources, which value was chosen and why,
and the expected effect on the result when it is known. Put audit-wide issues in warnings:
ambiguous entry-point choice, README contradicting code, unresolved runtime values,
requirements that map to nothing, and paper statements that could not be quoted verbatim.

## Ledger discipline

- Use exactly one ledger. Never produce separate paper-to-code and code-to-paper lists.
- Keep items atomic: optimizer name, learning rate, weight decay, schedule, and warmup are
  separate items when they are separate executable inputs.
- Give items stable, descriptive ids (for example `optimizer-learning-rate`) that later
  stages can reference.
- Do not include the expected result value, output paths, or environment settings as
  inputs.

## Self-check before returning

- [ ] The execution path reaches the exact metric and aggregation the target reports.
- [ ] Every traced input has code evidence whose snippet exists verbatim in the cited file.
- [ ] Every paper value has a verbatim quote and locator from the parsed paper.
- [ ] Every target requirement id appears in at least one item.
- [ ] Every `paper_missing` item has an explicit fallback chosen value.
- [ ] Every `paper_preferred` item chooses the paper value, and the notes name the
      overridden code value.
- [ ] No environment concern appears in the ledger.
