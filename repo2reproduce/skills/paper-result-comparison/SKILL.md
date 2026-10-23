---
name: paper-result-comparison
description: Compare a reproduced research-paper result against the paper-reported value from frozen evidence and decide the verdict. Use when finalizing a reproduction to locate the exact paper table or figure cell, compare scalars, mean-with-spread values, intervals, or vectors, judge baseline applicability and metric direction, compute a transparent relative error, choose match, miss, run_incomplete, or needs_review, and record assumptions, deviations, repairs, and gaps.
---

# Paper-result comparison

Decide, from persisted evidence only, whether the reproduced value matches the paper. You
never launch, repeat, or modify the experiment. The stage prompt supplies the artifact
paths, the frozen values that must be copied unchanged, and the run's match tolerance.

## Evidence handling

Authority, in order:

1. The **frozen target** defines the experiment, the paper value, the metric, and its
   location. Copy those values exactly.
2. The **deterministic execution result** defines the execution status and the achieved
   value. Copy them exactly. Never substitute a value you computed or found in a log.
3. The **metric artifact, execution log, and expected outputs** corroborate the achieved
   value. Check that the artifact contains it, and that the protocol completed: every
   seed, run, dataset, or split the target aggregates over is present, and the evaluated
   sample count matches the protocol.
4. The **configuration audit and execution manifest** describe what was applied.
5. **Repair records and patches** describe every change made to get the run to execute.

Scan the log for signs that invalidate a completed run: skipped batches, truncated data,
NaN or infinite values, fallback code paths, warnings that a requested setting was
ignored, or an evaluation subset. Report each as a deviation or gap, and treat a run that
did not cover the protocol as incomplete.

Label every source accurately: paper (verbatim), code default, experiment config, README,
environment, or agent inference. A README or code value is never "paper-reported".

## Procedure

### 1. Locate the paper result

Find the table or figure, row, column, and cell that hold the target value in the parsed
paper. Confirm the reported string, its units and scaling (percent, x100, milliseconds),
its precision, and any spread notation (plus-minus, parentheses, subscripts). Record what
the spread means when the paper says so (std, standard error, confidence interval).

### 2. Identify the value shape

- **Scalar:** a single number.
- **Scalar with spread:** mean plus-minus std or similar over seeds or runs.
- **Interval:** a confidence interval, a range, or a binned interval such as `(0.25, 0.30]`.
- **Vector:** several labeled values (a table row across datasets, per-class scores).
- **Figure read-off:** a value only visible in a plot without a numeric label.

### 3. Normalize before comparing

Put both values in the same units and scale (fraction versus percent, seconds versus
milliseconds). Pair vector elements by label, never by position alone. Do not round the
achieved value to manufacture agreement; compare the raw achieved value and mention the
paper's reporting precision when it matters.

### 4. Compare by shape

Use the tolerance supplied by the stage prompt, written below as `T` percent.

- **Scalar:** relative error = `|achieved - paper| / |paper| x 100`. Comparable result
  passes when relative error is at most `T`. The rule is symmetric: a result far better
  than the paper still fails the tolerance; say that the deviation is favorable.
  If the paper value is zero, relative error is undefined: report the absolute difference
  and use `needs_review` unless the paper defines a tolerance.
- **Scalar with spread:** compare the means with the scalar rule. Report whether the
  achieved mean also falls within the paper's spread, as context only.
- **Interval:** use the paper's own definition when it gives one. Otherwise, for an
  interval result compare each endpoint with the scalar rule; for a scalar achieved value
  against a reported interval or bin, test membership honoring open and closed bounds.
  Relative error is not meaningful for membership; leave it empty.
- **Vector:** use the paper's own reduction when the claim is about it (for example, the
  average column). Otherwise compare element-wise: every element must pass. Report each
  element's error and the maximum; report the maximum as the relative error only when
  every element's error is defined, and say so in the calculation.
- **Figure read-off:** approximate plot readings are not exact paper values; use
  `needs_review` unless the figure carries numeric labels.

Write the calculation with the numbers substituted, for example
`|91.84 - 92.10| / 92.10 x 100 = 0.28%`. See [worked-examples.md](worked-examples.md).

### 5. Decide baseline applicability and metric direction

A baseline is applicable only if it is reported by the paper for the **same** dataset and
split, metric, configuration, and evaluation protocol as the target, normally in the same
table and column. Prefer the method the paper explicitly claims to beat for this result;
when the claim is over all compared methods, use the strongest of them in that column.

Not applicable when the baseline uses a different setting or protocol, the target itself
is a baseline or ablation, or the paper makes no comparative claim for this result. Then
mark the baseline not applicable and explain why.

Metric direction comes from the paper first (arrows in headers, "lower is better"), then
from metric semantics: error, loss, perplexity, FID, latency, and memory are lower-better;
accuracy, F1, BLEU, AUC, and throughput are higher-better. If direction stays unknown and a
baseline is applicable, the comparison is ambiguous.

For an applicable baseline, the advantage is preserved only when the achieved value is
strictly better than the baseline value in the metric direction. Quote the baseline value
verbatim with its locator.

### 6. Choose the verdict

Apply these in order and stop at the first that holds:

1. **`run_incomplete`:** execution did not succeed, no achieved value was extracted, or the
   required protocol (all seeds, runs, splits, samples) did not complete. Not comparable.
2. **`needs_review`:** the comparison rule is genuinely ambiguous: interval, vector, or
   aggregation semantics are undefined; metric direction is unknown when a baseline
   applies; units cannot be reconciled; the paper value is zero; or two reasonable
   interpretations give different verdicts. Explain the ambiguity. Never choose the
   favorable interpretation.
3. **`match`:** comparable, within tolerance under the shape rule, and (when a baseline
   applies) the baseline advantage is preserved.
4. **`miss`:** comparable, but outside tolerance or the baseline advantage is lost.

## Assumptions, deviations, repairs, and gaps

- **Assumptions:** one record per configuration-audit item whose finding is
  `paper_missing`, referencing its exact audit item id, the fallback value used, and its
  source type (code default, experiment config, README). Add records for scientifically
  relevant environment or evaluation choices the paper does not specify (for example,
  hardware platform, numerical precision, or seeds when the paper is silent). Use "not
  specified" when the paper is silent, "ambiguous" when it is unclear, and "not applicable"
  when the setting has no paper counterpart. Leave routine package details out unless
  they changed or constrained the protocol.
- **Deviations:** every `paper_partially_implemented` or `paper_not_implemented` item, any
  paper/code conflict that could not be resolved in the paper's favor, every protocol or
  environment difference from the paper, and every implementation substitution.
- **Repairs:** every execution repair with its diagnosis, changed files, smoke evidence,
  and repository patch. A repaired failure is never hidden.
- **Gaps:** unverified paper quotes, audit warnings, missing outputs, execution failures,
  and anything you could not check.

## Self-check before returning

- [ ] Frozen target values and the deterministic achieved value are copied exactly.
- [ ] The paper location identifies table or figure, row, column, and cell when they exist.
- [ ] The calculation shows substituted numbers, or states why relative error is not
      meaningful.
- [ ] The baseline is either justified as applicable with a verbatim quote and direction,
      or explicitly marked not applicable with a reason.
- [ ] The verdict follows the ordered rules; ambiguity is named, not resolved favorably.
- [ ] Every `paper_missing` audit item has an assumption record with its id.
- [ ] Repairs, deviations, and gaps are all listed.
