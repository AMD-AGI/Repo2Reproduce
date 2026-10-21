# Worked comparison examples

All examples assume a tolerance `T = 5` percent. Values are illustrative.

## Scalar, higher is better, applicable baseline

- Paper: 92.10 (accuracy, %), Table 2, row "Ours", column "CIFAR-10".
- Baseline in the same column: "ResNet-50" 90.40.
- Achieved: 91.84.
- Calculation: `|91.84 - 92.10| / 92.10 x 100 = 0.28%`; 91.84 > 90.40.
- Verdict: `match` (within tolerance, baseline advantage preserved).

## Scalar within tolerance but baseline advantage lost

- Paper: 25.3 (perplexity, lower is better). Baseline: 25.9. Achieved: 26.1.
- Calculation: `|26.1 - 25.3| / 25.3 x 100 = 3.16%`; 26.1 is not lower than 25.9.
- Verdict: `miss` (the paper's claim over the baseline is not reproduced).

## Favorable deviation outside tolerance

- Paper: 40.0 BLEU. Achieved: 43.0.
- Calculation: `|43.0 - 40.0| / 40.0 x 100 = 7.5%`.
- Verdict: `miss` under the symmetric tolerance; the explanation states the deviation is
  favorable and suggests possible causes (for example, a newer checkpoint revision).

## Mean with spread over seeds

- Paper: `78.4 ± 0.6` over 5 seeds. Achieved mean over 5 seeds: 77.9.
- Calculation: `|77.9 - 78.4| / 78.4 x 100 = 0.64%`; within the reported std as context.
- Verdict: `match`. If only 3 of 5 seeds completed, the verdict is `run_incomplete`.

## Scalar against a binned interval

- Paper: certified radius falls in `(0.25, 0.30]`. Achieved: 0.30.
- Calculation: `0.25 < 0.30 <= 0.30`, so the value is inside the bin; relative error not
  meaningful.
- Verdict: `match`. An achieved 0.25 would be outside because the lower bound is open.

## Interval against interval

- Paper 95% CI: `[0.61, 0.69]`. Achieved CI: `[0.60, 0.70]`.
- Calculation: lower `|0.60 - 0.61| / 0.61 x 100 = 1.64%`; upper
  `|0.70 - 0.69| / 0.69 x 100 = 1.45%`; maximum 1.64%.
- Verdict: `match`, stating that endpoint-wise comparison was used because the paper
  defines no interval comparison.

## Vector row

- Paper row: `{MNLI: 86.1, QNLI: 92.3, SST-2: 94.0}`. Achieved:
  `{MNLI: 85.0, QNLI: 92.9, SST-2: 88.5}`.
- Element errors: MNLI 1.28%, QNLI 0.65%, SST-2 5.85%; maximum 5.85%.
- Verdict: `miss` (SST-2 exceeds tolerance). If the paper's claim were only about the
  average column, compare the averages instead and say so.

## Genuinely ambiguous

- Paper reports "about 3x faster" in prose and a bar chart without labels. Achieved
  speedup: 2.7x.
- Verdict: `needs_review`. "About 3x" defines no exact value, and reading the chart would
  be an approximation; explain both and do not pick the favorable reading.
