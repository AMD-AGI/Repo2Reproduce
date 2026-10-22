# Benchmark dataset

`papers.tsv` lists the 22 papers the batch scripts reproduce. It is the single source of
truth: `scripts/run_one.sh`, `scripts/run_all.sh`, and `scripts/list_papers.sh` all read it.

## File format

Tab-separated, one paper per line. Lines starting with `#` are comments; the first
non-comment line is the header.

| Column | Meaning |
| --- | --- |
| `id` | Stable number, used to select a paper (`scripts/run_one.sh 21`). |
| `name` | Short lowercase name, also usable as a selector (`scripts/run_one.sh nsnquant`). |
| `paper_url` | Paper link passed to `r2r` as `paper_url`. |
| `repo_url` | Official code repository passed to `r2r` as `repo_url`. A `/tree/<branch>/<path>` URL selects a subdirectory. |
| `porting` | `none`; `rocm` adds `--rocm-porting`; `force` adds `--rocm-porting --force-kernel-port`. |
| `timeout_seconds` | Limit for the final deterministic execution only (`--timeout-seconds`). |
| `target` | Default `--target`. `-` means no target: the agent picks a central, locally reproducible result. |

A `--target` on the command line always overrides the `target` column.

## Papers

`Area` is the broad kind of paper. `Benchmark aspect` is the result it evaluates: a task
score, a certificate, a generative fidelity measure, or an efficiency quantity.

| # | Name | Paper | Paper link | Official code repository | Area | Benchmark aspect |
|---:|---|---|---|---|---|---|
| 1 | bookcoref | BOOKCOREF: Coreference Resolution at Book Scale | [arXiv:2507.12075](https://arxiv.org/abs/2507.12075) | [sapienzanlp/bookcoref](https://github.com/sapienzanlp/bookcoref) | NLP | Book-scale coreference resolution |
| 2 | verbalized | Evaluating Design Decisions for Dual Encoder-based Entity Disambiguation (VerbalizED) | [arXiv:2505.11683](https://arxiv.org/abs/2505.11683) | [flairNLP/VerbalizED](https://github.com/flairNLP/VerbalizED) | NLP | Entity disambiguation |
| 3 | qpcert | Exact Certification of (Graph) Neural Networks Against Label Poisoning | [arXiv:2412.00537](https://arxiv.org/abs/2412.00537) | [saper0/qpcert](https://github.com/saper0/qpcert) | Robustness | Exact certificates against label poisoning |
| 4 | comres | CoMRes: Semi-Supervised Time Series Forecasting Utilizing Consensus Promotion of Multi-Resolution | [OpenReview bRa4JLPzii](https://openreview.net/pdf?id=bRa4JLPzii) | [yjucho1/CoMRes](https://github.com/yjucho1/CoMRes) | Time series | Semi-supervised forecasting |
| 5 | cogs | Composing Global Solutions to Reasoning Tasks via Algebraic Objects in Neural Nets (CoGS) | [arXiv:2410.01779](https://arxiv.org/abs/2410.01779) | [facebookresearch/luckmatters @ yuandong3, ssl/real-dataset](https://github.com/facebookresearch/luckmatters/tree/yuandong3/ssl/real-dataset) | Learning theory | Compositional reasoning on modular addition |
| 6 | mm-argfallacy | Leveraging Context for Multimodal Fallacy Classification in Political Debates | [arXiv:2507.15641](https://arxiv.org/abs/2507.15641) | [alessiopittiglio/mm-argfallacy](https://github.com/alessiopittiglio/mm-argfallacy) | Multimodal NLP | Fallacy classification in political debates |
| 7 | catch | CATCH: Channel-Aware multivariate Time Series Anomaly Detection via Frequency Patching | [arXiv:2410.12261](https://arxiv.org/abs/2410.12261) | [decisionintelligence/CATCH](https://github.com/decisionintelligence/CATCH) | Time series | Multivariate anomaly detection |
| 8 | kff | Class-aware Domain Knowledge Fusion and Fission for Continual Test-Time Adaptation (KFF) | [arXiv:2510.12150](https://arxiv.org/abs/2510.12150) | [zhoujiahuan1991/NeurIPS2025-KFF](https://github.com/zhoujiahuan1991/NeurIPS2025-KFF) | Adaptation | Continual test-time adaptation |
| 9 | convolutional-diffusion | An analytic theory of creativity in convolutional diffusion models | [arXiv:2412.20292](https://arxiv.org/abs/2412.20292) | [Kambm/convolutional_diffusion](https://github.com/Kambm/convolutional_diffusion) | Generative models | Analytic account of diffusion creativity |
| 10 | refersplat | ReferSplat: Referring Segmentation in 3D Gaussian Splatting | [arXiv:2508.08252](https://arxiv.org/abs/2508.08252) | [heshuting555/ReferSplat](https://github.com/heshuting555/ReferSplat) | 3D vision | Referring segmentation in Gaussian splatting |
| 11 | robustness-reprogramming | Robustness Reprogramming for Representation Learning | [arXiv:2410.04577](https://arxiv.org/abs/2410.04577) | [chris-hzc/Robustness-Reprogramming](https://github.com/chris-hzc/Robustness-Reprogramming) | Representation learning | Robustness of a frozen model under input noise |
| 12 | timeemb | TimeEmb: A Lightweight Static-Dynamic Disentanglement Framework for Time Series Forecasting | [arXiv:2510.00461](https://arxiv.org/abs/2510.00461) | [showmeon/TimeEmb](https://github.com/showmeon/TimeEmb) | Time series | Forecasting under distribution shift |
| 13 | mcnc | MCNC: Manifold-Constrained Reparameterization for Neural Compression | [arXiv:2406.19301](https://arxiv.org/abs/2406.19301) | [mint-vu/MCNC](https://github.com/mint-vu/MCNC) | Compression | Foundation-model weight compression |
| 14 | hymn | Balancing Efficiency and Expressiveness: Subgraph GNNs with Walk-Based Centrality (HyMN) | [arXiv:2501.03113](https://arxiv.org/abs/2501.03113) | [jks17/hymn](https://github.com/jks17/hymn) | Graph learning | Subgraph GNN accuracy versus cost |
| 15 | benign-attention | Benign Overfitting in Token Selection of Attention Mechanism | [arXiv:2409.17625](https://arxiv.org/abs/2409.17625) | [keitaroskmt/benign-attention](https://github.com/keitaroskmt/benign-attention) | Learning theory | Generalization of attention under label noise |
| 16 | grokformer | GrokFormer: Graph Fourier Kolmogorov-Arnold Transformers | [arXiv:2411.17296](https://arxiv.org/abs/2411.17296) | [gga23/grokformer](https://github.com/gga23/grokformer) | Graph learning | Spectral graph representation learning |
| 17 | timemcl | Winner-takes-all for Multivariate Probabilistic Time Series Forecasting (TimeMCL) | [arXiv:2506.05515](https://arxiv.org/abs/2506.05515) | [victorletzelter/timemcl](https://github.com/victorletzelter/timemcl) | Time series | Multivariate probabilistic forecasting |
| 18 | flatquant | FlatQuant: Flatness Matters for LLM Quantization | [arXiv:2410.09426](https://arxiv.org/abs/2410.09426) | [ruikangliu/FlatQuant](https://github.com/ruikangliu/FlatQuant) | LLM efficiency | Post-training weight and activation quantization |
| 19 | layerdag | LayerDAG: A Layerwise Autoregressive Diffusion Model for Directed Acyclic Graph Generation | [arXiv:2411.02322](https://arxiv.org/abs/2411.02322) | [Graph-COM/LayerDAG](https://github.com/Graph-COM/LayerDAG) | Graph generation | Synthetic directed-acyclic-graph generation |
| 20 | askqe | AskQE: Question Answering as Automatic Evaluation for Machine Translation | [arXiv:2504.11582](https://arxiv.org/abs/2504.11582) | [dayeonki/askqe](https://github.com/dayeonki/askqe) | Machine translation | Quality estimation without the target language |
| 21 | nsnquant | NSNQuant: A Double Normalization Approach for Calibration-Free Low-Bit Vector Quantization of KV Cache | [arXiv:2505.18231](https://arxiv.org/abs/2505.18231) | [DHdroid/NSNQuant](https://github.com/DHdroid/NSNQuant) | LLM inference | Low-bit KV-cache quantization (memory and throughput) |
| 22 | sageattention | SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration | [arXiv:2410.02367](https://arxiv.org/abs/2410.02367) | [thu-ml/SageAttention](https://github.com/thu-ml/SageAttention) | LLM inference | 8-bit attention accuracy and speed |

Titles come from the arXiv API. The CoMRes title is taken from the repository citation;
OpenReview did not return it.

## Notes

- **NSNQuant** and **SageAttention** contribute CUDA kernels, so they run with
  `--rocm-porting --force-kernel-port`: the paper's own kernel must run on ROCm through
  HIPIFY plus fixes or a targeted rewrite. See [`PORTING.md`](../PORTING.md).
- The SageAttention repository also hosts SageAttention2 (arXiv:2411.10958) and
  SageAttention3. This entry targets the first paper; change `paper_url` to reproduce a later one.
- **NSNQuant** needs LLaMA2-7B. `meta-llama/Llama-2-7b-hf` is gated on Hugging Face; give the
  agent a token (`HF_TOKEN`) or name an ungated mirror in the target.
- Only rows 5, 21, and 22 ship a default target. Every other paper starts with `-`, letting
  the agent choose. Edit the `target` column, or pass `--target`, to pin a specific result.

## Running

Run these from the repository root after `scripts/setup.sh`. `--gpu` is the physical AMD
GPU index from `rocm-smi`. Each script prints the same options with `--help`.

### One paper

An id or a name selects the row. The dataset supplies the paper URL, repository, porting
mode, timeout, and default target.

```bash
scripts/run_one.sh 9 --gpu 0
scripts/run_one.sh convolutional-diffusion --gpu 0    # the same paper, by name
scripts/run_one.sh nsnquant --gpu 1 --detach          # background; log next to the run
scripts/run_one.sh cogs --dry-run                     # print the r2r command and stop
```

A paper that is not in the dataset:

```bash
scripts/run_one.sh --paper https://arxiv.org/abs/<id> --repo https://github.com/<org>/<repo> --gpu 0
```

| Flag | Effect |
| --- | --- |
| `--gpu N` | Physical GPU. Default `0`. |
| `--target TEXT` | Result to reproduce. Overrides the `target` column. |
| `--no-target` | Ignore the dataset target and let the agent choose. |
| `--porting MODE` | `none`, `rocm`, or `force`. Overrides the `porting` column. |
| `--timeout SECONDS` | Limit for the final execution only. Default is the dataset column, else 14400. |
| `--runs-root DIR` | Where run directories go. Default `$R2R_RUNS_ROOT`, then `.env`, then `~/r2r-runs`. |
| `--run-id ID` | Run directory name. Default `<id>-<name>-<UTC timestamp>`. |
| `--prepare-only` | Freeze the execution manifest and do not execute. |
| `--detach` | Background the run. Follow `tail -f "$R2R_RUNS_ROOT"/<run-id>.log`. |
| `--dry-run` | Print the `r2r` command without running it. |
| `--recover DIR` | Resume an interrupted run from its first incomplete stage. |
| `--` | Pass the remaining arguments through to `r2r`, for example `-- --max-execution-repairs 1`. |

Resume a stopped run, or summarize what finished:

```bash
scripts/run_one.sh --recover "$R2R_RUNS_ROOT/9-convolutional-diffusion-20261022-101500" --gpu 0
scripts/summarize.sh                                              # every run under the runs root
scripts/summarize.sh "$R2R_RUNS_ROOT"/9-convolutional-diffusion-*
scripts/summarize.sh --tsv > results.tsv
```

### Target

`--target` names the table, row, column, or figure. Leave hyperparameters out; the
configuration audit discovers them. It replaces the dataset target for that invocation.
`--no-target` drops the dataset target.

```bash
scripts/run_one.sh cogs --gpu 0 \
  --target "Reproduce Table 2, the d=71 modular-addition row."

scripts/run_one.sh convolutional-diffusion --gpu 0 \
  --target "Reproduce Table 1, the CIFAR10 ResNet ELS correlation."

scripts/run_one.sh cogs --gpu 0 --no-target
```

### Porting

`--porting` overrides the `porting` column. `none` adds nothing. `rocm` adds
`--rocm-porting`. `force` adds `--rocm-porting --force-kernel-port`, so the paper's own
CUDA kernel must run on ROCm. NSNQuant and SageAttention are already `force` in
`papers.tsv`. See [`PORTING.md`](../PORTING.md).

```bash
scripts/run_one.sh sageattention --gpu 2 --porting force \
  --target "Reproduce the attention accuracy (cosine similarity and relative L1 against full-precision attention) of the official SageAttention INT8 CUDA kernel, ported to ROCm."

scripts/run_one.sh flatquant --gpu 0 --porting rocm
scripts/run_one.sh nsnquant --gpu 0 --porting none
```

### All papers

`run_all.sh` walks `papers.tsv` one paper after another on one GPU. A failed paper does
not stop the batch unless you pass `--stop-on-error`.

```bash
scripts/run_all.sh --gpu 0 --detach
tail -f "$R2R_RUNS_ROOT"/batch-*/batch.log
```

```bash
scripts/run_all.sh --gpu 0 --only 5,9,nsnquant     # a subset, in dataset order
scripts/run_all.sh --gpu 0 --skip 10,refersplat    # everything except these
scripts/run_all.sh --gpu 0 --no-targets            # the agent chooses every result
scripts/run_all.sh --gpu 0 --prepare-only          # freeze each manifest, do not execute
scripts/run_all.sh --gpu 0 --stop-on-error         # stop at the first non-zero exit
scripts/run_all.sh --dry-run                       # print each command
```

One batch per GPU, on disjoint subsets:

```bash
scripts/run_all.sh --gpu 0 --only 1,2,3,4,5,6,7,8,9,10,11 --detach
scripts/run_all.sh --gpu 1 --only 12,13,14,15,16,17,18,19,20,21,22 --detach
```

A batch directory `<runs-root>/batch-<UTC timestamp>/` holds `batch.log`, `runs.txt`,
`exit_codes.tsv`, and `summary.txt`.

```bash
scripts/summarize.sh --batch "$R2R_RUNS_ROOT"/batch-20261022-101500
```

Host setup, where run directories and logs are written, and how to read a status or a
verdict are in [`BENCHMARK.md`](../BENCHMARK.md).

## Adding a paper

Append a line with the next `id`, separated by real tab characters, and use `-` for an
empty `target`. Check that each line has seven columns:

```bash
awk -F'\t' '!/^#/ && NF != 7 {print "line " NR ": " NF " columns"}' dataset/papers.tsv
```
