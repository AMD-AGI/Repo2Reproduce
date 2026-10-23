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
| `conference` | Venue where the paper was published. |

The target, porting mode, and timeout are chosen per run with `--target`, `--porting`, and
`--timeout`. Without `--target`, the agent picks a central, locally reproducible result.

## Papers

`Area` is the broad kind of paper.

| # | Name | Paper | Paper link | Conference | Official code repository | Area |
|---:|---|---|---|---|---|---|
| 1 | bookcoref | BOOKCOREF: Coreference Resolution at Book Scale | [arXiv:2507.12075](https://arxiv.org/abs/2507.12075) | ACL 2025 | [sapienzanlp/bookcoref](https://github.com/sapienzanlp/bookcoref) | NLP |
| 2 | verbalized | Evaluating Design Decisions for Dual Encoder-based Entity Disambiguation (VerbalizED) | [arXiv:2505.11683](https://arxiv.org/abs/2505.11683) | ACL 2025 | [flairNLP/VerbalizED](https://github.com/flairNLP/VerbalizED) | NLP |
| 3 | qpcert | Exact Certification of (Graph) Neural Networks Against Label Poisoning | [arXiv:2412.00537](https://arxiv.org/abs/2412.00537) | ICLR 2025 | [saper0/qpcert](https://github.com/saper0/qpcert) | Robustness |
| 4 | comres | CoMRes: Semi-Supervised Time Series Forecasting Utilizing Consensus Promotion of Multi-Resolution | [OpenReview bRa4JLPzii](https://openreview.net/pdf?id=bRa4JLPzii) | ICLR 2025 | [yjucho1/CoMRes](https://github.com/yjucho1/CoMRes) | Time series |
| 5 | cogs | Composing Global Solutions to Reasoning Tasks via Algebraic Objects in Neural Nets (CoGS) | [arXiv:2410.01779](https://arxiv.org/abs/2410.01779) | NeurIPS 2025 | [facebookresearch/luckmatters @ yuandong3, ssl/real-dataset](https://github.com/facebookresearch/luckmatters/tree/yuandong3/ssl/real-dataset) | Learning theory |
| 6 | mm-argfallacy | Leveraging Context for Multimodal Fallacy Classification in Political Debates | [arXiv:2507.15641](https://arxiv.org/abs/2507.15641) | ACL 2025 (ArgMining Workshop) | [alessiopittiglio/mm-argfallacy](https://github.com/alessiopittiglio/mm-argfallacy) | Multimodal NLP |
| 7 | catch | CATCH: Channel-Aware multivariate Time Series Anomaly Detection via Frequency Patching | [arXiv:2410.12261](https://arxiv.org/abs/2410.12261) | ICLR 2025 | [decisionintelligence/CATCH](https://github.com/decisionintelligence/CATCH) | Time series |
| 8 | kff | Class-aware Domain Knowledge Fusion and Fission for Continual Test-Time Adaptation (KFF) | [arXiv:2510.12150](https://arxiv.org/abs/2510.12150) | NeurIPS 2025 | [zhoujiahuan1991/NeurIPS2025-KFF](https://github.com/zhoujiahuan1991/NeurIPS2025-KFF) | Adaptation |
| 9 | convolutional-diffusion | An analytic theory of creativity in convolutional diffusion models | [arXiv:2412.20292](https://arxiv.org/abs/2412.20292) | ICML 2025 | [Kambm/convolutional_diffusion](https://github.com/Kambm/convolutional_diffusion) | Generative models |
| 10 | refersplat | ReferSplat: Referring Segmentation in 3D Gaussian Splatting | [arXiv:2508.08252](https://arxiv.org/abs/2508.08252) | ICML 2025 | [heshuting555/ReferSplat](https://github.com/heshuting555/ReferSplat) | 3D vision |
| 11 | robustness-reprogramming | Robustness Reprogramming for Representation Learning | [arXiv:2410.04577](https://arxiv.org/abs/2410.04577) | ICLR 2025 | [chris-hzc/Robustness-Reprogramming](https://github.com/chris-hzc/Robustness-Reprogramming) | Representation learning |
| 12 | timeemb | TimeEmb: A Lightweight Static-Dynamic Disentanglement Framework for Time Series Forecasting | [arXiv:2510.00461](https://arxiv.org/abs/2510.00461) | NeurIPS 2025 | [showmeon/TimeEmb](https://github.com/showmeon/TimeEmb) | Time series |
| 13 | mcnc | MCNC: Manifold-Constrained Reparameterization for Neural Compression | [arXiv:2406.19301](https://arxiv.org/abs/2406.19301) | ICLR 2025 | [mint-vu/MCNC](https://github.com/mint-vu/MCNC) | Compression |
| 14 | hymn | Balancing Efficiency and Expressiveness: Subgraph GNNs with Walk-Based Centrality (HyMN) | [arXiv:2501.03113](https://arxiv.org/abs/2501.03113) | ICML 2025 | [jks17/hymn](https://github.com/jks17/hymn) | Graph learning |
| 15 | benign-attention | Benign Overfitting in Token Selection of Attention Mechanism | [arXiv:2409.17625](https://arxiv.org/abs/2409.17625) | ICML 2025 | [keitaroskmt/benign-attention](https://github.com/keitaroskmt/benign-attention) | Learning theory |
| 16 | grokformer | GrokFormer: Graph Fourier Kolmogorov-Arnold Transformers | [arXiv:2411.17296](https://arxiv.org/abs/2411.17296) | ICML 2025 | [gga23/grokformer](https://github.com/gga23/grokformer) | Graph learning |
| 17 | timemcl | Winner-takes-all for Multivariate Probabilistic Time Series Forecasting (TimeMCL) | [arXiv:2506.05515](https://arxiv.org/abs/2506.05515) | ICML 2025 | [victorletzelter/timemcl](https://github.com/victorletzelter/timemcl) | Time series |
| 18 | flatquant | FlatQuant: Flatness Matters for LLM Quantization | [arXiv:2410.09426](https://arxiv.org/abs/2410.09426) | ICML 2025 | [ruikangliu/FlatQuant](https://github.com/ruikangliu/FlatQuant) | LLM efficiency |
| 19 | layerdag | LayerDAG: A Layerwise Autoregressive Diffusion Model for Directed Acyclic Graph Generation | [arXiv:2411.02322](https://arxiv.org/abs/2411.02322) | ICLR 2025 | [Graph-COM/LayerDAG](https://github.com/Graph-COM/LayerDAG) | Graph generation |
| 20 | askqe | AskQE: Question Answering as Automatic Evaluation for Machine Translation | [arXiv:2504.11582](https://arxiv.org/abs/2504.11582) | ACL 2025 (Findings) | [dayeonki/askqe](https://github.com/dayeonki/askqe) | Machine translation |
| 21 | nsnquant | NSNQuant: A Double Normalization Approach for Calibration-Free Low-Bit Vector Quantization of KV Cache | [arXiv:2505.18231](https://arxiv.org/abs/2505.18231) | NeurIPS 2025 | [DHdroid/NSNQuant](https://github.com/DHdroid/NSNQuant) | LLM inference |
| 22 | sageattention | SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration | [arXiv:2410.02367](https://arxiv.org/abs/2410.02367) | ICLR 2025 | [thu-ml/SageAttention](https://github.com/thu-ml/SageAttention) | LLM inference |


## Notes

- **NSNQuant** and **SageAttention** contribute CUDA kernels. Run them with
  `--porting force` so the paper's own kernel must run on ROCm through HIPIFY plus fixes or
  a targeted rewrite. See [`PORTING.md`](../PORTING.md).
- The SageAttention repository also hosts SageAttention2 (arXiv:2411.10958) and
  SageAttention3. This entry targets the first paper; change `paper_url` to reproduce a later one.
- **NSNQuant** needs LLaMA2-7B. `meta-llama/Llama-2-7b-hf` is gated on Hugging Face; give the
  agent a token (`HF_TOKEN`) or name an ungated mirror in the target.

## Running

Before the first run, copy [`.env.example`](../.env.example) to `.env` and fill in your
provider key, model, runs root, and budget; the [README](../README.md#configuration)
explains each setting. Every flag is listed in [BENCHMARK.md](../BENCHMARK.md#2-run).

**One paper**, by id or name:

```bash
scripts/run_one.sh 9 --gpu 0
scripts/run_one.sh cogs --gpu 0 --target "Reproduce Table 2, the d=71 modular-addition row."
scripts/run_one.sh sageattention --gpu 0 --porting force --max-budget-usd 30
```

| Flag | Description |
| --- | --- |
| `--gpu N` | Physical AMD GPU index (default `0`). |
| `--target TEXT` | Result to reproduce: a table, row, or figure (default: the agent chooses). |
| `--porting MODE` | `none` (default), `rocm`, or `force`. See [PORTING.md](../PORTING.md). |
| `--max-budget-usd USD` | Spending cap for the whole run. Overrides `R2R_MAX_BUDGET_USD`. |
| `--detach` | Run in the background. |

**All papers**, one after another:

```bash
scripts/run_all.sh --gpu 0 --detach
scripts/run_all.sh --gpu 0 --only 5,9,nsnquant --max-budget-usd 30
```

| Flag | Description |
| --- | --- |
| `--gpu N` | GPU for every run. |
| `--only LIST` / `--skip LIST` | Comma-separated ids or names to run or leave out. |
| `--max-budget-usd USD` | Spending cap for each paper's run. |
| `--detach` | Run the batch in the background. |

## Adding a paper

Append a line with the next `id`, separated by real tab characters. Check that each line
has five columns:

```bash
awk -F'\t' '!/^#/ && NF != 5 {print "line " NR ": " NF " columns"}' dataset/papers.tsv
```
