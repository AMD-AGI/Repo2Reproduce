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

| # | Name | Paper | Paper link | Official code repository | Porting |
|---:|---|---|---|---|---|
| 1 | bookcoref | BOOKCOREF: Coreference Resolution at Book Scale | [arXiv:2507.12075](https://arxiv.org/abs/2507.12075) | [sapienzanlp/bookcoref](https://github.com/sapienzanlp/bookcoref) | none |
| 2 | verbalized | Evaluating Design Decisions for Dual Encoder-based Entity Disambiguation (VerbalizED) | [arXiv:2505.11683](https://arxiv.org/abs/2505.11683) | [flairNLP/VerbalizED](https://github.com/flairNLP/VerbalizED) | none |
| 3 | qpcert | Exact Certification of (Graph) Neural Networks Against Label Poisoning | [arXiv:2412.00537](https://arxiv.org/abs/2412.00537) | [saper0/qpcert](https://github.com/saper0/qpcert) | none |
| 4 | comres | CoMRes | [OpenReview bRa4JLPzii](https://openreview.net/pdf?id=bRa4JLPzii) | [yjucho1/CoMRes](https://github.com/yjucho1/CoMRes) | none |
| 5 | cogs | Composing Global Solutions to Reasoning Tasks via Algebraic Objects in Neural Nets (CoGS) | [arXiv:2410.01779](https://arxiv.org/abs/2410.01779) | [facebookresearch/luckmatters @ yuandong3, ssl/real-dataset](https://github.com/facebookresearch/luckmatters/tree/yuandong3/ssl/real-dataset) | none |
| 6 | mm-argfallacy | Leveraging Context for Multimodal Fallacy Classification in Political Debates | [arXiv:2507.15641](https://arxiv.org/abs/2507.15641) | [alessiopittiglio/mm-argfallacy](https://github.com/alessiopittiglio/mm-argfallacy) | none |
| 7 | catch | CATCH: Channel-Aware multivariate Time Series Anomaly Detection via Frequency Patching | [arXiv:2410.12261](https://arxiv.org/abs/2410.12261) | [decisionintelligence/CATCH](https://github.com/decisionintelligence/CATCH) | none |
| 8 | kff | Class-aware Domain Knowledge Fusion and Fission for Continual Test-Time Adaptation (KFF) | [arXiv:2510.12150](https://arxiv.org/abs/2510.12150) | [zhoujiahuan1991/NeurIPS2025-KFF](https://github.com/zhoujiahuan1991/NeurIPS2025-KFF) | none |
| 9 | convolutional-diffusion | An analytic theory of creativity in convolutional diffusion models | [arXiv:2412.20292](https://arxiv.org/abs/2412.20292) | [Kambm/convolutional_diffusion](https://github.com/Kambm/convolutional_diffusion) | none |
| 10 | refersplat | ReferSplat: Referring Segmentation in 3D Gaussian Splatting | [arXiv:2508.08252](https://arxiv.org/abs/2508.08252) | [heshuting555/ReferSplat](https://github.com/heshuting555/ReferSplat) | none |
| 11 | robustness-reprogramming | Robustness Reprogramming for Representation Learning | [arXiv:2410.04577](https://arxiv.org/abs/2410.04577) | [chris-hzc/Robustness-Reprogramming](https://github.com/chris-hzc/Robustness-Reprogramming) | none |
| 12 | timeemb | TimeEmb: A Lightweight Static-Dynamic Disentanglement Framework for Time Series Forecasting | [arXiv:2510.00461](https://arxiv.org/abs/2510.00461) | [showmeon/TimeEmb](https://github.com/showmeon/TimeEmb) | none |
| 13 | mcnc | MCNC: Manifold-Constrained Reparameterization for Neural Compression | [arXiv:2406.19301](https://arxiv.org/abs/2406.19301) | [mint-vu/MCNC](https://github.com/mint-vu/MCNC) | none |
| 14 | hymn | Balancing Efficiency and Expressiveness: Subgraph GNNs with Walk-Based Centrality (HyMN) | [arXiv:2501.03113](https://arxiv.org/abs/2501.03113) | [jks17/hymn](https://github.com/jks17/hymn) | none |
| 15 | benign-attention | Benign Overfitting in Token Selection of Attention Mechanism | [arXiv:2409.17625](https://arxiv.org/abs/2409.17625) | [keitaroskmt/benign-attention](https://github.com/keitaroskmt/benign-attention) | none |
| 16 | grokformer | GrokFormer: Graph Fourier Kolmogorov-Arnold Transformers | [arXiv:2411.17296](https://arxiv.org/abs/2411.17296) | [gga23/grokformer](https://github.com/gga23/grokformer) | none |
| 17 | timemcl | Winner-takes-all for Multivariate Probabilistic Time Series Forecasting (TimeMCL) | [arXiv:2506.05515](https://arxiv.org/abs/2506.05515) | [victorletzelter/timemcl](https://github.com/victorletzelter/timemcl) | none |
| 18 | flatquant | FlatQuant: Flatness Matters for LLM Quantization | [arXiv:2410.09426](https://arxiv.org/abs/2410.09426) | [ruikangliu/FlatQuant](https://github.com/ruikangliu/FlatQuant) | none |
| 19 | layerdag | LayerDAG: A Layerwise Autoregressive Diffusion Model for Directed Acyclic Graph Generation | [arXiv:2411.02322](https://arxiv.org/abs/2411.02322) | [Graph-COM/LayerDAG](https://github.com/Graph-COM/LayerDAG) | none |
| 20 | askqe | AskQE: Question Answering as Automatic Evaluation for Machine Translation | [arXiv:2504.11582](https://arxiv.org/abs/2504.11582) | [dayeonki/askqe](https://github.com/dayeonki/askqe) | none |
| 21 | nsnquant | NSNQuant: A Double Normalization Approach for Calibration-Free Low-Bit Vector Quantization of KV Cache | [arXiv:2505.18231](https://arxiv.org/abs/2505.18231) | [DHdroid/NSNQuant](https://github.com/DHdroid/NSNQuant) | force |
| 22 | sageattention | SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration | [arXiv:2410.02367](https://arxiv.org/abs/2410.02367) | [thu-ml/SageAttention](https://github.com/thu-ml/SageAttention) | force |

Titles come from the arXiv API. The CoMRes title could not be retrieved from OpenReview and
is shown by its repository name.

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

## Adding a paper

Append a line with the next `id`, separated by real tab characters, and use `-` for an
empty `target`. Check that each line has seven columns:

```bash
awk -F'\t' '!/^#/ && NF != 7 {print "line " NR ": " NF " columns"}' dataset/papers.tsv
```
