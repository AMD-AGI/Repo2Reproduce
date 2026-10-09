# How to run the staged Claude agent

Run this command directly from the allocated compute node:

```bash
cd /home/machaita/Repo2Reproduce
mkdir -p logs/two-stage-launches

nohup setsid .venv/bin/python -m claude_baseline_two_stage \
  https://arxiv.org/abs/2412.20292 \
  https://github.com/Kambm/convolutional_diffusion \
  --run-id convolutional-diffusion-100sample-r-20261008 \
  --gpu 2 \
  --target "Reproduce the 100-sample MNIST ResNet ELS comparison and report the median pixelwise Pearson correlation r." \
  > logs/two-stage-launches/convolutional-diffusion-100sample-r.log 2>&1 < /dev/null &
```

Replace `--gpu 2` with an available physical GPU index. The command runs in the
background and writes its launcher output to
`logs/two-stage-launches/convolutional-diffusion-100sample-r.log`.

To launch from another node, wrap the same command with:

```bash
spur exec <allocation-id> bash -lc '<command>'
```

The orchestrator persists these audit artifacts before full execution:

- `target.json`
- `configuration_audit.json`
- `execution_spec.json`
- `environment_manifest.json`
- `execution_manifest.json`
- `execution_result.json`
- `final_report.json`
- `final_report.md`

To resume from the first incomplete stage:

```bash
.venv/bin/python -m claude_baseline_two_stage \
  --recover-run /shared_nfs/machita/claude-code-two-stage-runs/<run-id>
```

Use `--prepare-only` to stop after the environment is smoke-tested and the immutable
execution manifest is written.
