# How to run the Claude two-stage agent

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
