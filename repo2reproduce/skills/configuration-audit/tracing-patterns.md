# Tracing patterns by configuration mechanism

Use the section that matches the repository. Each lists where values are declared, how
they are overridden, and the usual traps.

## argparse / click / typer

- Declared by `add_argument("--lr", type=float, default=...)` or `@click.option`.
- Overrides come from the command line in shell scripts, README commands, and
  `parser.set_defaults(...)`. Check for `parse_known_args` plus a second parser.
- Traps: `action="store_true"` flags whose default is the opposite of the paper;
  `type=int` truncation; arguments rewritten after parsing (`args.lr *= world_size`).

## Hydra / OmegaConf

- Declared in `conf/` YAML; the `defaults:` list composes groups (`model: resnet50`,
  `dataset: cifar10`). The selected experiment is often `+experiment=<name>`.
- Resolution order: defaults list in order, then the primary config, then command-line
  overrides. `${...}` interpolations and custom resolvers can derive values.
- Traps: `_self_` position in the defaults list changes precedence; values set in
  `@dataclass` structured configs; overrides of a key absent from the struct need `+key=`.

## MMEngine / MMCV / Detectron2 / yacs style configs

- Python or YAML configs with inheritance (`_base_ = [...]`, `_BASE_:`). Child files
  override parents key by key; `_delete_=True` replaces a whole dict.
- yacs: `cfg.merge_from_file` then `cfg.merge_from_list(opts)`; defaults live in
  `config/defaults.py`.
- Traps: learning rate scaled by GPU count (`auto_scale_lr`), schedule given in
  iterations while the paper reports epochs.

## Hugging Face Transformers / Accelerate / Lightning

- `TrainingArguments` / `Seq2SeqTrainingArguments` defaults (for example lr 5e-5,
  linear schedule, warmup 0) apply unless set; JSON/YAML args files or CLI override them.
- `accelerate` and `torchrun` launch flags set world size, which changes the global batch.
- Lightning: `Trainer(...)` arguments, `LightningCLI` YAML, and `configure_optimizers`.
- Traps: `gradient_accumulation_steps`, `per_device_*_batch_size`, generation
  defaults coming from `generation_config.json` in the checkpoint, tokenizer
  `max_length` and truncation side.

## gin / sacred / ml_collections / custom JSON

- gin: `@gin.configurable` bindings in `.gin` files plus `--gin_bindings`.
- ml_collections: `get_config()` functions with `config_flags` overrides.
- Custom: find the loader (`json.load`, `yaml.safe_load`) and every key read from it.

## Shell launchers and sweeps

- Read every `.sh`, `Makefile`, Slurm script, and sweep YAML referenced by the README.
- Loops over seeds or datasets define the run set and therefore the aggregation.
- Environment variables (`os.environ.get("SEED", 0)`) are inputs when scientific.

## Scientific-input checklist by experiment kind

Use this to catch omissions; include only inputs the selected path consumes.

- **Training then evaluation:** dataset and split, preprocessing and augmentation,
  model and initialization or pretrained checkpoint, optimizer, learning rate and
  schedule, warmup, weight decay, batch size (global), epochs or steps, loss and its
  weights, regularization, checkpoint selection rule, seeds, evaluation split, metric,
  aggregation.
- **Evaluation or inference only:** checkpoint identity and revision, dataset and split,
  number of evaluated samples, preprocessing and prompt or template, decoding settings
  (temperature, top-p, beam width, max length), precision or quantization, metric and
  its normalization, aggregation.
- **Simulation or numerical method:** problem size, discretization, step size, tolerance,
  iteration limit, random seeds, number of trials, reported statistic.
- **Systems or efficiency benchmark:** workload shape (batch, sequence, model size),
  warmup and measured iterations, timing boundaries, precision, reported statistic
  (mean, median, percentile), and the implementation under test.
