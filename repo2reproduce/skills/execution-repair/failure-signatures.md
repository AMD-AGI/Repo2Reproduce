# Common failure signatures

Each entry names the signature, the usual root cause, and the preferred minimal repair.
Always confirm the cause in the log and code before applying the repair.

## Environment and build

- `ModuleNotFoundError: No module named 'X'`: dependency missing from the image. Add the
  package at the version the repository pins (requirements file, `setup.py`, lockfile).
- `ImportError: ... undefined symbol` or `version 'GLIBCXX_...' not found`: binary built
  against a different framework or system library. Install a build matching the image's
  framework version, or build from source in the Dockerfile.
- `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`: pin
  `numpy<2` or upgrade the compiled dependency.
- GPU-only extension unavailable on this platform (vendor-specific fused kernels, custom
  CUDA extensions without a HIP build): look for an official build for the platform or a
  documented build flag first. A different implementation is a substitution; follow the
  substitution rule in the skill.
- `hipErrorNoBinaryForGpu`, `invalid device function`, or
  `no kernel image is available for execution on the device`: kernels compiled for another
  GPU architecture. Rebuild for the target architecture in the Dockerfile.
- `torch.cuda.is_available()` is False or no devices are visible inside the container:
  the container is missing device access or the framework build lacks GPU support. Fix the
  image or the run wrapper's device options, not the code.

## Dependency API drift

- `AttributeError: module 'numpy' has no attribute 'float'` (or `int`, `bool`): the code
  predates NumPy 1.24. Pin `numpy<1.24` when the rest of the stack allows it; otherwise
  patch the alias to the builtin type, which is numerically identical.
- `TypeError: __init__() got an unexpected keyword argument` from a library class: the
  installed library is newer or older than the code expects. Pin the version the code was
  written against before patching call sites.
- Removed scheduler, optimizer, or tokenizer arguments in newer framework releases: pin.

## Launch and delivery

- `error: unrecognized arguments`: flag spelled differently in the parser. Read the
  parser and render the exact flag name; never drop the value.
- Hydra `Could not override 'key'. ... Key 'key' is not in struct`: the key does not exist
  in the composed config; use the correct path or `+key=value` only when the code reads it.
- `FileNotFoundError` for a relative path: wrong working directory. `cd` to the directory
  the code expects in the launch script.
- Shell quoting splitting a value (lists, JSON, spaces): quote or move the value into a
  generated config file.

## Data and checkpoint access

- HTTP 401/403 or gated-repository errors: authentication or license acceptance is
  required. Use provided credentials through the environment; never switch to an
  ungated substitute checkpoint.
- Dataset expected at a fixed path: place or symlink the same dataset at that path, or
  pass the path through the existing path option.

## Resource

- `out of memory` on the GPU: first set allocator options or reduce fragmentation; reduce
  only non-audited knobs (for example, evaluation micro-batch size when it is not an
  audited value and the metric is per-sample and order-independent). If the only fix
  changes an audited value, the failure is not repairable under these rules.
- `Bus error` or DataLoader worker killed: shared memory too small; raise the container's
  shared-memory size or lower worker count.

## Output and metric extraction

- Run finished but expected output is missing: compare the paths the code writes with the
  manifest's expected outputs; fix the output location or the expected path, using the
  artifact the code really produced.
- Metric extraction error: open the real metric artifact and correct the selector or
  pattern to read the value it already contains. Never compute or edit the value.

## Scientific failures

- NaN or divergent loss, or a metric assertion: check for platform causes first
  (precision differences, nondeterministic kernels, an uninitialized buffer in a patched
  extension). If the cause is the method itself under the frozen configuration, there is
  no legitimate repair; report it.
