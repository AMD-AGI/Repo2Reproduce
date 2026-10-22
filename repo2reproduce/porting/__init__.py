"""Deterministic helpers for porting CUDA repositories to ROCm.

* ``scanner``: report-only scan for CUDA constructs that need attention on gfx942.
* ``official_hipify``: translates a copy of a repository with AMD's official HIPIFY
  (``hipify-perl`` from https://github.com/ROCm/HIPIFY) and reports what still needs a rewrite.
* ``profiler``: rocprofv3 kernel timings inside a ROCm container, warm-up plus median runs.

``fixtures/*.cu`` are unmodified copies of ROCmPort-AI's ``backend/demo_kernels``
(https://github.com/tazwaryayyyy/ROCmPort-AI, commit 943d16c), Copyright 2026 Tazwar Ahnaf
Enan, Apache License 2.0; see ``fixtures/LICENSE-ROCmPort-AI``.
"""
