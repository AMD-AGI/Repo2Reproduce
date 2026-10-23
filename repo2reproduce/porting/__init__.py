"""Deterministic helpers for porting CUDA repositories to ROCm.

* ``scanner``: report-only scan for CUDA constructs that need attention on gfx942.
* ``official_hipify``: translates a copy of a repository with AMD's official HIPIFY
  (``hipify-perl`` from https://github.com/ROCm/HIPIFY) and reports what still needs a rewrite.
* ``provenance``: checks that a ``hipify_with_fixes`` port derives from the pinned HIPIFY
  translation of its CUDA source.

``scanner`` is adapted from ROCmPort-AI (Apache License 2.0); see ``LICENSE-ROCmPort-AI``.
"""
