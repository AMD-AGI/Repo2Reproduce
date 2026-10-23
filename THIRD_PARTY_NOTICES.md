# Third-party notices

Code authored by Advanced Micro Devices, Inc. in this repository is licensed
under the Apache License, Version 2.0 (see [`LICENSE`](LICENSE) and
[`NOTICE`](NOTICE)).

Third-party code, tools, services, papers, repositories, datasets, and model
artifacts remain governed by their respective licenses and terms. Nothing in
this repository's Apache-2.0 license relicenses those materials.

## 1. Third-party code included in this repository

### ROCmPort-AI static analyzer

- Location: `repo2reproduce/porting/scanner.py`
- Upstream: [tazwaryayyyy/ROCmPort-AI](https://github.com/tazwaryayyyy/ROCmPort-AI)
- Based on commit: `943d16c`
- Copyright: 2026 Tazwar Ahnaf Enan
- License: Apache License 2.0
- License text:
  [`repo2reproduce/porting/LICENSE-ROCmPort-AI`](repo2reproduce/porting/LICENSE-ROCmPort-AI)

The adapted file identifies its upstream source and describes AMD's
modifications.

## 2. Third-party software installed or fetched at runtime

The following components are installed as separate distributions or fetched at
runtime. They are not licensed under Repo2Reproduce's Apache-2.0 license.
Exact installed versions can depend on the user's environment.

### Claude Agent SDK for Python

- Declared dependency: `claude-agent-sdk>=0.2.166`
- Upstream:
  [anthropics/claude-agent-sdk-python](https://github.com/anthropics/claude-agent-sdk-python)
- Software license: MIT
- Service terms:
  [Anthropic Commercial Terms of Service](https://www.anthropic.com/legal/commercial-terms)

Use of the SDK and Anthropic services is also governed by Anthropic's
applicable terms, except where a specific component states a different
license.

### Claude Code CLI and Claude services

The Claude Code CLI is installed separately using Anthropic's official
installer or by the user. The CLI and Claude services are governed by
Anthropic's applicable terms and are not part of the Apache-2.0-licensed
Repo2Reproduce source code.

### ROCm HIPIFY

- Upstream: [ROCm/HIPIFY](https://github.com/ROCm/HIPIFY)
- Release: `rocm-7.2.4`
- Commit: `fd26301dda3164cf82ec956cd7ba8d5adfe59e59`
- License for this pinned release: MIT

Repo2Reproduce clones this pinned release at runtime. HIPIFY is not
redistributed in this repository. A future HIPIFY release can have different
license terms and must be reviewed before the pin is updated.

### OpenDataLoader PDF

- Declared dependency: `opendataloader-pdf>=2.4.0`
- Upstream:
  [opendataloader-project/opendataloader-pdf](https://github.com/opendataloader-project/opendataloader-pdf)
- License for versions 2.0 and later: Apache License 2.0

OpenDataLoader PDF's upstream `LICENSE`, `NOTICE`, and third-party notices
remain applicable to redistributions that include it.

### Other direct Python dependencies

- [GitPython](https://github.com/gitpython-developers/GitPython):
  BSD-3-Clause
- [HTTPX](https://github.com/encode/httpx): BSD-3-Clause
- [Pydantic](https://github.com/pydantic/pydantic): MIT
- [python-dotenv](https://github.com/theskumar/python-dotenv): BSD-3-Clause

These packages are installed as separate distributions. Their upstream
license files and notices control their use and redistribution. Transitive
dependencies installed with them retain their own licenses.

## 3. User-provided and downloaded materials

Repo2Reproduce accepts paper and repository URLs and can clone, inspect,
modify, build, and execute third-party research code. Papers, repositories,
datasets, checkpoints, Docker base images, and other assets are not
redistributed as part of this repository unless explicitly stated.

Users are responsible for:

- complying with the licenses and terms of every paper, repository, dataset,
  model, checkpoint, container image, tool, and service they use;
- determining whether they are permitted to submit source code, paper content,
  or other material to their configured language-model provider;
- preserving upstream copyright and license notices in generated or ported
  code; and
- reviewing applicable licenses before distributing generated Docker images,
  ported repositories, run artifacts, or reports.

Generated CUDA-to-ROCm ports remain subject to the license of the original
source repository.
