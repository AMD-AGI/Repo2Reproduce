# gfx942 porting reference

Target: AMD Instinct MI300X / MI325X (gfx942, wave64, ROCm 7.x, no CUDA or nvcc).

## PyTorch C++ CUDA API to HIP

HIPIFY leaves these without a warning; rewrite them by hand.

| CUDA (as written) | ROCm replacement |
| --- | --- |
| `#include <ATen/cuda/CUDAContext.h>` | `#include <ATen/hip/HIPContext.h>` |
| `at::cuda::getCurrentCUDAStream()` | `at::hip::getCurrentHIPStreamMasqueradingAsCUDA()` |
| `c10::cuda::CUDAGuard` / `OptionalCUDAGuard` | `c10::hip::HIPGuardMasqueradingAsCUDA` / `OptionalHIPGuardMasqueradingAsCUDA` |
| `#include <c10/cuda/CUDAGuard.h>` | `#include <c10/hip/HIPGuard.h>` plus `ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h` |
| `C10_CUDA_KERNEL_LAUNCH_CHECK()` | `C10_HIP_KERNEL_LAUNCH_CHECK()` |
| `C10_CUDA_CHECK(expr)` | `C10_HIP_CHECK(expr)` |
| `#if __CUDA_ARCH__ >= 800` | `#if defined(__HIP_DEVICE_COMPILE__)` or a gfx-architecture macro such as `__gfx942__` |

## Wavefront width

- NVIDIA warps have 32 lanes; gfx942 wavefronts have 64.
- `__shfl_*_sync(0xffffffff, v, offset)` becomes `__shfl_*(v, offset, 32)` when the kernel
  works in 32-lane groups, or `__shfl_*(v, offset)` with logic adjusted for 64 lanes.
- Lane masks and ballots are 64-bit: `__ballot` returns `uint64_t`; use `__popcll` and
  64-bit shifts. `0xffffffff` masks silently cover only half a wavefront.
- `warpSize` is 64 at runtime; constants such as `#define WARP_SIZE 32` must be revisited
  wherever they size shared memory, reductions, or thread-to-data mappings.

## Inline PTX and NVIDIA-only device code

HIPIFY cannot translate inline PTX. Treat these as rewrite signals:

- `mma.sync`, `ldmatrix`, `cp.async`, `<cuda_pipeline_primitives.h>`: tensor-core tiling and
  asynchronous copies. Rewrite the kernel in Triton (`tl.dot` uses MFMA) or with rocWMMA or
  Composable Kernel.
- `ex2.approx`, `rcp.approx` and similar: use `exp2f`, `__frcp_rn`, or the Triton
  equivalents, and account for accuracy differences in the parity tolerance.
- CUTLASS/CuTe device templates have no AMD backend; host-side PyTorch glue may still be
  translated by HIPIFY while the device GEMM is rewritten.

## Missing headers, types, and constants

- `math_constants.h`: define the needed constants (`CUDART_INF_F` becomes
  `__builtin_huge_valf()`).
- `cuda_device_runtime_api.h` and similar CUDA-only headers: remove or replace with
  `hip/hip_runtime.h`.
- Some `atomicAdd` overloads for `__half2` and `__nv_bfloat16` are missing; add small
  device shims with the same semantics.
- FP8 formats differ between vendors (`e4m3fn` versus `e4m3fnuz` on gfx942); confirm the
  exponent bias and NaN encoding before reusing scale factors.

## Compiler quirks

- Stacked `#pragma unroll N` and `#pragma unroll` on one loop is a hipcc error; keep one.
- Drop NVIDIA-only flags (`-gencode`, `--expt-relaxed-constexpr` where unneeded,
  `-Xptxas`); keep `-O3` and add `-D__HIP_PLATFORM_AMD__=1`.

## Shared memory

gfx942 allows 64 KiB of LDS per workgroup. Kernels that opt in to more with
`cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, ...)` must be
retiled, or have their tile sizes reduced, without changing the math.

## Building PyTorch extensions

```python
from torch.utils.cpp_extension import BuildExtension, CppExtension

ext = CppExtension(
    name="ext_name",
    sources=["csrc/pybind.cpp", "csrc/kernel.hip"],
    include_dirs=["/opt/rocm/include"],
    libraries=["c10_hip", "torch_hip", "amdhip64"],
    define_macros=[("__HIP_PLATFORM_AMD__", "1"), ("USE_ROCM", "1")],
    extra_compile_args={"cxx": ["-O3"]},
)
# setup(..., ext_modules=[ext], cmdclass={"build_ext": BuildExtension})
```

- Set `PYTORCH_ROCM_ARCH=gfx942` (the image build has no GPU to detect).
- Import `torch` before the compiled extension so `libc10_hip.so` is loaded.
- Confirm the build log shows hipcc compiling the translated sources and no
  `torch.utils.hipify` mapping lines.
