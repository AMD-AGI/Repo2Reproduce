#include <stdio.h>
#include <stdlib.h>

// compile: hipcc -arch=sm_60 -nocudalib reduction.cu

// --- IDE & COMPILER COMPATIBILITY LAYER ---
#if !defined(__CUDACC__) && !defined(__HIPCC__)
    // Mock definitions for IDEs (VS Code, Cursor, etc.) lacking CUDA toolchains
    #define __global__ 
    #define __shared__ 
    #define __syncthreads() 
    struct dim3 { 
        int x, y, z; 
        dim3(int _x = 1, int _y = 1, int _z = 1) : x(_x), y(_y), z(_z) {} 
    };
    typedef unsigned int cudaError_t;
    typedef void* cudaStream_t;
    dim3 threadIdx, blockIdx, blockDim;
    int warpSize = 64;
    #define cudaMalloc(p, s) (0)
    #define cudaFree(p) (0)
    #define cudaMemcpy(d, s, n, k) (0)
    #define cudaMemcpyHostToDevice 1
    #define cudaMemcpyDeviceToHost 2
    #define cudaSuccess 0
    #define cudaDeviceSynchronize() (0)
    #define LAUNCH_REDUCTION(g, b, m, ...) reduction_kernel(__VA_ARGS__)
#else
    // Real kernel launch for NVCC/HIPCC
    #define LAUNCH_REDUCTION(g, b, m, ...) reduction_kernel<<<g, b, m>>>(__VA_ARGS__)
#endif
// ------------------------------------------

// Standard reduction template (first pass: block-level)
__global__ void reduction_kernel(float* g_idata, float* g_odata, unsigned int n) {
    extern __shared__ float sdata[];

    // Each thread loads one element from global to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float mySum = (i < n) ? g_idata[i] : 0;
    if (i + blockDim.x < n) 
        mySum += g_idata[i + blockDim.x];

    sdata[tid] = mySum;
    __syncthreads();

    // Do reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] = mySum = mySum + sdata[tid + s];
        }
        __syncthreads();
    }

    // DELIBERATE WARP-SIZE BUG: Assuming warpSize=32 for final unrolled reduction
    // This will produce incorrect results on AMD (warpSize=64)
    if (tid < 32) {
        volatile float* vsmem = sdata;
        vsmem[tid] = mySum = mySum + vsmem[tid + 32];
        vsmem[tid] = mySum = mySum + vsmem[tid + 16];
        vsmem[tid] = mySum = mySum + vsmem[tid + 8];
        vsmem[tid] = mySum = mySum + vsmem[tid + 4];
        vsmem[tid] = mySum = mySum + vsmem[tid + 2];
        vsmem[tid] = mySum = mySum + vsmem[tid + 1];
    }

    // Write result for this block to global memory
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

int main() {
    const int N = 1048576; // 1M elements
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + (threadsPerBlock * 2) - 1) / (threadsPerBlock * 2);

    float *h_input = (float*)malloc(N * sizeof(float));
    float *h_output = (float*)malloc(blocksPerGrid * sizeof(float));

    for (int i = 0; i < N; i++) h_input[i] = 1.0f;

    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, blocksPerGrid * sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Run kernel
    LAUNCH_REDUCTION(blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float), d_input, d_output, N);

    cudaMemcpy(h_output, d_output, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

    // Final sum on host
    float gpu_sum = 0;
    for (int i = 0; i < blocksPerGrid; i++) gpu_sum += h_output[i];
    float cpu_sum = (float)N;

    printf("Parallel Reduction (1M elements)\n");
    printf("CPU Sum: %.1f\n", cpu_sum);
    printf("GPU Sum: %.1f\n", gpu_sum);
    printf("Result: %s\n", (gpu_sum == cpu_sum) ? "PASS" : "FAIL (Warp size issue suspected)");

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
