#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Matrix multiplication kernel with intentional warp size bug
// C = A * B
// A: M x K, B: K x N, C: M x N
__global__ void matrixMultiply(const float *A, const float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
        
        // Intentional warp size bug - assumes 32 threads per warp
        // This will cause incorrect behavior on AMD wavefront (64 threads)
        if (threadIdx.x % 32 == 0 && threadIdx.y % 32 == 0) {
            // This warp-level synchronization only works for CUDA
            printf("Block (%d,%d) warp (%d,%d) computed element (%d,%d) = %f\n", 
                   blockIdx.x, blockIdx.y, threadIdx.x / 32, threadIdx.y / 32, row, col, sum);
        }
    }
}

// Optimized version with shared memory (for comparison)
__global__ void matrixMultiplyShared(const float *A, const float *B, float *C, int M, int N, int K) {
    __shared__ float tileA[32][32];
    __shared__ float tileB[32][32];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    for (int tile = 0; tile < (K + 31) / 32; ++tile) {
        // Load tiles into shared memory
        if (row < M && tile * 32 + threadIdx.x < K) {
            tileA[threadIdx.y][threadIdx.x] = A[row * K + tile * 32 + threadIdx.x];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && tile * 32 + threadIdx.y < K) {
            tileB[threadIdx.y][threadIdx.x] = B[(tile * 32 + threadIdx.y) * N + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product
        for (int k = 0; k < 32; ++k) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

int main(int argc, char **argv) {
    int M = 512;
    int N = 512;
    int K = 512;
    
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(size_A);
    float *h_B = (float *)malloc(size_B);
    float *h_C = (float *)malloc(size_C);
    float *h_C_ref = (float *)malloc(size_C);

    // Initialize matrices
    for (int i = 0; i < M * K; ++i) h_A[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = rand() / (float)RAND_MAX;

    // Allocate device memory
    float *d_A, *d_B, *d_C, *d_C_ref;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    cudaMalloc(&d_C_ref, size_C);

    // Copy to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // Setup kernel launch parameters
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    printf("Matrix dimensions: %dx%d * %dx%d = %dx%d\n", M, K, K, N, M, N);
    printf("Launching kernel with grid (%d,%d) and block (%d,%d)\n", 
           blocksPerGrid.x, blocksPerGrid.y, threadsPerBlock.x, threadsPerBlock.y);

    // Warmup
    matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C_ref, M, N, K);
    cudaDeviceSynchronize();

    // Time the basic kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C_ref, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float basic_time = 0;
    cudaEventElapsedTime(&basic_time, start, stop);
    printf("Basic kernel time: %.3f ms\n", basic_time);

    // Time the shared memory kernel
    cudaEventRecord(start);
    matrixMultiplyShared<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float shared_time = 0;
    cudaEventElapsedTime(&shared_time, start, stop);
    printf("Shared memory kernel time: %.3f ms\n", shared_time);
    
    printf("Speedup: %.2fx\n", basic_time / shared_time);

    // Copy results back
    cudaMemcpy(h_C_ref, d_C_ref, size_C, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    // Verify results
    bool correct = true;
    for (int i = 0; i < M * N; ++i) {
        if (fabs(h_C[i] - h_C_ref[i]) > 1e-5) {
            printf("Mismatch at element %d: %f != %f\n", i, h_C[i], h_C_ref[i]);
            correct = false;
            break;
        }
    }
    
    if (correct) {
        printf("Verification PASSED\n");
    } else {
        printf("Verification FAILED\n");
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_C_ref);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);

    printf("Done\n");
    return 0;
}
