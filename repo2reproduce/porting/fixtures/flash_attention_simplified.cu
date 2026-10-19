#include <hip/hip_runtime.h>
#include <math.h>
#include <stdio.h>

#define BLOCK_SIZE 32
#define HEAD_DIM 64

__global__ void flash_attention_forward(
    const float* Q, const float* K, const float* V,
    float* O, float* L,
    int seq_len, int head_dim, float scale
) {
    extern __shared__ float sram[];
    float* q_tile = sram;
    float* k_tile = sram + BLOCK_SIZE * HEAD_DIM;
    float* v_tile = k_tile + BLOCK_SIZE * HEAD_DIM;
    float* s_tile = v_tile + BLOCK_SIZE * HEAD_DIM;

    int tid = threadIdx.x;
    int block_row = blockIdx.x;

    for (int d = tid; d < head_dim; d += BLOCK_SIZE)
        q_tile[tid * HEAD_DIM + d] = Q[block_row * BLOCK_SIZE * head_dim + tid * head_dim + d];
    __syncthreads();

    float row_max = -1e9f, row_sum = 0.0f;
    float acc[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; d++) acc[d] = 0.0f;

    for (int block_col = 0; block_col < (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE; block_col++) {
        for (int d = tid; d < head_dim; d += BLOCK_SIZE) {
            k_tile[tid * HEAD_DIM + d] = K[block_col * BLOCK_SIZE * head_dim + tid * head_dim + d];
            v_tile[tid * HEAD_DIM + d] = V[block_col * BLOCK_SIZE * head_dim + tid * head_dim + d];
        }
        __syncthreads();

        for (int j = 0; j < BLOCK_SIZE; j++) {
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++)
                score += q_tile[tid * HEAD_DIM + d] * k_tile[j * HEAD_DIM + d];
            s_tile[tid * BLOCK_SIZE + j] = score * scale;
        }

        // BUG: 0xffffffff mask assumes 32-lane warp - wrong on AMD wavefront-64
        float thread_max = s_tile[tid * BLOCK_SIZE];
        for (int j = 1; j < BLOCK_SIZE; j++)
            thread_max = fmaxf(thread_max, s_tile[tid * BLOCK_SIZE + j]);
        for (int offset = 16; offset > 0; offset >>= 1)
            thread_max = fmaxf(thread_max, __shfl_down(thread_max, offset));
        float block_max = __shfl(thread_max, 0);

        float exp_sum = 0.0f;
        for (int j = 0; j < BLOCK_SIZE; j++) {
            s_tile[tid * BLOCK_SIZE + j] = expf(s_tile[tid * BLOCK_SIZE + j] - block_max);
            exp_sum += s_tile[tid * BLOCK_SIZE + j];
        }
        // BUG: offset=16 is half of warp-32, should be 32 for AMD wavefront-64
        for (int offset = 16; offset > 0; offset >>= 1)
            exp_sum += __shfl_down(exp_sum, offset);

        float new_max = fmaxf(row_max, block_max);
        float correction = expf(row_max - new_max);
        row_sum = correction * row_sum + exp_sum;
        row_max = new_max;

        for (int d = 0; d < head_dim; d++) {
            float pv = 0.0f;
            for (int j = 0; j < BLOCK_SIZE; j++)
                pv += s_tile[tid * BLOCK_SIZE + j] * v_tile[j * HEAD_DIM + d];
            acc[d] = correction * acc[d] + pv;
        }
        __syncthreads();
    }

    for (int d = 0; d < head_dim; d++)
        O[block_row * BLOCK_SIZE * head_dim + tid * head_dim + d] = acc[d] / row_sum;
    L[block_row * BLOCK_SIZE + tid] = row_max + logf(row_sum);
}

int main() {
    int seq_len = 128, head_dim = HEAD_DIM;
    float scale = 1.0f / sqrtf((float)head_dim);
    printf("Flash Attention Forward (seq=%d head_dim=%d)\n", seq_len, head_dim);
    printf("AMD-specific bugs: warp-32 shuffle mask, offset=16 for wavefront-64\n");
    size_t sz = seq_len * head_dim * sizeof(float);
    float *d_Q, *d_K, *d_V, *d_O, *d_L;
    hipMalloc(&d_Q, sz); hipMalloc(&d_K, sz); hipMalloc(&d_V, sz);
    hipMalloc(&d_O, sz); hipMalloc(&d_L, seq_len * sizeof(float));
    dim3 grid(seq_len / BLOCK_SIZE), block(BLOCK_SIZE);
    size_t shmem = (3 * BLOCK_SIZE * HEAD_DIM + BLOCK_SIZE * BLOCK_SIZE) * sizeof(float);
    flash_attention_forward<<<grid, block, shmem>>>(d_Q, d_K, d_V, d_O, d_L, seq_len, head_dim, scale);
    hipDeviceSynchronize();
    printf("Done - kernel executed on gfx942\n");
    hipFree(d_Q); hipFree(d_K); hipFree(d_V); hipFree(d_O); hipFree(d_L);
    return 0;
}
