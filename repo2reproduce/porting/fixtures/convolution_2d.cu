#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// 2D Convolution kernel with intentional warp size bug
__global__ void convolution2D(const float *input, const float *kernel, float *output, 
                            int input_height, int input_width, int kernel_size, int output_height, int output_width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < output_height && col < output_width) {
        float sum = 0.0f;
        int kernel_radius = kernel_size / 2;
        
        // Apply convolution
        for (int i = -kernel_radius; i <= kernel_radius; i++) {
            for (int j = -kernel_radius; j <= kernel_radius; j++) {
                int input_row = row + i;
                int input_col = col + j;
                
                // Check bounds
                if (input_row >= 0 && input_row < input_height && 
                    input_col >= 0 && input_col < input_width) {
                    
                    int kernel_row = i + kernel_radius;
                    int kernel_col = j + kernel_radius;
                    
                    sum += input[input_row * input_width + input_col] * 
                           kernel[kernel_row * kernel_size + kernel_col];
                }
            }
        }
        
        output[row * output_width + col] = sum;
        
        // Intentional warp size bug - assumes 32 threads per warp
        // This will break on AMD wavefront (64 threads)
        if (threadIdx.x % 32 == 0 && threadIdx.y % 32 == 0) {
            // This warp-level operation only works for CUDA
            printf("Warp (%d,%d) processed output pixel (%d,%d) = %f\n", 
                   threadIdx.x / 32, threadIdx.y / 32, row, col, sum);
        }
    }
}

// Shared memory version for comparison
__global__ void convolution2DShared(const float *input, const float *kernel, float *output,
                                   int input_height, int input_width, int kernel_size, int output_height, int output_width) {
    __shared__ float shared_input[32 + 6][32 + 6]; // +6 for 3x3 kernel padding
    __shared__ float shared_kernel[7][7]; // Max 7x7 kernel
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    int kernel_radius = kernel_size / 2;
    
    // Load kernel into shared memory
    if (threadIdx.x < kernel_size && threadIdx.y < kernel_size) {
        shared_kernel[threadIdx.y][threadIdx.x] = kernel[threadIdx.y * kernel_size + threadIdx.x];
    }
    
    // Load input tile with padding
    int input_row = blockIdx.y * blockDim.y + threadIdx.y - kernel_radius;
    int input_col = blockIdx.x * blockDim.x + threadIdx.x - kernel_radius;
    
    if (input_row >= 0 && input_row < input_height && input_col >= 0 && input_col < input_width) {
        shared_input[threadIdx.y][threadIdx.x] = input[input_row * input_width + input_col];
    } else {
        shared_input[threadIdx.y][threadIdx.x] = 0.0f;
    }
    
    __syncthreads();
    
    // Compute convolution
    if (row < output_height && col < output_width) {
        float sum = 0.0f;
        
        for (int i = 0; i < kernel_size; i++) {
            for (int j = 0; j < kernel_size; j++) {
                sum += shared_input[threadIdx.y + i][threadIdx.x + j] * shared_kernel[i][j];
            }
        }
        
        output[row * output_width + col] = sum;
    }
}

int main(int argc, char **argv) {
    int input_height = 1024;
    int input_width = 1024;
    int kernel_size = 3;
    
    int output_height = input_height - kernel_size + 1;
    int output_width = input_width - kernel_size + 1;
    
    size_t input_size = input_height * input_width * sizeof(float);
    size_t kernel_size_bytes = kernel_size * kernel_size * sizeof(float);
    size_t output_size = output_height * output_width * sizeof(float);
    
    printf("Input: %dx%d, Kernel: %dx%d, Output: %dx%d\n", 
           input_height, input_width, kernel_size, kernel_size, output_height, output_width);
    
    // Allocate host memory
    float *h_input = (float *)malloc(input_size);
    float *h_kernel = (float *)malloc(kernel_size_bytes);
    float *h_output = (float *)malloc(output_size);
    float *h_output_ref = (float *)malloc(output_size);
    
    // Initialize input and kernel
    for (int i = 0; i < input_height * input_width; i++) {
        h_input[i] = rand() / (float)RAND_MAX;
    }
    
    // Simple 3x3 edge detection kernel
    float kernel_3x3[9] = {-1, -1, -1, -1, 8, -1, -1, -1, -1};
    for (int i = 0; i < kernel_size * kernel_size; i++) {
        h_kernel[i] = kernel_3x3[i];
    }
    
    // Allocate device memory
    float *d_input, *d_kernel, *d_output, *d_output_ref;
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_kernel, kernel_size_bytes);
    cudaMalloc(&d_output, output_size);
    cudaMalloc(&d_output_ref, output_size);
    
    // Copy to device
    cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernel_size_bytes, cudaMemcpyHostToDevice);
    
    // Setup kernel launch parameters
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((output_width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (output_height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    printf("Launching kernel with grid (%d,%d) and block (%d,%d)\n",
           blocksPerGrid.x, blocksPerGrid.y, threadsPerBlock.x, threadsPerBlock.y);
    
    // Warmup
    convolution2D<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_kernel, d_output_ref, 
                                                     input_height, input_width, kernel_size, 
                                                     output_height, output_width);
    cudaDeviceSynchronize();
    
    // Time basic kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    convolution2D<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_kernel, d_output_ref,
                                                     input_height, input_width, kernel_size,
                                                     output_height, output_width);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float basic_time = 0;
    cudaEventElapsedTime(&basic_time, start, stop);
    printf("Basic kernel time: %.3f ms\n", basic_time);
    
    // Time shared memory kernel
    cudaEventRecord(start);
    convolution2DShared<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_kernel, d_output,
                                                          input_height, input_width, kernel_size,
                                                          output_height, output_width);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float shared_time = 0;
    cudaEventElapsedTime(&shared_time, start, stop);
    printf("Shared memory kernel time: %.3f ms\n", shared_time);
    
    printf("Speedup: %.2fx\n", basic_time / shared_time);
    
    // Copy results back
    cudaMemcpy(h_output_ref, d_output_ref, output_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost);
    
    // Verify results (first few elements)
    bool correct = true;
    for (int i = 0; i < min(100, output_height * output_width); i++) {
        if (fabs(h_output[i] - h_output_ref[i]) > 1e-5) {
            printf("Mismatch at element %d: %f != %f\n", i, h_output[i], h_output_ref[i]);
            correct = false;
            break;
        }
    }
    
    if (correct) {
        printf("Verification PASSED (first 100 elements)\n");
    } else {
        printf("Verification FAILED\n");
    }
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    cudaFree(d_output_ref);
    free(h_input);
    free(h_kernel);
    free(h_output);
    free(h_output_ref);
    
    printf("Done\n");
    return 0;
}
