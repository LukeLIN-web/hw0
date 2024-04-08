#include <stdio.h>

#define TILE_WIDTH 16

__global__ void matrixMul(float *A, float *B, float *C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k) {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            sum += A[row * n + i] * B[i * k + col];
        }
        C[row * k + col] = sum;
    }
}

void matrixMulCPU(float *A, float *B, float *C, int m, int n, int k) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            float sum = 0.0f;
            for (int p = 0; p < n; ++p) {
                sum += A[i * n + p] * B[p * k + j];
            }
            C[i * k + j] = sum;
        }
    }
}

int main() {
    int m = 512;
    int n = 512;
    int k = 512;

    float *h_A, *h_B, *h_C_CPU, *h_C_CUDA;

    size_t size_A = m * n * sizeof(float);
    size_t size_B = n * k * sizeof(float);
    size_t size_C = m * k * sizeof(float);

    // Allocate host memory
    h_A = (float *)malloc(size_A);
    h_B = (float *)malloc(size_B);
    h_C_CPU = (float *)malloc(size_C);
    h_C_CUDA = (float *)malloc(size_C);

    // Initialize matrices A and B
    for (int i = 0; i < m * n; ++i) {
        h_A[i] = 1.0f;
    }
    for (int i = 0; i < n * k; ++i) {
        h_B[i] = 2.0f;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size_A);
    cudaMalloc((void **)&d_B, size_B);
    cudaMalloc((void **)&d_C, size_C);

    // Transfer data from host to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((k + TILE_WIDTH - 1) / TILE_WIDTH, (m + TILE_WIDTH - 1) / TILE_WIDTH);

    // Launch kernel
    matrixMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, m, n, k);

    // Transfer results from device to host
    cudaMemcpy(h_C_CUDA, d_C, size_C, cudaMemcpyDeviceToHost);

    // Perform matrix multiplication on CPU
    matrixMulCPU(h_A, h_B, h_C_CPU, m, n, k);

    // Compare results from CPU and CUDA
    bool isEqual = true;
    for (int i = 0; i < m * k; ++i) {
        if (h_C_CPU[i] != h_C_CUDA[i]) {
            isEqual = false;
            break;
        }
    }

    if (isEqual) {
        printf("Results match between CPU and CUDA.\n");
    } else {
        printf("Results do not match between CPU and CUDA.\n");
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C_CPU);
    free(h_C_CUDA);

    return 0;
}
