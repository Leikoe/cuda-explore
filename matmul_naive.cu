#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <assert.h>
#include <cuda_fp16.h>
#include "utils.cu"

#define N 1024

__global__ void matmul(__half *a, __half *b, float *c, int n)
{
    int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_id_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (thread_id_x >= N || thread_id_y >= N)
    {
        return;
    }

    float acc = 0.0f;
    for (int k = 0; k < n; k++)
    {
        acc = __half2float(__hmul(a[thread_id_y * n + k], b[k * n + thread_id_x])) + acc;
    }

    c[thread_id_y * n + thread_id_x] = acc;
}

int main()
{
    srand(time(NULL));

    float *a = (float *)malloc(N * N * sizeof(float));
    float *b = (float *)malloc(N * N * sizeof(float));
    float *c = (float *)malloc(N * N * sizeof(float));

    // fill a & b
    matrix_random_fp16valued(a, N * N);
    matrix_random_fp16valued(b, N * N);

    __half *a_h = (__half *)malloc(N * N * sizeof(__half));
    __half *b_h = (__half *)malloc(N * N * sizeof(__half));

    for (int i = 0; i < N * N; i++)
    {
        a_h[i] = __float2half(a[i]);
        b_h[i] = __float2half(b[i]);
    }

    __half *d_a, *d_b;
    float *d_c;
    cudaMalloc(&d_a, N * N * sizeof(__half));
    cudaMalloc(&d_b, N * N * sizeof(__half));
    cudaMalloc(&d_c, N * N * sizeof(float));
    cudaMemcpy(d_a, a_h, N * N * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_h, N * N * sizeof(__half), cudaMemcpyHostToDevice);

    dim3 block_dim(16, 16);
    dim3 grid_dim(CEIL_DIV(N, block_dim.x), CEIL_DIV(N, block_dim.y));
    printf("LAUNCHING with grid_dim: (%d, %d) and block_dim: (%d, %d)\n", grid_dim.x, grid_dim.y, block_dim.x, block_dim.y);

    uint64_t start = nanos();
    matmul<<<grid_dim, block_dim>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    uint64_t end = nanos();

    cudaMemcpy(c, d_c, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    double gflop = (2.0 * N * N * N) * 1e-9;
    double s = (end - start) * 1e-9;
    printf("%f GFLOP/S -- %.2f ms\n", gflop / s, s * 1e3);

    {
        // compute naive reference matmul on cpu
        printf("Computing reference matmul result on cpu\n");
        float *reference_c = (float *)malloc(N * N * sizeof(float));
        matmul_c(a, b, reference_c, N);

        // check each item
        printf("Comparing reference result with gpu result\n");
        matrix_eq(reference_c, c, N);
        printf("ALL GOOD\n");
        free(reference_c);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a_h);
    free(b_h);
    free(a);
    free(b);
    free(c);
}
