#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <assert.h>
#include <bits/stdc++.h>

using namespace std;


uint64_t nanos()
{
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    return (uint64_t)start.tv_sec * 1000000000 + (uint64_t)start.tv_nsec;
}

#define N 1024
#define CEIL_DIV(a, b) ((a + b - 1) / b)

__global__ void matmul(float *a, float *b, float *c, int n)
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
        assert((thread_id_y * n + k) < (n*n));  // check out of bounds
        assert((k * n + thread_id_x) < (n*n));  // check out of bounds
        acc += a[thread_id_y * n + k] * b[k * n + thread_id_x];
    }

    assert((thread_id_y * n + thread_id_x) < (n*n));  // check out of bounds
    c[thread_id_y * n + thread_id_x] = acc;
}

int main()
{
    srand(time(NULL));

    float *a = (float *)malloc(N * N * sizeof(float));
    float *b = (float *)malloc(N * N * sizeof(float));
    float *c = (float *)malloc(N * N * sizeof(float));

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * N * sizeof(float));
    cudaMalloc(&d_b, N * N * sizeof(float));
    cudaMalloc(&d_c, N * N * sizeof(float));

    // fill a & b and zero out c
    for (int i = 0; i < (N * N); i++)
    {
        a[i] = ((double)rand()) / INT_MAX;
        b[i] = ((double)rand()) / INT_MAX;
        c[i] = 0.0f;
    }

    cudaMemcpy(d_a, a, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * N * sizeof(float), cudaMemcpyHostToDevice);

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
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                float acc = 0.0f;
                for (int k = 0; k < N; k++)
                {
                    acc += a[i * N + k] * b[k * N + j];
                }
                reference_c[i * N + j] = acc;
            }
        }

        // check each item
        printf("Comparing reference result with gpu result\n");
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                if (reference_c[i * N + j] - c[i * N + j] > 1e-3)
                {
                    printf("ERROR at i=%d j=%d (should be %f, is %f)\n", i, j, reference_c[i * N + j], c[i * N + j]);
                    exit(1);
                }
            }
        }
        free(reference_c);
        printf("ALL GOOD\n");
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c);
}
