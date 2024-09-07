#include <stdio.h>
#include "utils.cu"
#include <cuda_fp16.h>

#define N 1024
#define WARP_SIZE 32
#define BLOCK_SIZE 32

__global__ void matmul(__half *a, __half *b, __half *c, int n)
{
    int block_i = blockIdx.y;                // block index along row (y) axis
    int block_j = blockIdx.x;                // block index along col (x) axis
    int thread_i = threadIdx.x / BLOCK_SIZE; // thread item y index inside the 32x32 block
    int thread_j = threadIdx.x % BLOCK_SIZE; // thread item x index inside the 32x32 block

    int row = block_i * BLOCK_SIZE + thread_i;
    int col = block_j * BLOCK_SIZE + thread_j;

    if (row >= n or col >= n)
    {
        return;
    }

    __shared__ __half tile_a[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ __half tile_b[BLOCK_SIZE * BLOCK_SIZE];

    __half acc = 0.0f;
    for (int block_start_i = 0; block_start_i < n; block_start_i += BLOCK_SIZE)
    {
        tile_a[thread_i * BLOCK_SIZE + thread_j] = a[row * n + (block_start_i + thread_j)];
        tile_b[thread_i * BLOCK_SIZE + thread_j] = b[(block_start_i + thread_i) * n + col];

        __syncthreads(); // wait for all the threads in the warp to load their item of the block into the block (smem)

        for (int k = 0; k < BLOCK_SIZE; k++)
        {
            acc = __hfma(tile_a[thread_i * BLOCK_SIZE + k], tile_b[k * BLOCK_SIZE + thread_j], acc);
        }

        __syncthreads(); // we don't want to change the tiles in smem while some threads are still accumulating
    }

    c[row * n + col] = acc;
}

int main()
{
    srand(time(NULL));

    float *a = (float *)malloc(N * N * sizeof(float));
    float *b = (float *)malloc(N * N * sizeof(float));
    float *c = (float *)malloc(N * N * sizeof(float));

    __half *a_h = (__half *)malloc(N * N * sizeof(__half));
    __half *b_h = (__half *)malloc(N * N * sizeof(__half));
    __half *c_h = (__half *)malloc(N * N * sizeof(__half));

    __half *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * N * sizeof(__half));
    cudaMalloc(&d_b, N * N * sizeof(__half));
    cudaMalloc(&d_c, N * N * sizeof(__half));

    // fill a & b and zero out c
    matrix_random(a, N);
    matrix_random(b, N);
    matrix_zeros(c, N);

    for (int i = 0; i < N * N; i++)
    {
        a_h[i] = __half2float(a[i]);
        b_h[i] = __half2float(b[i]);
    }

    cudaMemcpy(d_a, a_h, N * N * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_h, N * N * sizeof(__half), cudaMemcpyHostToDevice);

    dim3 grid_dim(CEIL_DIV(N, WARP_SIZE), CEIL_DIV(N, WARP_SIZE));
    dim3 block_dim(WARP_SIZE * WARP_SIZE);
    printf("LAUNCHING with grid_dim: (%d, %d) and block_dim: (%d, %d)\n", grid_dim.x, grid_dim.y, block_dim.x, block_dim.y);

    uint64_t start = nanos();
    matmul<<<grid_dim, block_dim>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    uint64_t end = nanos();

    cudaMemcpy(c_h, d_c, N * N * sizeof(__half), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N * N; i++)
    {
        c[i] = __half2float(c_h[i]);
    }

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
    free(c_h);
    free(a);
    free(b);
    free(c);
}
