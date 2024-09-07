#include <mma.h>
#include <stdio.h>
#include "utils.cu"

#define N 4096
#define WARP_SIZE 32
#define TILE_SIZE 16

__global__ void matmul(__half *a, __half *b, float *c, int n)
{
    int block_i = blockIdx.y; // block index along row (y) axis
    int block_j = blockIdx.x; // block index along col (x) axis
    // int thread_i = threadIdx.x / BLOCK_SIZE; // thread item y index inside the 32x32 block
    // int thread_j = threadIdx.x % BLOCK_SIZE; // thread item x index inside the 32x32 block

    // int row = block_i * BLOCK_SIZE + thread_i;
    // int col = block_j * BLOCK_SIZE + thread_j;

    // if (row >= n or col >= n)
    // {
    //     return;
    // }

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, TILE_SIZE, TILE_SIZE, TILE_SIZE, __half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, TILE_SIZE, TILE_SIZE, TILE_SIZE, __half, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, TILE_SIZE, TILE_SIZE, TILE_SIZE, float> acc_frag;

    nvcuda::wmma::fill_fragment(acc_frag, 0.0);
#pragma unroll
    for (int wmma_block_index = 0; wmma_block_index < N / TILE_SIZE; wmma_block_index++)
    {
        nvcuda::wmma::load_matrix_sync(a_frag, a + (block_i * TILE_SIZE * n) + (wmma_block_index * TILE_SIZE), n);
        nvcuda::wmma::load_matrix_sync(b_frag, b + (wmma_block_index * TILE_SIZE * n) + (block_j * TILE_SIZE), n);

        nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    nvcuda::wmma::store_matrix_sync(c + (block_i * TILE_SIZE * n) + (block_j * TILE_SIZE), acc_frag, n, nvcuda::wmma::mem_row_major);
}

int main()
{
    srand(time(NULL));

    float *a = (float *)malloc(N * N * sizeof(float));
    float *b = (float *)malloc(N * N * sizeof(float));
    float *c = (float *)malloc(N * N * sizeof(float));

    // fill a & b
    matrix_random(a, N * N);
    matrix_random(b, N * N);

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

    dim3 grid_dim(CEIL_DIV(N, TILE_SIZE), CEIL_DIV(N, TILE_SIZE));
    dim3 block_dim(WARP_SIZE);
    printf("LAUNCHING with grid_dim: (%d, %d) and block_dim: %d\n", grid_dim.x, grid_dim.y, block_dim.x);

    uint64_t start = nanos();
    matmul<<<grid_dim, block_dim>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    uint64_t end = nanos();

    cudaMemcpy(c, d_c, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    double gflop = (2.0 * N * N * N) * 1e-9;
    double s = (end - start) * 1e-9;
    printf("%f GFLOP/S -- %.2f ms\n", gflop / s, s * 1e3);

    // {
    //     // compute naive reference matmul on cpu
    //     printf("Computing reference matmul result on cpu\n");
    //     float *reference_c = (float *)malloc(N * N * sizeof(float));
    //     matmul_c(a, b, reference_c, N);

    //     // check each item
    //     printf("Comparing reference result with gpu result\n");
    //     matrix_eq(reference_c, c, N);
    //     printf("ALL GOOD\n");
    //     free(reference_c);
    // }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a_h);
    free(b_h);
    free(a);
    free(b);
    free(c);
}
