// from https://github.com/tinygrad/tinygrad/blob/master/extra/gemm/cuda_matmul.py
#include <mma.h>
#include <stdio.h>
#include "utils.cu"

using namespace nvcuda;

#define N 1024
#define TILE_SIZE 16
#define WARP_SIZE 32

__global__ void matmul(half *a, half *b, float *c)
{
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    warpM *= 4;
    warpN *= 4;

    wmma::fragment<wmma::matrix_a, TILE_SIZE, TILE_SIZE, TILE_SIZE, half, wmma::col_major> a_frag[4];
    wmma::fragment<wmma::matrix_b, TILE_SIZE, TILE_SIZE, TILE_SIZE, half, wmma::col_major> b_frag[4];
    wmma::fragment<wmma::accumulator, TILE_SIZE, TILE_SIZE, TILE_SIZE, float> acc_frag[4][4];
    for (int j = 0; j < 4; j++)
    {
        for (int i = 0; i < 4; i++)
        {
            wmma::fill_fragment(acc_frag[i][j], 0.0f);
        }
    }

    for (int k = 0; k < N; k += TILE_SIZE)
    {
        int aRow = warpM * TILE_SIZE;
        int aCol = k;
        int bRow = k;
        int bCol = warpN * TILE_SIZE;

        wmma::load_matrix_sync(a_frag[0], a + aRow + 0 * TILE_SIZE + aCol * N, N);
        wmma::load_matrix_sync(a_frag[1], a + aRow + 1 * TILE_SIZE + aCol * N, N);
        wmma::load_matrix_sync(a_frag[2], a + aRow + 2 * TILE_SIZE + aCol * N, N);
        wmma::load_matrix_sync(a_frag[3], a + aRow + 3 * TILE_SIZE + aCol * N, N);

        wmma::load_matrix_sync(b_frag[0], b + bRow + (0 * TILE_SIZE + bCol) * N, N);
        wmma::load_matrix_sync(b_frag[1], b + bRow + (1 * TILE_SIZE + bCol) * N, N);
        wmma::load_matrix_sync(b_frag[2], b + bRow + (2 * TILE_SIZE + bCol) * N, N);
        wmma::load_matrix_sync(b_frag[3], b + bRow + (3 * TILE_SIZE + bCol) * N, N);

#pragma unroll
        for (int j = 0; j < 4; j++)
        {
#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                wmma::mma_sync(acc_frag[i][j], a_frag[i], b_frag[j], acc_frag[i][j]);
            }
        }
    }

    for (int j = 0; j < 4; j++)
    {
        for (int i = 0; i < 4; i++)
        {
            // wmma::fragment<wmma::accumulator, TILE_SIZE, TILE_SIZE, TILE_SIZE, float> acc_store;
            // for (int t = 0; t < acc_frag[i][j].num_elements; t++)
            //     acc_store.x[t] = acc_frag[i][j].x[t];
            int cRow = (warpM + i) * TILE_SIZE;
            int cCol = (warpN + j) * TILE_SIZE;
            wmma::store_matrix_sync(c + cRow + cCol * N, acc_frag[i][j], N, wmma::mem_col_major);
        }
    }
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

    dim3 grid_dim(CEIL_DIV(N, TILE_SIZE * 4), CEIL_DIV(N, TILE_SIZE * 4));
    dim3 block_dim(WARP_SIZE);
    printf("LAUNCHING with grid_dim: (%d, %d) and block_dim: %d\n", grid_dim.x, grid_dim.y, block_dim.x);

    uint64_t start = nanos();
    matmul<<<grid_dim, block_dim>>>(d_a, d_b, d_c);
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
