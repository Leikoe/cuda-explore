#include <stdio.h>
#include <time.h>
#include <stdint.h>

uint64_t nanos()
{
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    return (uint64_t)start.tv_sec * 1000000000 + (uint64_t)start.tv_nsec;
}

#define N 1024
#define CEIL_DIV(a,b) ((a+b-1)/b)
#define WARP_SIZE 32
#define BLOCK_SIZE 32

__global__ void matmul(float *a, float *b, float *c, int n)
{
	int block_i = blockIdx.y;  // block index along row (y) axis
	int block_j = blockIdx.x;  // block index along col (x) axis
    int thread_i = threadIdx.x / BLOCK_SIZE;  // thread item y index inside the 32x32 block
    int thread_j = threadIdx.x % BLOCK_SIZE;  // thread item x index inside the 32x32 block

	int row = block_i * BLOCK_SIZE + thread_i;
	int col = block_j * BLOCK_SIZE + thread_j;

    if (row >= n or col >= n)
    {
        return;
    }

	__shared__ float tile_a[BLOCK_SIZE * BLOCK_SIZE];
	__shared__ float tile_b[BLOCK_SIZE * BLOCK_SIZE];

    float acc = 0.0f;

    for (int block_start_i = 0; block_start_i < n; block_start_i += BLOCK_SIZE)
    {
		tile_a[thread_i * BLOCK_SIZE + thread_j] = a[(block_start_i + thread_i) * n + col];
		tile_b[thread_i * BLOCK_SIZE + thread_j] = b[(block_start_i + thread_i) * n + col];
		
		__syncthreads();  // wait for all the threads in the warp to load their item of the block into the block (smem)

		for (int k = 0; k < BLOCK_SIZE; k++) {
			acc += tile_a[thread_i * BLOCK_SIZE + k] * tile_b[k * BLOCK_SIZE + thread_j];
		}

		__syncthreads();  // we don't want to change the tiles in smem while some threads are still accumulating
		
    }

    c[row * n + col] = acc;
}

int main()
{
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
        a[i] = 1.0f;
        b[i] = 2.0f;
        c[i] = 0.0f;
    }

    cudaMemcpy(d_a, a, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * N * sizeof(float), cudaMemcpyHostToDevice);

	dim3 grid_dim(CEIL_DIV(N, WARP_SIZE), CEIL_DIV(N, WARP_SIZE));
	dim3 block_dim(WARP_SIZE * WARP_SIZE);
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
				if (c[i * N + j] != reference_c[i * N + j])
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
