#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "utils.cu"

#define N 4096

int main()
{
    cublasStatus_t stat;   // cuBLAS functions status
    cublasHandle_t handle; // cuBLAS context

    srand(time(NULL));

    CudaDeviceInfo();

    float *a = (float *)malloc(N * N * sizeof(float));
    float *b = (float *)malloc(N * N * sizeof(float));
    float *c = (float *)malloc(N * N * sizeof(float));

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * N * sizeof(float));
    cudaMalloc(&d_b, N * N * sizeof(float));
    cudaMalloc(&d_c, N * N * sizeof(float));

    stat = cublasCreate(&handle); // initialize CUBLAS context

    // fill a & b and zero out c
    matrix_random(a, N*N);
    matrix_random(b, N*N);
    matrix_zeros(c, N*N);

    cudaMemcpy(d_a, a, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * N * sizeof(float), cudaMemcpyHostToDevice);
	
    float alpha = 1.0;
    float beta = 1.0;

    uint64_t start = nanos();
    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_b, N,
                     d_a, N, &beta, d_c, N);

    cudaDeviceSynchronize();
    uint64_t end = nanos();

    cudaMemcpy(c, d_c, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    double gflop = (2.0 * N * N * N) * 1e-9;
    double s = (end - start) * 1e-9;
    printf("%f GFLOP/S -- %.2f ms\n", gflop / s, s * 1e3);

    // {
	// 	// compute naive reference matmul on cpu
    // 	printf("Computing reference matmul result on cpu\n");
	// 	float *reference_c = (float *)malloc(N * N * sizeof(float));
    // 	matmul_c(a, b, reference_c, N);

	// 	// check each item
	// 	matrix_eq(reference_c, c, N);
	// 	free(reference_c);
	// 	printf("ALL GOOD\n");
    // }
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cublasDestroy(handle); // destroy CUBLAS context
    free(a);
    free(b);
    free(c);
}
