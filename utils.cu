#include <time.h>
#include <stdint.h>
#include <bits/stdc++.h>
#include <cuda_fp16.h>

using namespace std;

#define CEIL_DIV(a, b) (((a) + (b - 1)) / (b))

void matrix_random(float *a, int numel)
{
    for (int i = 0; i < numel; i++)
    {
        a[i] = ((double)rand()) / INT_MAX;
    }
}

void matrix_random_fp16valued(float *a, int numel)
{
    for (int i = 0; i < numel; i++)
    {
        a[i] = __half2float(__float2half(((double)rand()) / INT_MAX));
    }
}

void matrix_zeros(float *a, int numel)
{
    for (int i = 0; i < numel; i++)
    {
        a[i] = 0.0;
    }
}

void matmul_c(float *a, float *b, float *c, int N)
{
    for (int i = 0; i < N; i++)
    {

        for (int j = 0; j < N; j++)
        {
            float acc = 0.0f;
            for (int k = 0; k < N; k++)
            {
                acc += a[i * N + k] * b[k * N + j];
            }
            c[i * N + j] = acc;
        }
    }
}

void matrix_eq(float *a, float *b, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (fabs(b[i * N + j] - a[i * N + j]) > 2e-2)
            {
                printf("ERROR at i=%d j=%d (should be %f, is %f)\n", i, j, a[i * N + j], b[i * N + j]);
                exit(1);
            }
        }
    }
}

// from https://github.com/siboehm/SGEMM_CUDA/blob/master/src/runner.cu#L24
void CudaDeviceInfo()
{
    int deviceId;

    cudaGetDevice(&deviceId);

    cudaDeviceProp props{};
    cudaGetDeviceProperties(&props, deviceId);

    printf("Device ID: %d\n\
    Name: %s\n\
    Compute Capability: %d.%d\n\
    memoryBusWidth: %d\n\
    maxThreadsPerBlock: %d\n\
    maxThreadsPerMultiProcessor: %d\n\
    maxRegsPerBlock: %d\n\
    maxRegsPerMultiProcessor: %d\n\
    totalGlobalMem: %zuMB\n\
    sharedMemPerBlock: %zuKB\n\
    sharedMemPerMultiprocessor: %zuKB\n\
    totalConstMem: %zuKB\n\
    multiProcessorCount: %d\n\
    Warp Size: %d\n",
           deviceId, props.name, props.major, props.minor, props.memoryBusWidth,
           props.maxThreadsPerBlock, props.maxThreadsPerMultiProcessor,
           props.regsPerBlock, props.regsPerMultiprocessor,
           props.totalGlobalMem / 1024 / 1024, props.sharedMemPerBlock / 1024,
           props.sharedMemPerMultiprocessor / 1024, props.totalConstMem / 1024,
           props.multiProcessorCount, props.warpSize);
};

uint64_t nanos()
{
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    return (uint64_t)start.tv_sec * 1000000000 + (uint64_t)start.tv_nsec;
}