/**
 * main-cuda.cpp: This file is part of the mixbench GPU micro-benchmark suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string.h>
#include "lcutil.h"
#include "mix_kernels_cuda.h"
#include "version_info.h"

#define VECTOR_SIZE (32 * 1024 * 1024)

typedef struct {
    int device_index;
    bool use_zeros;
    bool run_gemm;
    int matrix_size;
} ArgParams;

void print_usage(const char* program_name) {
    printf("Usage: %s [options] [device_index]\n\n", program_name);
    printf("Options:\n");
    printf("  -h, --help          Show this help message\n");
    printf("  -z, --zeros         Use zero-initialized data (control energy mode)\n");
    printf("  --gemm              Run GEMM benchmark instead of compute sweep\n");
    printf("  --matrix-size N     Matrix size for GEMM (default: 4096)\n");
    printf("\n");
    printf("Arguments:\n");
    printf("  device_index        CUDA device index (default: 0)\n");
}

bool parse_arguments(int argc, char* argv[], ArgParams* params) {
    params->device_index = 0;
    params->use_zeros = false;
    params->run_gemm = false;
    params->matrix_size = 4096;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            return false;
        } else if (strcmp(argv[i], "-z") == 0 || strcmp(argv[i], "--zeros") == 0) {
            params->use_zeros = true;
        } else if (strcmp(argv[i], "--gemm") == 0) {
            params->run_gemm = true;
        } else if (strcmp(argv[i], "--matrix-size") == 0) {
            if (i + 1 < argc) {
                params->matrix_size = atoi(argv[++i]);
                if (params->matrix_size <= 0) {
                    fprintf(stderr, "Error: Invalid matrix size\n");
                    return false;
                }
            } else {
                fprintf(stderr, "Error: --matrix-size requires an argument\n");
                return false;
            }
        } else if (argv[i][0] != '-') {
            params->device_index = atoi(argv[i]);
        } else {
            fprintf(stderr, "Error: Unknown option: %s\n", argv[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char* argv[]) {
    printf("mixbench (%s)\n", VERSION_INFO);

    ArgParams params;
    if (!parse_arguments(argc, argv, &params)) {
        print_usage(argv[0]);
        return 1;
    }

    printf("Use \"-h\" argument to see available options\n");

    unsigned int datasize = VECTOR_SIZE * sizeof(double);

    cudaSetDevice(params.device_index);
    StoreDeviceInfo(stdout);

    size_t freeCUDAMem, totalCUDAMem;
    cudaMemGetInfo(&freeCUDAMem, &totalCUDAMem);
    printf("Total GPU memory %lu, free %lu\n", totalCUDAMem, freeCUDAMem);
    printf("Buffer size:          %dMB\n", datasize / (1024 * 1024));

    // Print mode information
    printf("# Mode: %s\n", params.use_zeros ? "zeros" : "random");
    printf("# Workload: %s\n", params.run_gemm ? "gemm" : "compute");

    double* c;
    c = (double*)malloc(datasize);

    if (params.run_gemm) {
        runGemmBenchmark(params.matrix_size, params.use_zeros);
    } else {
        mixbenchGPU(c, VECTOR_SIZE, params.use_zeros);
    }

    free(c);

    return 0;
}
