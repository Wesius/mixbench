/**
 * main.cpp: This file is part of the mixbench GPU micro-benchmark suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#include <omp.h>

#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>

#include "mix_kernels_cpu.h"
#include "version_info.h"

constexpr auto DEF_VECTOR_SIZE_PER_THREAD = 4 * 1024 * 1024;

struct ArgParams {
    unsigned int vecwidth;
    bool use_zeros;
    bool run_gemm;
    int matrix_size;
};

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options] [array size(1024^2)]" << std::endl
              << std::endl
              << "Options:" << std::endl
              << "  -h, --help          Show this help message" << std::endl
              << "  -z, --zeros         Use zero-initialized data (control energy mode)" << std::endl
              << "  --gemm              Run GEMM benchmark instead of compute sweep" << std::endl
              << "  --matrix-size N     Matrix size for GEMM (default: 4096)" << std::endl;
}

bool argument_parsing(int argc, char* argv[], ArgParams* output) {
    const auto hardware_concurrency = omp_get_max_threads();
    output->vecwidth = static_cast<unsigned int>(
        hardware_concurrency * DEF_VECTOR_SIZE_PER_THREAD / (1024 * 1024));
    output->use_zeros = false;
    output->run_gemm = false;
    output->matrix_size = 4096;

    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "-h") == 0) || (strcmp(argv[i], "--help") == 0)) {
            return false;
        } else if ((strcmp(argv[i], "-z") == 0) || (strcmp(argv[i], "--zeros") == 0)) {
            output->use_zeros = true;
        } else if (strcmp(argv[i], "--gemm") == 0) {
            output->run_gemm = true;
        } else if (strcmp(argv[i], "--matrix-size") == 0) {
            if (i + 1 < argc) {
                output->matrix_size = atoi(argv[++i]);
                if (output->matrix_size <= 0) {
                    std::cerr << "Error: Invalid matrix size" << std::endl;
                    return false;
                }
            } else {
                std::cerr << "Error: --matrix-size requires an argument" << std::endl;
                return false;
            }
        } else if (argv[i][0] != '-') {
            unsigned long value = strtoul(argv[i], NULL, 10);
            output->vecwidth = value;
        } else {
            std::cerr << "Error: Unknown option: " << argv[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main(int argc, char* argv[]) {
    std::cout << "mixbench-cpu (" << VERSION_INFO << ")" << std::endl;

    const auto hardware_concurrency = omp_get_max_threads();

    ArgParams args;

    if (!argument_parsing(argc, argv, &args)) {
        print_usage(argv[0]);
        exit(1);
    }

    std::cout << "Use \"-h\" argument to see available options" << std::endl;

    // Print mode information
    std::cout << "# Mode: " << (args.use_zeros ? "zeros" : "random") << std::endl;
    std::cout << "# Workload: " << (args.run_gemm ? "gemm" : "compute") << std::endl;

    const size_t VEC_WIDTH = 1024 * 1024 * args.vecwidth;

    std::unique_ptr<double[]> c;

    c.reset(new (std::align_val_t(64)) double[VEC_WIDTH]);

    std::cout << "Working memory size: " << args.vecwidth * sizeof(double) << "MB"
              << std::endl;
    std::cout << "Total threads: " << hardware_concurrency << std::endl;

    if (args.run_gemm) {
        runGemmBenchmark(args.matrix_size, args.use_zeros);
    } else {
        mixbenchCPU(c.get(), VEC_WIDTH, args.use_zeros);
    }

    return 0;
}
