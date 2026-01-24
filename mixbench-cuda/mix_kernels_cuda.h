/**
 * mix_kernels_cuda.h: This file is part of the mixbench GPU micro-benchmark suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#pragma once

extern "C" void mixbenchGPU(double*, long size, bool use_zeros, float duration_per_point = 0.0f);
extern "C" void runGemmBenchmark(int matrix_size, bool use_zeros);
