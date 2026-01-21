/**
 * mix_kernels_hip.h: This file is part of the mixbench GPU micro-benchmark suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#pragma once

extern "C" void mixbenchGPU(double*, long size, bool use_zeros);
extern "C" void runGemmBenchmark(int matrix_size, bool use_zeros);
