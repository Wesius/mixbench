/**
 * mix_kernels_cuda_ro.cu: This file is part of the mixbench GPU micro-benchmark suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#include <stdio.h>
#include <math_constants.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>
#include <sys/time.h>
#include <cublas_v2.h>
#include "lcutil.h"

// Get current Unix timestamp in seconds with microsecond precision
static double get_timestamp() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

#define ELEMENTS_PER_THREAD (8)
#define FUSION_DEGREE (4)

template<class T>
inline __device__ T conv_int(const int i){ return static_cast<T>(i); }

template<class T>
inline __device__ T mad(const T a, const T b, const T c){ return a*b+c; }

template<class T>
inline __device__ bool equal(const T a, const T b){ return a==b; }

#if __CUDA_ARCH__ >= 530
template<>
inline __device__ half2 conv_int(const int i){ return __half2half2( __int2half_rd(i) ); }
template<>
inline __device__ half2 mad(const half2 a, const half2 b, const half2 c){ return __hfma2(a, b, c)/*__hadd2(__hmul2(a, b), c)*/; }
template<>
inline __device__ bool equal(const half2 a, const half2 b){ return __hbeq2(a, b); }
#else
// a dummy implementations as a workaround
template<>
inline __device__ half2 conv_int(const int i){ return half2(); }
template<>
inline __device__ half2 mad(const half2 a, const half2 b, const half2 c){ return half2(); }
template<>
inline __device__ bool equal(const half2 a, const half2 b){ return false; }
#endif

template <class T, int blockdim, unsigned int granularity, unsigned int fusion_degree, unsigned int compute_iterations, bool TemperateUnroll>
__global__ void benchmark_func(T seed, T *g_data){
	const unsigned int blockSize = blockdim;
	const int stride = blockSize;
	int idx = blockIdx.x*blockSize*granularity + threadIdx.x;
	const int big_stride = gridDim.x*blockSize*granularity;

	T tmps[granularity];
	for(int k=0; k<fusion_degree; k++){
		#pragma unroll
		for(int j=0; j<granularity; j++){
			// Load elements (memory intensive part)
			tmps[j] = g_data[idx+j*stride+k*big_stride];
			// Perform computations (compute intensive part)
			#pragma unroll TemperateUnroll ? 4 : 128
			for(int i=0; i<compute_iterations; i++){
				tmps[j] = mad(tmps[j], tmps[j], seed);
			}
		}
		// Multiply add reduction
		T sum = conv_int<T>(0);
		#pragma unroll
		for(int j=0; j<granularity; j+=2)
			sum = mad(tmps[j], tmps[j+1], sum);
		// Dummy code
		if( equal(sum, conv_int<T>(-1)) ) // Designed so it never executes
			g_data[idx+k*big_stride] = sum;
	}
}

void initializeEvents(cudaEvent_t *start, cudaEvent_t *stop){
	CUDA_SAFE_CALL( cudaEventCreate(start) );
	CUDA_SAFE_CALL( cudaEventCreate(stop) );
	CUDA_SAFE_CALL( cudaEventRecord(*start, 0) );
}

float finalizeEvents(cudaEvent_t start, cudaEvent_t stop){
	CUDA_SAFE_CALL( cudaGetLastError() );
	CUDA_SAFE_CALL( cudaEventRecord(stop, 0) );
	CUDA_SAFE_CALL( cudaEventSynchronize(stop) );
	float kernel_time;
	CUDA_SAFE_CALL( cudaEventElapsedTime(&kernel_time, start, stop) );
	CUDA_SAFE_CALL( cudaEventDestroy(start) );
	CUDA_SAFE_CALL( cudaEventDestroy(stop) );
	return kernel_time;
}

// Initialize host data with random values in range [1.0, 2.0] to avoid denormals
template<typename T>
void init_random_data(T* data, size_t n) {
	srand(42);  // Fixed seed for reproducibility
	for (size_t i = 0; i < n; i++) {
		data[i] = (T)(1.0 + (double)rand() / (double)RAND_MAX);
	}
}

void runbench_warmup(double *cd, long size){
	const long reduced_grid_size = size/(ELEMENTS_PER_THREAD)/128;
	const int BLOCK_SIZE = 256;
	const int TOTAL_REDUCED_BLOCKS = reduced_grid_size/BLOCK_SIZE;

	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimReducedGrid(TOTAL_REDUCED_BLOCKS, 1, 1);

	benchmark_func< short, BLOCK_SIZE, ELEMENTS_PER_THREAD, FUSION_DEGREE, 0, true ><<< dimReducedGrid, dimBlock >>>((short)1, (short*)cd);
	CUDA_SAFE_CALL( cudaGetLastError() );
	CUDA_SAFE_CALL( cudaDeviceSynchronize() );
}

int out_config = 1;

// Global duration setting for sustained workloads
static float g_duration_per_point = 0.0f;

template<unsigned int compute_iterations>
void runbench(double *cd, long size, bool doHalfs){
	const long compute_grid_size = size/ELEMENTS_PER_THREAD/FUSION_DEGREE;
	const int BLOCK_SIZE = 256;
	const int TOTAL_BLOCKS = compute_grid_size/BLOCK_SIZE;
	const long long computations = (ELEMENTS_PER_THREAD*(long long)compute_grid_size+(2*ELEMENTS_PER_THREAD*compute_iterations)*(long long)compute_grid_size)*FUSION_DEGREE;
	const long long memoryoperations = size;

	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid(TOTAL_BLOCKS, 1, 1);
	cudaEvent_t start, stop;

	// Determine number of iterations based on duration_per_point
	// If duration_per_point > 0, run repeatedly until duration is reached
	int iterations = 1;
	float target_duration_ms = g_duration_per_point * 1000.0f;

	// Single precision
	initializeEvents(&start, &stop);
	benchmark_func< float, BLOCK_SIZE, ELEMENTS_PER_THREAD, FUSION_DEGREE, compute_iterations, false ><<< dimGrid, dimBlock >>>(1.0f, (float*)cd);
	float kernel_time_mad_sp = finalizeEvents(start, stop);
	int sp_iterations = 1;
	double sp_start_ts = 0, sp_end_ts = 0;

	if (target_duration_ms > 0 && kernel_time_mad_sp > 0) {
		sp_iterations = (int)(target_duration_ms / kernel_time_mad_sp) + 1;
		// Always run sustained workload in duration mode (minimum 2 iterations)
		sp_iterations = sp_iterations < 2 ? 2 : sp_iterations;
		sp_start_ts = get_timestamp();
		initializeEvents(&start, &stop);
		for (int i = 0; i < sp_iterations; i++) {
			benchmark_func< float, BLOCK_SIZE, ELEMENTS_PER_THREAD, FUSION_DEGREE, compute_iterations, false ><<< dimGrid, dimBlock >>>(1.0f, (float*)cd);
		}
		// Report total time for power measurement, not average
		kernel_time_mad_sp = finalizeEvents(start, stop);
		sp_end_ts = get_timestamp();
	}

	// Double precision
	initializeEvents(&start, &stop);
	benchmark_func< double, BLOCK_SIZE, ELEMENTS_PER_THREAD, FUSION_DEGREE, compute_iterations, false ><<< dimGrid, dimBlock >>>(1.0, cd);
	float kernel_time_mad_dp = finalizeEvents(start, stop);
	int dp_iterations = 1;
	double dp_start_ts = 0, dp_end_ts = 0;

	if (target_duration_ms > 0 && kernel_time_mad_dp > 0) {
		dp_iterations = (int)(target_duration_ms / kernel_time_mad_dp) + 1;
		dp_iterations = dp_iterations < 2 ? 2 : dp_iterations;
		dp_start_ts = get_timestamp();
		initializeEvents(&start, &stop);
		for (int i = 0; i < dp_iterations; i++) {
			benchmark_func< double, BLOCK_SIZE, ELEMENTS_PER_THREAD, FUSION_DEGREE, compute_iterations, false ><<< dimGrid, dimBlock >>>(1.0, cd);
		}
		// Report total time for power measurement, not average
		kernel_time_mad_dp = finalizeEvents(start, stop);
		dp_end_ts = get_timestamp();
	}

	// Half precision
	float kernel_time_mad_hp = 0.f;
	int hp_iterations = 1;
	double hp_start_ts = 0, hp_end_ts = 0;
	if( doHalfs ){
		initializeEvents(&start, &stop);
		half2 h_ones;
		*((int32_t*)&h_ones) = 15360 + (15360 << 16); // 1.0 as half
		benchmark_func< half2, BLOCK_SIZE, ELEMENTS_PER_THREAD, FUSION_DEGREE, compute_iterations, false ><<< dimGrid, dimBlock >>>(h_ones, (half2*)cd);
		kernel_time_mad_hp = finalizeEvents(start, stop);

		if (target_duration_ms > 0 && kernel_time_mad_hp > 0) {
			hp_iterations = (int)(target_duration_ms / kernel_time_mad_hp) + 1;
			hp_iterations = hp_iterations < 2 ? 2 : hp_iterations;
			hp_start_ts = get_timestamp();
			initializeEvents(&start, &stop);
			for (int i = 0; i < hp_iterations; i++) {
				benchmark_func< half2, BLOCK_SIZE, ELEMENTS_PER_THREAD, FUSION_DEGREE, compute_iterations, false ><<< dimGrid, dimBlock >>>(h_ones, (half2*)cd);
			}
			// Report total time for power measurement, not average
			kernel_time_mad_hp = finalizeEvents(start, stop);
			hp_end_ts = get_timestamp();
		}
	}

	// Integer
	initializeEvents(&start, &stop);
	benchmark_func< int, BLOCK_SIZE, ELEMENTS_PER_THREAD, FUSION_DEGREE, compute_iterations, true ><<< dimGrid, dimBlock >>>(1, (int*)cd);
	float kernel_time_mad_int = finalizeEvents(start, stop);
	int int_iterations = 1;
	double int_start_ts = 0, int_end_ts = 0;

	if (target_duration_ms > 0 && kernel_time_mad_int > 0) {
		int_iterations = (int)(target_duration_ms / kernel_time_mad_int) + 1;
		int_iterations = int_iterations < 2 ? 2 : int_iterations;
		int_start_ts = get_timestamp();
		initializeEvents(&start, &stop);
		for (int i = 0; i < int_iterations; i++) {
			benchmark_func< int, BLOCK_SIZE, ELEMENTS_PER_THREAD, FUSION_DEGREE, compute_iterations, true ><<< dimGrid, dimBlock >>>(1, (int*)cd);
		}
		// Report total time for power measurement, not average
		kernel_time_mad_int = finalizeEvents(start, stop);
		int_end_ts = get_timestamp();
	}

	// When using duration mode, scale computations and memory by iterations
	// to get correct GFLOPS and GB/s for total work done
	long long sp_computations = computations * sp_iterations;
	long long dp_computations = computations * dp_iterations;
	long long hp_computations = (long long)2 * computations * hp_iterations;
	long long int_computations = computations * int_iterations;
	long long sp_memory = memoryoperations * sp_iterations;
	long long dp_memory = memoryoperations * dp_iterations;
	long long hp_memory = memoryoperations * hp_iterations;
	long long int_memory = memoryoperations * int_iterations;

	printf("         %4d,   %8.3f,%8.2f,%8.2f,%7.2f,   %8.3f,%8.2f,%8.2f,%7.2f,   %8.3f,%8.2f,%8.2f,%7.2f,  %8.3f,%8.2f,%8.2f,%7.2f\n",
		compute_iterations,
		((double)computations)/((double)memoryoperations*sizeof(float)),  // AI is same regardless of iterations
		kernel_time_mad_sp,
		((double)sp_computations)/kernel_time_mad_sp*1000./(double)(1000*1000*1000),
		((double)sp_memory*sizeof(float))/kernel_time_mad_sp*1000./(1000.*1000.*1000.),
		((double)computations)/((double)memoryoperations*sizeof(double)),
		kernel_time_mad_dp,
		((double)dp_computations)/kernel_time_mad_dp*1000./(double)(1000*1000*1000),
		((double)dp_memory*sizeof(double))/kernel_time_mad_dp*1000./(1000.*1000.*1000.),
		((double)2*computations)/((double)memoryoperations*sizeof(half2)),
		kernel_time_mad_hp,
		((double)hp_computations)/kernel_time_mad_hp*1000./(double)(1000*1000*1000),
		((double)hp_memory*sizeof(half2))/kernel_time_mad_hp*1000./(1000.*1000.*1000.),
		((double)computations)/((double)memoryoperations*sizeof(int)),
		kernel_time_mad_int,
		((double)int_computations)/kernel_time_mad_int*1000./(double)(1000*1000*1000),
		((double)int_memory*sizeof(int))/kernel_time_mad_int*1000./(1000.*1000.*1000.) );

	// Output timestamps for power correlation (only in duration mode)
	if (g_duration_per_point > 0) {
		printf("# TS:%d,sp,%.6f,%.6f,dp,%.6f,%.6f,hp,%.6f,%.6f,int,%.6f,%.6f\n",
			compute_iterations,
			sp_start_ts, sp_end_ts,
			dp_start_ts, dp_end_ts,
			hp_start_ts, hp_end_ts,
			int_start_ts, int_end_ts);
		fflush(stdout);  // Ensure timestamps are output immediately
	}
}

extern "C" void mixbenchGPU(double *c, long size, bool use_zeros, float duration_per_point){
	g_duration_per_point = duration_per_point;
	if (duration_per_point > 0) {
		printf("# Sustained duration per AI point: %.1f seconds\n", duration_per_point);
	}
	const char *benchtype = "compute with global memory (block strided)";

	printf("Trade-off type:       %s\n", benchtype);
	printf("Elements per thread:  %d\n", ELEMENTS_PER_THREAD);
	printf("Thread fusion degree: %d\n", FUSION_DEGREE);
	double *cd;
	bool doHalfs = IsFP16Supported();
	if( !doHalfs )
		printf("Warning:              Half precision computations are not supported\n");

	CUDA_SAFE_CALL( cudaMalloc((void**)&cd, size*sizeof(double)) );

	if (use_zeros) {
		// Initialize to zeros (control energy mode)
		CUDA_SAFE_CALL( cudaMemset(cd, 0, size*sizeof(double)) );
	} else {
		// Initialize with random data in range [1.0, 2.0] to avoid denormals
		init_random_data(c, size);
		CUDA_SAFE_CALL( cudaMemcpy(cd, c, size*sizeof(double), cudaMemcpyHostToDevice) );
	}

	// Synchronize in order to wait for memory operations to finish
	CUDA_SAFE_CALL( cudaDeviceSynchronize() );

	printf("----------------------------------------------------------------------------- CSV data -----------------------------------------------------------------------------\n");
	printf("Experiment ID, Single Precision ops,,,,              Double precision ops,,,,              Half precision ops,,,,                Integer operations,,, \n");
	printf("Compute iters, Flops/byte, ex.time,  GFLOPS, GB/sec, Flops/byte, ex.time,  GFLOPS, GB/sec, Flops/byte, ex.time,  GFLOPS, GB/sec, Iops/byte, ex.time,   GIOPS, GB/sec\n");

	runbench_warmup(cd, size);

	runbench<0>(cd, size, doHalfs);
	runbench<1>(cd, size, doHalfs);
	runbench<2>(cd, size, doHalfs);
	runbench<3>(cd, size, doHalfs);
	runbench<4>(cd, size, doHalfs);
	runbench<5>(cd, size, doHalfs);
	runbench<6>(cd, size, doHalfs);
	runbench<7>(cd, size, doHalfs);
	runbench<8>(cd, size, doHalfs);
	runbench<9>(cd, size, doHalfs);
	runbench<10>(cd, size, doHalfs);
	runbench<11>(cd, size, doHalfs);
	runbench<12>(cd, size, doHalfs);
	runbench<13>(cd, size, doHalfs);
	runbench<14>(cd, size, doHalfs);
	runbench<15>(cd, size, doHalfs);
	runbench<16>(cd, size, doHalfs);
	runbench<17>(cd, size, doHalfs);
	runbench<18>(cd, size, doHalfs);
	runbench<20>(cd, size, doHalfs);
	runbench<22>(cd, size, doHalfs);
	runbench<24>(cd, size, doHalfs);
	runbench<28>(cd, size, doHalfs);
	runbench<32>(cd, size, doHalfs);
	runbench<40>(cd, size, doHalfs);
	runbench<48>(cd, size, doHalfs);
	runbench<56>(cd, size, doHalfs);
	runbench<64>(cd, size, doHalfs);
	runbench<80>(cd, size, doHalfs);
	runbench<96>(cd, size, doHalfs);
	runbench<128>(cd, size, doHalfs);
	runbench<192>(cd, size, doHalfs);
	runbench<256>(cd, size, doHalfs);
	runbench<512>(cd, size, doHalfs);
	runbench<1024>(cd, size, doHalfs);

	printf("--------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");

	// Copy results back to host memory
	CUDA_SAFE_CALL( cudaMemcpy(c, cd, size*sizeof(double), cudaMemcpyDeviceToHost) );

	CUDA_SAFE_CALL( cudaFree(cd) );

	CUDA_SAFE_CALL( cudaDeviceReset() );
}

// GEMM benchmark using cuBLAS
extern "C" void runGemmBenchmark(int M, bool use_zeros) {
	cublasHandle_t handle;
	cublasStatus_t status;

	status = cublasCreate(&handle);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "Error: cuBLAS initialization failed\n");
		return;
	}

	// Allocate device memory
	float *d_A, *d_B, *d_C;
	size_t matrix_size = (size_t)M * M * sizeof(float);

	CUDA_SAFE_CALL( cudaMalloc(&d_A, matrix_size) );
	CUDA_SAFE_CALL( cudaMalloc(&d_B, matrix_size) );
	CUDA_SAFE_CALL( cudaMalloc(&d_C, matrix_size) );

	if (use_zeros) {
		// Zero-initialized data (control energy mode)
		CUDA_SAFE_CALL( cudaMemset(d_A, 0, matrix_size) );
		CUDA_SAFE_CALL( cudaMemset(d_B, 0, matrix_size) );
	} else {
		// Random data in range [1.0, 2.0]
		float* h_A = (float*)malloc(matrix_size);
		float* h_B = (float*)malloc(matrix_size);
		init_random_data(h_A, (size_t)M * M);
		init_random_data(h_B, (size_t)M * M);
		CUDA_SAFE_CALL( cudaMemcpy(d_A, h_A, matrix_size, cudaMemcpyHostToDevice) );
		CUDA_SAFE_CALL( cudaMemcpy(d_B, h_B, matrix_size, cudaMemcpyHostToDevice) );
		free(h_A);
		free(h_B);
	}
	CUDA_SAFE_CALL( cudaMemset(d_C, 0, matrix_size) );

	float alpha = 1.0f, beta = 0.0f;

	// Warmup
	status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
	                     M, M, M, &alpha, d_A, M, d_B, M, &beta, d_C, M);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "Error: cuBLAS SGEMM failed\n");
		goto cleanup;
	}
	CUDA_SAFE_CALL( cudaDeviceSynchronize() );

	// Print header
	printf("----------------------------------------------------------------------------- CSV data -----------------------------------------------------------------------------\n");
	printf("matrix_size, GFLOPS, time_ms\n");

	// Timed iterations
	{
		cudaEvent_t start, stop;
		CUDA_SAFE_CALL( cudaEventCreate(&start) );
		CUDA_SAFE_CALL( cudaEventCreate(&stop) );

		const int iters = 100;
		CUDA_SAFE_CALL( cudaEventRecord(start) );
		for (int i = 0; i < iters; i++) {
			cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
			            M, M, M, &alpha, d_A, M, d_B, M, &beta, d_C, M);
		}
		CUDA_SAFE_CALL( cudaEventRecord(stop) );
		CUDA_SAFE_CALL( cudaEventSynchronize(stop) );

		float ms;
		CUDA_SAFE_CALL( cudaEventElapsedTime(&ms, start, stop) );

		double flops = 2.0 * (double)M * (double)M * (double)M * iters;
		double gflops = flops / (ms * 1e6);
		double avg_time = ms / iters;

		printf("%d, %.2f, %.3f\n", M, gflops, avg_time);

		CUDA_SAFE_CALL( cudaEventDestroy(start) );
		CUDA_SAFE_CALL( cudaEventDestroy(stop) );
	}

	printf("--------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");

cleanup:
	CUDA_SAFE_CALL( cudaFree(d_A) );
	CUDA_SAFE_CALL( cudaFree(d_B) );
	CUDA_SAFE_CALL( cudaFree(d_C) );
	cublasDestroy(handle);

	CUDA_SAFE_CALL( cudaDeviceReset() );
}
