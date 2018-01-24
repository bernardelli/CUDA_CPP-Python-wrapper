#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <tiny_helper_cuda.h>

__global__ void
vectormult_kernel(float *A, float k, int numElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < numElements)
	{
		A[i] *= k;
	}
}

__global__ void
vectoradd_kernel(const float* A, const float* B,  float* C, int numElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < numElements)
	{
		C[i] = A[i] + B[i];
	}
}




extern "C"
void vectormult(float* dev_vector, float k, int size)
{
	const int n_threads = 128;
	const int n_blocks = (size + n_threads - 1) / n_threads;
	vectormult_kernel <<<n_blocks, n_threads >>>(dev_vector, k, size);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("vectormult_kernel");
}

extern "C"
void vectoradd(const float* A, const float* B,  float* C, int size)
{
	const int n_threads = 128;
	const int n_blocks = (size + n_threads - 1) / n_threads;
	vectoradd_kernel <<<n_blocks, n_threads >>>(A, B, C, size);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("vectormult_kernel");
}