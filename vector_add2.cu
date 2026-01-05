#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

// Define thread block size and shared memory size
#define THREADS_PER_BLOCK 256
#define SHARED_MEM_SIZE THREADS_PER_BLOCK

        // Kernel function definition
        __global__ void
        vectorAdd(const float* A, const float* B, float* C, int numElements) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  // Use shared memory
  __shared__ float shared_A[SHARED_MEM_SIZE];
  __shared__ float shared_B[SHARED_MEM_SIZE];

  if (i < numElements) {
    shared_A[threadIdx.x] = A[i];
    shared_B[threadIdx.x] = B[i];
    __syncthreads();  // Ensure all threads have loaded data into shared memory

    C[i] = shared_A[threadIdx.x] + shared_B[threadIdx.x];
  }
}

int main(void) {
  int numElements = 50000;
  size_t size = numElements * sizeof(float);
  printf("[Vector addition of %d elements]\n", numElements);

  // Dynamically allocate vectors on the host
  float* h_A = new float[numElements];
  float* h_B = new float[numElements];
  float* h_C = new float[numElements];

  // Initialize vectors
  for (int i = 0; i < numElements; ++i) {
    h_A[i] = rand() / (float)RAND_MAX;
    h_B[i] = rand() / (float)RAND_MAX;
  }

  // Allocate input vectors A, B and output vector C on the device
  float* d_A = nullptr;
  float* d_B = nullptr;
  float* d_C = nullptr;
  cudaMalloc((void**)&d_A, size);
  cudaMalloc((void**)&d_B, size);
  cudaMalloc((void**)&d_C, size);

  // // Copy vector data from host to device
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  // Call the kernel function
  int threadsPerBlock = THREADS_PER_BLOCK;
  int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

  // Copy the result from device back to host
  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  // Verify the result
  for (int i = 0; i < numElements; ++i) {
    if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
      fprintf(stderr, "Result verification failed at element %d!\n", i);
      exit(EXIT_FAILURE);
    }
  }

  printf("Test PASSED\n");

  // Free device memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  // Free host memory
  delete[] h_A;
  delete[] h_B;
  delete[] h_C;

  return 0;
}
