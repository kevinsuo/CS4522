#include <stdio.h>

// CUDA kernel
__global__ void hello_cuda() {
int i = threadIdx.x;
if (i < 5) {
printf("Hello World CUDA from thread %d\n", i);
}
}

int main() {
// Launch 1 block with 5 threads
printf("This is the main thread\n");
hello_cuda<<<1, 5>>>();

// Wait for GPU to finish
cudaDeviceSynchronize();

return 0;
}
