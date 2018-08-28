#include <stdio.h>

// Device code
__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

int main(void)
{
  int N = 1<<20;
// Host Code
  float *x, *y, *d_x, *d_y; 
// x,y Points to the host arrays - d_x, d_y to the device arrays.

// malloc
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));
// cudaMalloc
  cudaMalloc(&d_x, N*sizeof(float)); 
  cudaMalloc(&d_y, N*sizeof(float));

// init host arr
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }
// last option - direction of copy
  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  // Perform SAXPY on 1M elements
  // thread blocks required to process all N elements of the arrays
  saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);
  // Launching grid of thread blocks - number of thread blocks in grid, number of threads in a thread block
  //

  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(y[i]-4.0f));
  printf("Max error: %f\n", maxError);

  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);
}
