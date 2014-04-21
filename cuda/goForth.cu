#include <cuda.h>

#include "../support/squaret.h"
#include "goForth.h"
#include "step.cuh"

void goForthAndMultiply(square_t *gridA, square_t *gridB, const long rows,
                        const long cols, const long numSteps) {
  square_t *gridDevA, *gridDevB;
  cudaMalloc(&gridDevA, rows*cols);
  cudaMalloc(&gridDevB, rows*cols);
  cudaMemcpy(&gridDevA, &gridA, rows*cols, cudaMemcpyHostToDevice);
  cudaMemcpy(&gridDevB, &gridB, rows*cols, cudaMemcpyHostToDevice);

  dim3 dims(cols, rows);

  for(long i = 0; i < numSteps; i++) {
    step<<<1,dims>>>(gridDevA, gridDevB, rows, cols);
  }

  cudaMemcpy(&gridA, &gridDevA, rows*cols, cudaMemcpyDeviceToHost);
  cudaMemcpy(&gridB, &gridDevB, rows*cols, cudaMemcpyDeviceToHost);
  cudaFree(&gridDevA);
  cudaFree(&gridDevB);
}