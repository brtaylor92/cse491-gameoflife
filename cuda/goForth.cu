#include <cuda.h>

#include "../support/squaret.h"
#include "../support/io.h"
#include "goForth.h"
#include "step.cuh"

void goForthAndMultiply(square_t *gridA, square_t *gridB, const long rows,
                        const long cols, const long numSteps) {
  square_t *gridDevA, *gridDevB;
  cudaMalloc(&gridDevA, rows*cols);
  cudaMalloc(&gridDevB, rows*cols);
  cudaMemcpy(gridDevA, gridA, rows*cols, cudaMemcpyHostToDevice);
  cudaMemcpy(gridDevB, gridB, rows*cols, cudaMemcpyHostToDevice);

  dim3 dimBlock(rows/32 + 1, cols/32 + 1);
  dim3 dimGrid(32, 32);
  printGrid(gridA, rows, cols);
  for(long i = 0; i < numSteps; i++) {
    step<<<dimBlock, dimGrid>>>(gridDevA, gridDevB, rows, cols);
    square_t *temp = gridDevA;
    gridDevA = gridDevB;
    gridDevB = temp;
  }

  cudaMemcpy(gridA, gridDevA, rows*cols, cudaMemcpyDeviceToHost);
  printGrid(gridA, rows, cols);

  cudaFree(gridDevA);
  cudaFree(gridDevB);
}