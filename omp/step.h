#ifndef _STEP_H
#define _STEP_H

#include "../support/singleStep.h"

// For ease of use
#define grid(i, j) grid[(i) + (j) * rows]
#define newGrid(i, j) newGrid[(i) + (j) * rows]

void step(square_t *grid, square_t *newGrid, const long rows, const long cols) {
  #pragma omp parallel for
  for(long i = 0; i < rows; i++) {
    for(long j = 0; j < cols; j++) {
      newGrid(i, j) = singleStep(grid, i, j, rows, cols);
    }
  }
}

#endif //_STEP_H