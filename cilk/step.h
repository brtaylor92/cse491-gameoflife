#ifndef _STEP_H
#define _STEP_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cilk/cilk.h>

#include "../support/singleStep.h"

// For ease of use
#define grid(i, j) grid[(j) + (i) * cols]
#define newGrid(i, j) newGrid[(j) + (i) * cols]
typedef unsigned char square_t;

void step(square_t *grid, square_t *newGrid, const long rows, const long cols) {
  cilk_for(long i = 0; i < rows; i++) {
    #pragma simd
    for(long j = 0; j < cols; j++) {
      newGrid(i, j) = singleStep(grid, i, j, rows, cols);
    }
  }
}

#endif //_STEP_H