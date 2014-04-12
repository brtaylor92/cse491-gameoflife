#ifndef _BASICSTEP_H
#define _BASICSTEP_H

#include "../support/singleStep.h"

// For ease of use
#define grid(i, j) grid[(i) + (j) * rows]
#define newGrid(i, j) newGrid[(i) + (j) * rows]

void step(const square_t *grid, square_t *newGrid, const long rows, 
          const long cols) {
  for(long i = 0; i < rows; i++) {
    for(long j = 0; j < cols; j++) {
        newGrid(i, j) = singleStep(grid, i, j, rows, cols);
    }
  }
}

#endif //_BASICSTEP_H