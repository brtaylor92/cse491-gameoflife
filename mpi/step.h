#ifndef _MPISTEP_H
#define _MPISTEP_H

#include "../support/singleStep.h"

// For ease of use
#define grid(i, j) grid[(i) + (j) * rows]
#define newGrid(i, j) newGrid[(i) + (j) * rows]

void step(const square_t *grid, square_t *newGrid, const long rows, 
          const long cols, 
          const int left, const int right, const int above, const int below) {
  }

#endif //_MPISTEP_H
