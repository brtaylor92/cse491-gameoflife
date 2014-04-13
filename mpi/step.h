#ifndef _MPISTEP_H
#define _MPISTEP_H

#include "../support/singleStep.h"

// For ease of use
#define grid(i, j) grid[(j) + (i) * rows]
#define newGrid(i, j) newGrid[(j) + (i) * rows]

void step(const square_t *grid, square_t *newGrid, const long rows, 
          const long cols, 
          const int left, const int right, const int above, const int below) {
  (void) grid;
  (void) newGrid;
  (void) rows;
  (void) cols;
  (void) left;
  (void) right;
  (void) above;
  (void) below;

}

#endif //_MPISTEP_H
