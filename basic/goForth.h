#ifndef _GOFORTH_H
#define _GOFORTH_H

#ifdef omp
  #include "../omp/step.h"
#endif

#ifdef cilk
  #include "../cilk/step.h"
#endif

#if defined(basic)
  #include "step.h"
#endif

void goForthAndMultiply(square_t *gridA, square_t *gridB, const long rows, 
                        const long cols, const long numSteps) {
  //printGrid(gridA, rows, cols);
  for(long i = 0; i < numSteps; i++) {
    step(gridA, gridB, rows, cols);
    square_t *temp = gridA;
    gridA = gridB;
    gridB = temp;
    //printGrid(gridA, rows, cols);
  }
}

#endif //_GOFORTH_H