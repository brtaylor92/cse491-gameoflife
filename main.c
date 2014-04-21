#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*******************************************************************************
 * Grid type
 ******************************************************************************/
#include "support/squaret.h"

/*******************************************************************************
 * Input/Output
 ******************************************************************************/
#include "support/io.h"

/*******************************************************************************
 * Step and goForth functions
 ******************************************************************************/

#ifdef cuda
  #include "cuda/goForth.cuh"
#endif

#if !defined(cuda)
  #include "basic/goForth.h"
#endif

/*******************************************************************************
 * Main
 *    Check command line args
 *    Read input
 *    Populate grid
 *    Call goForthAndMultiply()
 ******************************************************************************/
int main(int argc, char *argv[])
{
  if(argc != 5) {
      printf("format: %s [fileName] [numRows] [numCols] [numSteps]\n", argv[0]);
      return 1;
  }

  FILE *fp = fopen(argv[1], "r");
  if(!fp) {
    printf("Could not open file\n");
    return 1;
  }

  const long rows = strtol(argv[2], NULL, 10);
  if (errno != 0) {
    printf("Unable to process argument [numRows] as int\n");
    return 1;
  }

  const long cols = strtol(argv[3], NULL, 10);
  if (errno != 0) {
    printf("Unable to process argument [numCols] as int\n");
    return 1;
  }

  long numSteps = strtol(argv[4], NULL, 10);
  if (errno != 0) {
    printf("Unable to process argument [numSteps] as int\n");
    return 1;
  }

  #ifdef gui
    initGUI(rows, cols);
  #endif

  
  square_t *gridA = (square_t*) malloc(sizeof(square_t) * rows * cols);
  square_t *gridB = (square_t*) malloc(sizeof(square_t) * rows * cols);
  
  readGrid(gridA, rows, cols, fp);
  fclose(fp);

  goForthAndMultiply(gridA, gridB, rows, cols, numSteps);

  free(gridA);
  free(gridB);

  return 0;
}
