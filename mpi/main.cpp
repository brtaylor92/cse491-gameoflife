#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

/*******************************************************************************
 * Grid type
 ******************************************************************************/
typedef unsigned char square_t;

/*******************************************************************************
 * Input/Output
 ******************************************************************************/
#include "../support/io.h"

/*******************************************************************************
 * Step and goForth functions
 ******************************************************************************/
#include "step.h"
#include "goForth.h"

/*******************************************************************************
 * Main
 *    Check command line args
 *    Read input
 *    Populate grid
 *    Call goForthAndMultiply()
 ******************************************************************************/
int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if(argc != 6) {
      if(rank == 0) printf("format: %s [fileName] [numRows] [numCols] [numSteps] [minSize]\n", argv[0]);
      MPI_Finalize();
      return 1;
  }

  FILE *fp = nullptr;
  bool error = false;
  if(rank == 0) {
    fp = fopen(argv[1], "r");
    if(!fp) {
      printf("Could not open file\n");
      error = true;
    }
    MPI_Bcast(&error, sizeof(bool), MPI_BYTE, 0, MPI_COMM_WORLD);
  } else {
    MPI_Bcast(&error, sizeof(bool), MPI_BYTE, 0, MPI_COMM_WORLD);
  }
  if(error) {
    MPI_Finalize();
    return 1;
  }

  const long rows = strtol(argv[2], NULL, 10);
  if (errno != 0 || rows == 0) {
    if(rank == 0) printf("Unable to process argument [numRows] as int\n");
    MPI_Finalize();
    return 1;
  }

  const long cols = strtol(argv[3], NULL, 10);
  if (errno != 0 || cols == 0) {
    if(rank == 0) printf("Unable to process argument [numCols] as int\n");
    MPI_Finalize();
    return 1;
  }

  const long numSteps = strtol(argv[4], NULL, 10);
  if (errno != 0) {
    if(rank == 0) printf("Unable to process argument [numSteps] as int\n");
    MPI_Finalize();
    return 1;
  }

  const long minSize = strtol(argv[5], NULL, 10);
  if (errno != 0) {
    if(rank == 0) printf("Unable to process argument [minSize] as int\n");
    MPI_Finalize();
    return 1;
  }

  square_t *gridA = nullptr, *gridB = nullptr;
  if(rank == 0) {
    gridA = (square_t*) malloc(sizeof(square_t) * rows * cols);
    gridB = (square_t*) malloc(sizeof(square_t) * rows * cols);
  
    readGrid(gridA, rows, cols, fp);
    fclose(fp);
  }
  goForthAndMultiply(gridA, gridB, rows, cols, numSteps, rank, size, minSize);

  if(rank == 0) {
    free(gridA);
    free(gridB);
  }

  MPI_Finalize();
  return 0;
}
