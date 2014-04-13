#ifndef _MPISTEP_H
#define _MPISTEP_H

#include <stdlib.h>
#include <mpi.h>
#include "../support/singleStep.h"


#include <iostream>
using std::cout;
using std::endl;

// For ease of use
#define grid(i, j) grid[(j) + (i) * cols]
#define newGrid(i, j) newGrid[(j) + (i) * cols]

void step(square_t *grid, square_t *newGrid, const long rows, 
          const long cols, int rank, 
          const int left, const int right, const int above, const int below) {

  //initialize mpi requests/trackers
  MPI_Request topReq, botReq, leftReq, rightReq;
  bool topRecv = false, botRecv = false, sidesSent = false;
  
  //send the top and bottoms rows off
  MPI_Isend(&grid(1,1), (cols-2)*sizeof(square_t), MPI_BYTE, above, 1,
            MPI_COMM_WORLD, &topReq);
  MPI_Isend(&grid(rows-2,1), (cols-2)*sizeof(square_t), MPI_BYTE, below, 2,
            MPI_COMM_WORLD, &botReq);
  
  //initialize and fill the buffers to hold the outgoing left/right data
  square_t* leftSendBuff = (square_t*) malloc(rows*sizeof(square_t));
  square_t* rightSendBuff =(square_t*) malloc(rows*sizeof(square_t));
  for(long i = 1; i < rows-1; i++) {
    leftSendBuff[i] = grid(i, 1);
    rightSendBuff[i] = grid(i, cols-2);
  }

  //update the inner parts of the grid (that don't need info from other grids)
  for(long i = 2; i < rows-2; i++) {
    for(long j = 2; j < cols-2; j++) {
      newGrid(i, j) = singleStep(grid, i, j, rows, cols);
    }
  }

  MPI_Recv(&grid(0, 1), (cols-2)*sizeof(square_t), MPI_BYTE, above, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  leftSendBuff[0] = grid(0, 1);
  rightSendBuff[0] = grid(0, cols-2);
  topRecv = true;
  MPI_Recv(&grid(rows-1 ,1), (cols-2)*sizeof(square_t), MPI_BYTE, below, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  leftSendBuff[rows-1] = grid(rows-1, 1);
  rightSendBuff[rows-1] = grid(rows-1, cols-2);
  botRecv = true;
  
  MPI_Isend(leftSendBuff, rows*sizeof(square_t), MPI_BYTE, left, 3, MPI_COMM_WORLD, &leftReq);
  MPI_Isend(rightSendBuff, rows*sizeof(square_t), MPI_BYTE, right, 4, MPI_COMM_WORLD, &rightReq);
  sidesSent = true;
  (void) topRecv;
  (void) botRecv;
  (void) sidesSent;

  //initialize, fill, and dump the incoming left/right data into the old grid's edges
  square_t* leftRecvBuff = (square_t*) malloc(rows*sizeof(square_t));
  square_t* rightRecvBuff = (square_t*) malloc(rows*sizeof(square_t));
  MPI_Recv(leftRecvBuff, rows*sizeof(square_t), MPI_BYTE, left, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Recv(rightRecvBuff, rows*sizeof(square_t), MPI_BYTE, right, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  
  for(long i = 0; i < rows; i++) {
    grid(i, 0) = leftRecvBuff[i];
    grid(i, cols-1) = rightRecvBuff[i];
  }

  //compute the behavior on the edges, now that that information is available
  for(long i = 1; i < rows-1; i++) {
    newGrid(i, 1) = singleStep(grid, i, 1, rows, cols);
    newGrid(i, cols-2) = singleStep(grid, i, cols-2, rows, cols);
  }
  for(long j = 1; j < cols-1; j++) {
    newGrid(1, j) = singleStep(grid, 1, j, rows, cols);
    newGrid(rows-2, j) = singleStep(grid, rows-2, j, rows, cols);
  }

  (void) rank;
  
  //wait for neighbors to have all information they need
  MPI_Wait(&topReq, MPI_STATUS_IGNORE);
  MPI_Wait(&botReq, MPI_STATUS_IGNORE);
  MPI_Wait(&leftReq, MPI_STATUS_IGNORE);
  MPI_Wait(&rightReq, MPI_STATUS_IGNORE);

  free(leftSendBuff);
  free(leftRecvBuff);
  free(rightSendBuff);
  free(rightRecvBuff);
}

#endif //_MPISTEP_H
