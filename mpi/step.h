#ifndef _MPISTEP_H
#define _MPISTEP_H

#include <stdlib.h>
#include <mpi.h>
#include "../support/singleStep.h"

// For ease of use
#define grid(i, j) grid[(j) + (i) * rows]
#define newGrid(i, j) newGrid[(j) + (i) * rows]

void step(square_t *grid, square_t *newGrid, const long rows, 
          const long cols, 
          const int left, const int right, const int above, const int below) {
  
  //initialize mpi requests/trackers
  MPI_Request topReq, botReq, leftReq, rightReq;
  bool topRecv = false, botRecv = false, sidesSent = false;
  
  //send the top and bottoms rows off
  MPI_Isend(grid+(cols+2)+1, cols*sizeof(square_t), MPI_BYTE, above, 0,
            MPI_COMM_WORLD, &topReq);
  MPI_Isend(grid+rows*(cols+2)+1, cols*sizeof(square_t), MPI_BYTE, below, 0,
            MPI_COMM_WORLD, &botReq);

  //initialize and fill the buffers to hold the outgoing left/right data
  square_t* leftSendBuff = (square_t*) malloc((rows+2)*sizeof(square_t));
  square_t* rightSendBuff =(square_t*) malloc((rows+2)*sizeof(square_t));
  for(long i = 1; i < cols+1; i++) {
    leftSendBuff[i] = grid(i, 1);
    rightSendBuff[i] = grid(i, rows);
  }

  //update the inner parts of the grid (that don't need info from other grids)
  for(long i = 1; i < rows; i++) {
    for(long j = 1; j < cols; j++) {
      newGrid(i, j) = singleStep(grid, i, j, rows+2, cols+2);
    }
  }

  MPI_Recv(grid, rows*sizeof(square_t), MPI_BYTE, above, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  leftSendBuff[0] = grid(0, 1);
  rightSendBuff[0] = grid(0, rows);
  topRecv = true;
  MPI_Recv(grid, rows*sizeof(square_t), MPI_BYTE, below, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  leftSendBuff[cols+1] = grid(cols+1, 1);
  rightSendBuff[cols+1] = grid(cols+1, rows);
  botRecv = true;
  MPI_Isend(leftSendBuff, (cols+2)*sizeof(square_t), MPI_BYTE, left, 0, MPI_COMM_WORLD, &leftReq);
  MPI_Isend(rightSendBuff, (cols+2)*sizeof(square_t), MPI_BYTE, right, 0, MPI_COMM_WORLD, &rightReq);
  sidesSent = true;

  (void) topRecv;
  (void) botRecv;
  (void) sidesSent;

  //initialize, fill, and dump the incoming left/right data into the old grid's edges
  square_t* leftRecvBuff = (square_t*) malloc((rows+2)*sizeof(square_t));
  square_t* rightRecvBuff = (square_t*) malloc((rows+2)*sizeof(square_t));
  MPI_Recv(leftRecvBuff, (rows+2)*sizeof(square_t), MPI_BYTE, left, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Recv(rightRecvBuff, (rows+2)*sizeof(square_t), MPI_BYTE, right, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  for(long i = 0; i < rows+2; i++) {
    grid(i, 0) = leftRecvBuff[i];
    grid(i, rows+1) = rightRecvBuff[i];
  }

  //compute the behavior on the edges, now that that information is available
  for(long i = 1; i < rows+1; i++) {
    newGrid(i, 1) = singleStep(grid, i, 1, rows+2, cols+2);
    newGrid(i, cols) = singleStep(grid, i, cols, rows+2, cols+2);
  }
  for(long j = 1; j < cols+1; j++) {
    newGrid(1, j) = singleStep(grid, 1, j, rows+2, cols+2);
    newGrid(rows, j) = singleStep(grid, rows, j, rows+2, cols+2);
  }

  //wait for neighbors to have all information they need
  MPI_Wait(&topReq, MPI_STATUS_IGNORE);
  MPI_Wait(&botReq, MPI_STATUS_IGNORE);
  MPI_Wait(&leftReq, MPI_STATUS_IGNORE);
  MPI_Wait(&rightReq, MPI_STATUS_IGNORE);
}

#endif //_MPISTEP_H
