#ifndef _GOFORTH_H
#define _GOFORTH_H

#include <algorithm>
using std::max;
using std::min;
using std::swap;
#include <mpi.h>
#include <vector>
using std::vector;

#include <iostream>
using std::cout;
using std::endl;

void goForthAndMultiply(square_t *gridA, square_t *gridB, const long rows, 
                        const long cols, const long numSteps, const int rank, 
                        const int size, const long minSize) {
	
  //determine the overall grid striping
  double aspectRatio = 1.0*cols/rows;
  long rProcs = max(static_cast<long>(1), 
       min(rows/minSize, static_cast<long>(round(sqrt(size/aspectRatio)))));
  long cProcs = size/rProcs;
  if(cProcs > cols/minSize) {
    cProcs = max(static_cast<long>(1), rows/minSize);
    rProcs = max(static_cast<long>(1), min(rows/minSize, size/cProcs));
  }
  long tileRows = rows/rProcs, tileCols = rows/cProcs;
  long lastRows = rows - (rProcs-1)*tileRows, lastCols = cols - (cProcs-1)*tileCols;

  //each process determines the size of the grid it'll be working with
  long myRows = 0, myCols = 0;
  if(rank < rProcs*cProcs) {
    if(rank%rProcs == rProcs-1) {
      myCols = lastCols;
    } else {
      myCols = tileCols;
    }
    if(rank/rProcs == cProcs-1) {
      myRows = lastRows;
    } else {
      myRows = tileRows;
    }
  }
  
  //each process determines its neighbors
  int left = (rank/rProcs)*rProcs + (rank + rProcs - 1)%rProcs;
  int right = (rank/rProcs)*rProcs + (rank + 1)%rProcs;
  int above = ((rank/rProcs - 1)*rProcs + rank%rProcs + size)%size;
  int below = ((rank/rProcs +1)*rProcs + rank%rProcs)%size;

  //process 0 sends each process the contents of the initial grid its working with
  vector<square_t> myGridA((myRows+2)*(myCols+2), 0), myGridB((myRows+2)*(myCols+2), 0);
  if(rank == 0) {
    cout << "rProcs: " << rProcs << "\tcProcs: " << cProcs << endl
         << "tileRows: " << tileRows << "\ttileCols: " << tileCols << endl
         << "lastRows: " << lastRows << "\tlastCols: " << lastCols << endl
         << "procs used: " << rProcs*cProcs << " (" << 100.0*rProcs*cProcs/size << "%)" << endl;

    for(long i = 0; i < rProcs; i++) {
      long thisTileRows = (i != rProcs-1 ? tileRows : lastRows);
      
      for(long j = 1/(i+1); j < cProcs; j++) {
        long thisTileCols = (j != cProcs-1 ? tileCols : lastCols);
        vector<square_t> outbound((thisTileRows+2)*(thisTileCols+2), 0);
        
        /*for(long k = 0; k < thisTileRows; k++) {
          for(long l = 0; k < thisTileCols; l++) {
            if((i == 0 && k == 0) || (i == rProcs-1 && k == thisTileRows-1)) {
              outbound[(k+1)*thisTileRows+(l+1)] = gridA[(i*tileCols+k)*rows+(j*tileRows+l)];
            }
          }
        }*/
        MPI_Request request;
        cout << "sending data to process " << i*rProcs+j << endl;
        MPI_Isend(outbound.data(), (tileRows+1)*(tileCols+1)*sizeof(square_t), 
                  MPI_BYTE, i*rProcs + j, 0, MPI_COMM_WORLD, &request);
      }
    }
    for(long i = 0; i < myRows+2; i++) {
      for(long j = 0; j < myCols+2; j++) {
        myGridA[(i+1)*myCols+(j+1)] = gridA[i*rows+j];
      }
    }
  } else if(rank < rProcs*cProcs) {
    MPI_Recv(myGridA.data(), (myRows+2)*(myCols+2)*sizeof(square_t), MPI_BYTE,
             0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    cout << "data received in process " << rank << endl;
  }

  int baton = 0;
  if(rank != 0 && rank < rProcs*cProcs) {
    MPI_Recv(&baton, 1, MPI_INT, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  cout << endl << "begin tile " << rank << " (" << myRows << " rows, " << myCols << " cols):" << endl;
  for(long i = 0; i < myRows; i++) {
    for(long j = 0; j < myCols; j++) { 
      cout << int(myGridA[(i+1)*myCols+(j+1)]) << " ";
    }
    cout << endl;
  }
  cout << "end tile " << rank << endl;
  if(rank < rProcs*cProcs-1) {
    MPI_Request request;
    MPI_Isend(&baton, 1, MPI_INT, rank+1, 0, MPI_COMM_WORLD, &request);
  }
  (void) gridA;
  (void) gridB;
  (void) numSteps;
  (void) left;
  (void) right;
  (void) above;
  (void) below;
}
#endif //_GOFORTH_H
