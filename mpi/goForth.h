#ifndef _GOFORTH_H
#define _GOFORTH_H

#include <algorithm>
using std::max;
using std::min;
using std::swap;
#include <mpi.h>
#include <vector>

void goForthAndMultiply(square_t *gridA, square_t *gridB, const long rows, 
                        const long cols, const long numSteps, const int rank, 
                        const int size, const long minSize) {
	
  //determine the overall grid striping
  double aspectRatio = 1.0*cols/rows;
  int rProcs = max(1, min(rows/minSize, int(round(sqrt(size/aspectRatio)))));
  int cProcs = numProcs/rProcs;
  if(cProcs > cols/minSize) {
    cProcs = max(1, width/minSize);
    rProcs = max(1, min(rows/minSize, numProcs/cProcs));
  }
  int rSubsize = rows/rProcs, cSubSize = width/cProcs;
  int rSubLast = rows - (rProcs-1)*rSubSize, cSubLast = cols - (cProcs-1)*cSubSize;

  //each process determines the size of the grid it'll be working with
  int rSize, cSize;
  if(rank%rProcs == rProcs-1) {
    cSize = cSubLast;
  } else {
    cSize = cSubSize;
  }
  if(rank%cProcs == cProcs-1) {
    rSize = rSubLast;
  } else {
    rSize = rSubSize;
  }
  
  //each process determines its neighbors
  int left = (rank/rProcs)*rProcs + (rank + rProcs - 1)%rProcs;
  int right = (rank/rProcs)*rProcs + (rank + 1)%rProcs;
  int above = ((rank/rProcs - 1)*rProcs + rank%rProcs + size)%size;
  int below = ((rank/rProcs +1)*rProcs + rank%rProcs)%size;

  //process 0 sends each process the contents of the initial grid its working with
  vector<square_t> myGridA((rSize+2)*(cSize+2), 0), myGridB((rSize+2)*(cSize+2), 0);
  if(rank == 0) {
    for(int i = 0; i < cProcs; i++) {
      int cGridSize = (i != cProcs-1 ? rSubSize : cSubLast);
      
      for(int j = 1/(i+1); j < rProcs; j++) {
        int rGridSize = (j != rProcs-1 ? rSubSize : rSubLast);
        vector<square_t> outbound((rGridSize+2)*(cGridSize+2), 0);
        
        for(int k = 0; k < cGridSize; k++) {
          for(int l = 0; k < rGridSize; l++) {
            if((i == 0 && k == 0) || (i == rProcs-1 && k == rGridSize-1)) {
            outbound[(k+1)*rGridSize+(l+1)] = gridA[(i*cSubSize+k)*rows+(j*rSubSize+l)];
            }
          }
        }
        MPI_Request request;
        MPI_Isend(outbound.data(), (rSubsize+1)*(cSubsize+1)*sizeof(square_t), 
                  MPI_BYTE, i*rProcs + j, 0, MPI_COMM_WORLD, &request);
      }
    }
    for(int k = 0; k < cGridSize+2; k++) {
      for(int l = 0; k < rGridSize+2; l++) {
        outbound[(k+1)*rGridSize+(l+1)] = gridA[i*rows+j];
      }
    }
  } else {
    MPI_Recv(gridA.data(), (rSize+2)*(cSize+2)*sizeof(square_t), MPI_BYTE,
             0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  //move each grid through time
  for(long i = 0; i < numSteps; i++) {
    step(&myGridA, &myGridB, rSize, cSize, left, right, above, below);
    square_t* temp = gridA;
    gridA = gridB;
    gridB = temp;
  }

  //collect all the final grids and paste them into the output grid
  if(rank == 0) {
    for(int i = 0; i < cProcs; i++) {
      int cGridSize = (i != cProcs-1 ? rSubSize : cSubLast);
      
      for(int j = 1/(i+1); j < rProcs; j++) {
        int rGridSize = (j != rProcs-1 ? rSubSize : rSubLast);
        vector<square_t> inbound((rGridSize+2)*(cGridSize+2), 0);
        
        MPI_Recv(inbound.data(), (rSubsize+1)*(cSubsize+1)*sizeof(square_t), 
                  MPI_BYTE, i*rProcs + j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for(int k = 0; k < cGridSize; k++) {
          for(int l = 0; k < rGridSize; l++) {
            if((i == 0 && k == 0) || (i == rProcs-1 && k == rGridSize-1)) {
              gridB[(i*cSubSize+k)*rows+(j*rSubSize+l)] = outbound[(k+1)*rGridSize+(l+1)];
            }
          }
        }
      }
    }
    for(int k = 0; k < cGridSize+2; k++) {
      for(int l = 0; k < rGridSize+2; l++) {
        outbound[(k+1)*rGridSize+(l+1)] = gridA[i*rows+j];
      }
    }
  } else {
    MPI_Request request;
    MPI_Isend(gridA.data(), (rSize+2)*(cSize+2)*sizeof(square_t), MPI_BYTE,
             0, 0, MPI_COMM_WORLD, &request);
  }
}

#endif //_GOFORTH_H
