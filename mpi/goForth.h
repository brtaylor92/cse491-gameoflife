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
  long procRows = max(static_cast<long>(1), 
       min(rows/minSize, static_cast<long>(round(sqrt(size/aspectRatio)))));
  long procCols = size/procRows;
  if(procCols > cols/minSize) {
    procCols = max(static_cast<long>(1), rows/minSize);
    procRows = max(static_cast<long>(1), min(rows/minSize, size/procCols));
  }
 
  if(rank < procRows*procCols) {
    long tileRows = rows/procRows, tileCols = rows/procCols;
    long lastRows = rows - (procRows-1)*tileRows, lastCols = cols - (procCols-1)*tileCols;

    //each process determines the size of the grid it'll be working with
    long myRows = 0, myCols = 0;
    if(rank%procCols == procCols-1) {
      myCols = lastCols;
    } else {
      myCols = tileCols;
    }
    if(rank/procCols == procRows-1) {
      myRows = lastRows;
    } else {
      myRows = tileRows;
    }
    
    //each process determines its neighbors
    int left = (rank/procRows)*procRows + (rank + procRows - 1)%procRows;
    int right = (rank/procRows)*procRows + (rank + 1)%procRows;
    int above = ((rank/procRows - 1)*procRows + rank%procRows + size)%size;
    int below = ((rank/procRows +1)*procRows + rank%procRows)%size;

    //process 0 sends each process the contents of the initial grid its working with
    vector<square_t> myGridA((myRows+2)*(myCols+2), 0), myGridB((myRows+2)*(myCols+2), 0);
    if(rank == 0) {
      cout << "procRows: " << procRows << "\tprocCols: " << procCols << endl
           << "tileRows: " << tileRows << "\ttileCols: " << tileCols << endl
           << "lastRows: " << lastRows << "\tlastCols: " << lastCols << endl
           << "procs used: " << procRows*procCols << " (" << 100.0*procRows*procCols/size << "%)" << endl;

      for(long i = 0; i < procRows; i++) {
        long thisTileRows = (i != procRows-1 ? tileRows : lastRows);
        
        for(long j = 1/(i+1); j < procCols; j++) {
          long thisTileCols = (j != procCols-1 ? tileCols : lastCols);
          vector<square_t> outbound((thisTileRows+2)*(thisTileCols+2), 0);
         
          for(long k = 0; k < thisTileRows; k++) {
            for(long l = 0; l < thisTileCols; l++) {
              outbound[(k+1)*(thisTileCols+2)+(l+1)] = gridA[(i*tileCols+k)*rows+(j*tileRows+l)];
            }
          }
          MPI_Request request;
          MPI_Isend(outbound.data(), (thisTileRows+2)*(thisTileCols+2)*sizeof(square_t), 
                    MPI_BYTE, i*procCols + j, 0, MPI_COMM_WORLD, &request);
        }
      }
      for(long i = 0; i < myRows; i++) {
        for(long j = 0; j < myCols; j++) {
          myGridA[(i+1)*(myCols+2)+(j+1)] = gridA[i*rows+j];
        }
      }
    } else if(rank < procRows*procCols) {
      MPI_Recv(myGridA.data(), (myRows+2)*(myCols+2)*sizeof(square_t), MPI_BYTE,
               0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    //print what you just got
    /*int baton = 0;
    if(rank != 0 && rank < procRows*procCols) {
      MPI_Recv(&baton, 1, MPI_INT, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    cout << endl << "tile " << rank << " (" << myRows << " rows, " << myCols << " cols):" << endl;
    for(long i = 0; i < myRows; i++) {
      for(long j = 0; j < myCols; j++) { 
        cout << int(myGridA[(i+1)*(myCols+2)+(j+1)]) << " ";
      }
      cout << endl;
    }
    if(rank < procRows*procCols-1) {
      MPI_Request request;
      MPI_Isend(&baton, 1, MPI_INT, rank+1, 0, MPI_COMM_WORLD, &request);
    }*/

    //move to the future
    for(long i = 0; i < numSteps; i++) {
      step(myGridA.data(), myGridB.data(), myRows, myCols, left, right, above, below);
      //(myGridA.data())[(myCols+2)+1] = 3;
      //myGridA.swap(myGridB);
    }
        
    //return your grid to the master process for final return
    if(rank == 0) {
      for(long i = 0; i < procRows; i++) {
        long thisTileRows = (i != procRows-1 ? tileRows : lastRows);
        
        for(long j = 1/(i+1); j < procCols; j++) {
          long thisTileCols = (j != procCols-1 ? tileCols : lastCols);
          vector<square_t> inbound((thisTileRows+2)*(thisTileCols+2), 0);
          
          MPI_Recv(inbound.data(), (thisTileRows+2)*(thisTileCols+2)*sizeof(square_t), MPI_BYTE,
                   i*procCols+j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          
          for(long k = 0; k < thisTileRows; k++) {
            for(long l = 0; l < thisTileCols; l++) {
              gridB[(i*tileCols+k)*rows+(j*tileRows+l)] = inbound[(k+1)*(thisTileCols+2)+(l+1)];
            }
          }
        }
      }
      for(long i = 0; i < myRows; i++) {
        for(long j = 0; j < myCols; j++) {
          gridB[i*rows+j] = myGridA[(i+1)*(myCols+2)+(j+1)];
        }
      }
    } else if(rank < procRows*procCols) {
      MPI_Request request;
      MPI_Isend(myGridA.data(), (myRows+2)*(myCols+2)*sizeof(square_t), 
                MPI_BYTE, 0, 0, MPI_COMM_WORLD, &request); 
    }

    if(rank == 0) {
      cout << "\nfinal grid:" << endl;
      for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols;  j++) {
          cout << int(gridB[i*cols+j]) << " ";
        }
        cout << endl;
      }
      cout << endl;
    }
  }
  (void) numSteps;
  }
#endif //_GOFORTH_H
