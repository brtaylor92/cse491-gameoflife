#ifndef _STEP_H
#define _STEP_H

// For ease of use
#define grid(i, j) grid[(j) + (i) * cols]
#define newGrid(i, j) newGrid[(j) + (i) * cols]

__global__ void step(square_t *grid, square_t *newGrid, 
                     const long rows, const long cols) {
  const int j = threadIdx.x;
  const int i = threadIdx.y;
  // (0,0) => 0
  // (0,1) => 1
  // (0,2) => 2
  // (0,3) => 3
  // (0,4) => 4
  // (0,5) => 5
  // (0,6) => 6
  // (0,7) => 7
  // (0,8) => 8
  // (1,0) => 8
  // (1,1) => 9
  // (1,2) => 10
  // (1,3) => 11
  // (1,4) => 12
  // (1,5) => 13
  // (1,6) => 14
  // (1,7) => 15
  // (1,8) => 16
  const unsigned char rules[16] = {0,0,0,1,0,0,0,0,0,0,1,1,0,0,0,0};
  square_t neighbors = grid((i - 1 + rows) % rows,  j                    )
                     + grid((i - 1 + rows) % rows, (j - 1  + cols) % cols)
                     + grid((i - 1 + rows) % rows, (j + 1) % cols        )
                     + grid(i                    , (j - 1  + cols) % cols)
                     + grid(i                    , (j + 1) % cols        )
                     + grid((i + 1) % rows       ,  j                    )
                     + grid((i + 1) % rows       , (j - 1 + cols) % cols )
                     + grid((i + 1) % rows       , (j + 1) % cols        );
  newGrid(i,j) = rules[(grid(i,j) << 3) | neighbors];
  square_t *temp = grid;
  grid = newGrid;
  newGrid = temp;
}
	
#endif //_STEP_H