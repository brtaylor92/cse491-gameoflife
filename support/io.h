#ifndef _PRINTGRID_H
#define _PRINTGRID_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef gui
  #include <SDL2/SDL.h>
  
  #include "gui.h"

  #define W_HEIGHT 800
  #define W_WIDTH 800

  SDL_Window *window;
  SDL_Renderer *renderer;
#endif

#define grid(i, j) grid[(j) + (i) * rows]

void readGrid(square_t *grid, const long rows, const long cols, FILE *fp) {
  for(long i = 0; i < rows; i++) {
    for(long j = 0; j < cols; j++) {
      fscanf(fp, "%hhu", &grid(i, j));
    }
  }
}

#ifdef gui
  void initGUI() {
    if(SDL_Init(SDL_INIT_EVERYTHING) != 0) {
      logSDLError("SDL_Init");
    }

    window = SDL_CreateWindow("Game of Life GUI", 100, 100, W_WIDTH,
                              W_HEIGHT, SDL_WINDOW_SHOWN
                             );
    if(!window) {
      logSDLError("CreateWindow");
    }

    renderer = SDL_CreateRenderer(window, -1,
                                  SDL_RENDERER_ACCELERATED | 
                                  SDL_RENDERER_PRESENTVSYNC
                                 );
    if(!renderer) {
      logSDLError("CreateRenderer");
    }
  }
#endif

void printGrid(square_t *grid, const long rows, const long cols) {
  for(long i = 0; i < rows; i++) {
    for(long j = 0; j < cols; j++) {
      printf("%hhu ", grid(i,j));
    }
    printf("\n");
  }
  printf("\n");
  #ifdef gui
    drawGrid(grid, rows, cols, W_WIDTH, W_HEIGHT, renderer);
    SDL_RenderPresent(renderer);
    SDL_Delay(250);
  #endif
}

#endif //_PRINTGRID_H
