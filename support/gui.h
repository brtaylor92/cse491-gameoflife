#ifndef SDL_H
#define SDL_H

#include <stdio.h>
#include <SDL2/SDL.h>

#define grid(i, j) grid[(i) + (j) * rows]

void logSDLError(const char *msg){
  printf("%s error: %s\n", msg, SDL_GetError());
}

void renderTexture(SDL_Texture *tex, SDL_Renderer *ren, 
                  int x, int y, int w, int h) {
  //Setup the destination rectangle to be at the position we want
  SDL_Rect dst;
  dst.x = x;
  dst.y = y;
  dst.w = w;
  dst.h = h;
  SDL_RenderCopy(ren, tex, NULL, &dst);
}

void renderTextureNoScale(SDL_Texture *tex, SDL_Renderer *ren, int x, int y) {
  int w, h;
  SDL_QueryTexture(tex, NULL, NULL, &w, &h);
  renderTexture(tex, ren, x, y, w, h);
}

void drawGrid(square_t *grid, const long rows, const long cols,
              const long width, const long height,
              SDL_Renderer *renderer) {
  const long stepw = width/cols;
  const long steph = height/rows;
  //Clear screen
  SDL_SetRenderDrawColor(renderer, 0x00, 0x00, 0x00, 0x00);
  SDL_RenderClear(renderer);

  SDL_SetRenderDrawColor(renderer, 0x00, 0xFF, 0x00, 0xFF);
  for(long i = 0; i < height; i++) {
    if(i % steph == 0 || i % 4 == steph - 1) {
      SDL_RenderDrawLine(renderer, 0, i, width, i);
    }
    for(long j = 0; j < width; j++) {
      if(j % stepw == 0 || j % 4 == stepw-1) {
        SDL_RenderDrawPoint(renderer, j, i);
      }
      else if(i/stepw < rows && j/steph < cols && grid(i/stepw, j/steph)) {
        SDL_RenderDrawPoint(renderer, j, i);
      }
    }
  }
}

#endif