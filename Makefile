CC = gcc
CILKCC = icc
CUCC = gcc 
MPICC = mpic++
FLAGS = -Wall -Wextra -Werror -pedantic-errors -O2
CCFLAGS = -std=c99 $(FLAGS)
CPPFLAGS = -std=c++11 $(FLAGS)
LDLIBS = 
SDLLIBS = -I/Library/Frameworks/SDL.framework/Headers
SDL = -framework SDL2
EXECNAME = gol

.PHONY: basic gui omp cilk cuda mpigol test clean

basic: main.c support/*.h
	$(CC) $(CCFLAGS) $(LDLIBS) -D$@ $< -o $(EXECNAME)

gui: main.c support/*.h support/gui.h
	$(CC) $(CCFLAGS) $(LDLIBS) $(SDL) -D$@ -Dbasic $< -o $(EXECNAME)

omp: main.c support/*.h omp/step.h
	$(CC) $(CCFLAGS) $(LDLIBS) -D$@ $< -o $(EXECNAME) -fopenmp

cilk: main.c support/*.h cilk/step.h
	$(CILKCC) $(CCFLAGS) $(LDLIBS) -D$@ $< -o $(EXECNAME)

cuda: main.c support/*.h cuda/step.h cuda/goForth.h
	$(CUCC) $(CCFLAGS) $(LDLIBS) -D$@ $< -o $(EXECNAME)

mpigol: mpi/main.cpp support/*.h mpi/step.h mpi/goForth.h
	$(MPICC) $(CPPFLAGS) $(LDLIBS) -D$@ $< -o $(EXECNAME)

test:
	make basic
	make gui
	make omp
	make cilk
	make cuda
	make mpigol
	make clean

clean:
	rm -rf gol
