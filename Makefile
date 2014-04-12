CC = gcc
CILKCC = icc
CUCC = gcc 
MPICC = mpicc
CCFLAGS = -std=c99 -Wall -Wextra -Werror -pedantic-errors -O2
LDLIBS = 
SDLLIBS = -I/Library/Frameworks/SDL.framework/Headers
SDL = -framework SDL2
EXECNAME = gol

.PHONY: basic gui omp cilk cuda mpi test clean

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

mpi: main.c support/*.h mpi/step.h mpi/goForth.h
	$(MPICC) $(CCFLAGS) $(LDLIBS) -D$@ $< -o $(EXECNAME)

test:
	make basic
	make gui
	make omp
	make cilk
	make cuda
	make mpi
	make clean

clean:
	rm -rf gol
