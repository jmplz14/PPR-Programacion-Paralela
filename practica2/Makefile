
all: floyd_par2d

floyd_par2d: floyd_par2d.cc Graph.cc Graph.h
	mpicxx -I./includes  -O3      floyd_par2d.cc Graph.cc -o floyd_par2d

clean:
	rm -f floyd_seq floyd_par1d  *.o.

