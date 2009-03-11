
all : fast

fast : fast.cpp
	g++ -O3 -march=core2 -o $@ $^

.PHONY: run
run : fast
	time ./fast
