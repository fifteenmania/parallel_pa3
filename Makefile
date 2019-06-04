CXX = g++
CFLAGS = -Wall -O3
LDLIBS = -fopenmp
RM = rm -f
UTILS = utils.cpp utils.hpp

.SUFFIXES : .cpp

all : P1 P2 P3

all_bench : P1_bench P2 P3

P1 : $(UTILS) P1.cu
	nvcc -arch sm_52 -Xptxas -O3,-v --link utils.cpp P1.cu -o P1

P2 : $(UTILS) P2.cpp
	$(CXX) $(CFLAGS) utils.cpp P2.cpp -o P2 $(LDLIBS)

P3 : mmreader.hpp mmreader.cpp $(UTILS) P3.cpp
	$(CXX) $(CFLAGS) mmreader.cpp utils.cpp P3.cpp -o P3 $(LDLIBS) -std=c++11

P1_bench : $(UTILS) P1.cu
	$(CXX) -DBENCH=1 $(CFLAGS) utils.cpp P1.cpp -o P1 $(LDLIBS)

test_all : testset1 testset2 testset3

testset1 : test1 test2 test3 test4 test5

testset2 : test6 test7 test8 test9 test10

testset3 : test11 test12 test13 test14 test15

testset4 : test16 test17 test18 test19 test20

test1 :
	./P3 ./matrix/2cubes_sphere.mtx 2048
test2 : 
	./P3 ./matrix/cage12.mtx 1024
test3 : 
	./P3 ./matrix/consph.mtx 2048
test4 :
	./P3 ./matrix/cop20k_A.mtx 2048
test5 : 
	./P3 ./matrix/filter3D.mtx 2048
test6 :
	./P3 ./matrix/hood.mtx 1024
test7 :  
	./P3 ./matrix/m133-b3.mtx 1024
test8 :
	./P3 ./matrix/mac_econ_fwd500.mtx 1024
test9 :
	./P3 ./matrix/majorbasis.mtx 1024
test10 :
	./P3 ./matrix/mario002.mtx 512
test11 :
	./P3 ./matrix/mc2depi.mtx 512
test12 :
	./P3 ./matrix/offshore.mtx 1024
test13 :
	./P3 ./matrix/patents_main.mtx 1024
test14 :
	./P3 ./matrix/pdb1HYS.mtx 4096
test15 :
	./P3 ./matrix/poisson3Da.mtx 16384
test16 :
	./P3 ./matrix/pwtk.mtx 1024
test17 :
	./P3 ./matrix/rma10.mtx 4096
test18 :
	./P3 ./matrix/scircuit.mtx 1024
test19 :
	./P3 ./matrix/shipsec1.mtx 1024
test20 : 
	./P3 ./matrix/webbase-1M.mtx 256

clean : 
	$(RM) P1 P2 P3
