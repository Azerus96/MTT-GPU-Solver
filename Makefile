NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++20 -arch=sm_75 --use_fast_math -Xcompiler -fopenmp -diag-suppress 177,550

all: ev_dumper preflop_gen hu_solver multiway_solver

ev_dumper: src/tools/mtt_ev_dumper.cu
	$(NVCC) $(NVCC_FLAGS) $< -o bin/$@

preflop_gen: src/preflop/preflop_grid_generator.cu
	$(NVCC) $(NVCC_FLAGS) $< -o bin/$@

hu_solver: src/postflop/honest_multistreet_solver.cu
	$(NVCC) $(NVCC_FLAGS) $< -o bin/$@

multiway_solver: src/postflop/multiway_4way_engine.cu
	$(NVCC) $(NVCC_FLAGS) $< -o bin/$@

clean:
	rm -rf bin/*
