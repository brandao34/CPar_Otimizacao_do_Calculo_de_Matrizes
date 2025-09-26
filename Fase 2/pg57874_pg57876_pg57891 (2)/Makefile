CPP = g++ -std=gnu++11
SRCS = src/main.cpp src/fluid_solver.cpp src/EventManager.cpp
FLAGS =   -Wall -pg -fno-omit-frame-pointer -g -Ofast -funroll-loops -ftree-vectorize  -fopenmp
FLAGS_seq =  -fno-omit-frame-pointer -g -Ofast -funroll-loops -ftree-vectorize  


all: phase2

phase2:
	$(CPP) $(FLAGS) $(SRCS) -o fluid_sim
	$(CPP) $(FLAGS_seq) $(SRCS) -o fluid_sim_seq


clean:
	@echo Cleaning up...
	@rm fluid_sim_seq
	@rm fluid_sim
	@rm gmon.out
	@echo Done.

runseq:
	@echo Running sequential version...
	@export OMP_NUM_THREADS=1
	@./fluid_sim_seq 
	@echo Done.

runpar:
	@echo Running parallel version...
	@export OMP_NUM_THREADS=16
	@./fluid_sim
	@echo Done.

