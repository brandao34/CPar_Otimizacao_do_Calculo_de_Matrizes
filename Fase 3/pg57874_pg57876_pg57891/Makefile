CPP = nvcc

SRCS = src/main.cu src/fluid_solver.cu src/EventManager.cpp

FLAGS =  -O2 -g -std=c++11 -arch=sm_35 -use_fast_math  -Wno-deprecated-gpu-targets

FLAGS_PROFILE = -pg -g -fno-omit-frame-pointer



TARGET = fluid_sim

SCRIPT = run.sh

all:
	@echo Building Release Version
	$(CPP) $(SRCS) $(FLAGS) -o $(TARGET)
	@echo "Build complete."

clean:
	@echo Cleaning up...
	@rm  $(TARGET)
	@echo Done.

run: 
	@echo "Enviando o job para o cluster..."
	sbatch $(SCRIPT)

