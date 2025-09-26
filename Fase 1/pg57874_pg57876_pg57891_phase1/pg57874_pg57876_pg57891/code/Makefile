CPP = g++ -std=gnu++11  -Wall -Ofast -fno-omit-frame-pointer -funroll-all-loops
SRCS = main.cpp fluid_solver.cpp EventManager.cpp

all:
	$(CPP) $(SRCS) -o fluid_sim

clean:
	@echo Cleaning up...
	@rm fluid_sim
	@echo Done.

