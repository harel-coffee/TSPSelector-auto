g++ -o ../bin/GA-EAX -O3 main.cpp environment.cpp cross.cpp evaluator.cpp indi.cpp randomize.cpp kopt.cpp sort.cpp -lm

./GA-EAX tsp_file NPOP NCH optimum tmax seed

default values
NPOP: 100
NCH: 30