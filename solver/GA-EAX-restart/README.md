GA-EAX with restart mechanism

To make GA-EAX-restart
cd src
g++ -o ../bin/GA-EAX-restart -O3 main.cpp environment.cpp cross.cpp evaluator.cpp indi.cpp randomize.cpp kopt.cpp sort.cpp -lm

To call GA-EAX

./GA-EAX-restart tsp_file NPOP NCH optimum tmax seed

default values
NPOP: 100
NCH: 30