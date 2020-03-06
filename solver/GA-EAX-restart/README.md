GA-EAX with restart mechanism

To make GA-EAX-restart
cd src
g++ -o GA-EAX-restart -O3 main.cpp environment.cpp cross.cpp evaluator.cpp indi.cpp randomize.cpp kopt.cpp sort.cpp -lm

To call GA-EAX

./GA-EAX tsp_file NPOP NCH optimum tmax