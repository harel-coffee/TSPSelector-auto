/*
 * main.cpp
 *   created on: March 6, 2020
 *       author: shengcailiu
 */

#ifndef __ENVIRONMENT__
#include "environment.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <csignal>
#include <iostream>
using namespace std;

int gBestValue; // global best value
TIndi gBest;	// global best solution

void signalHandler(int signum)
{
	cout << endl
		 << "Signal (" << signum << ") received.\n";
	cout << endl;
	printf("bestval = %d\n", gBestValue);
	fflush(stdout);

	exit(signum);
}

int main( int argc, char* argv[] )
{
	signal(SIGTERM, signalHandler);
	signal(SIGINT, signalHandler);
	InitURandom(); 
	int maxNumOfTrial;

	TEnvironment* gEnv = new TEnvironment();
	gEnv->fFileNameTSP=(char*)malloc(100);

	// ./GA-EAX-restart tsp_file NPOP NCH optimum tmax
	// default: 100, 30, -1(unknown optimum), 3600
	if(argc != 6)
	{
		cout << "./GA-EAX-restart tsp_file NPOP NCH optimum tmax";
		exit(-1);
	}
	gEnv->fFileNameTSP = argv[1];
	gEnv->Npop = atoi(argv[2]);
	gEnv->Nch = atoi(argv[3]);
	gEnv->optimum = atoi(argv[4]);
	gEnv->tmax = atoi(argv[5]);

	cout<<"Initialization ..."<<endl;

	gEnv->doIt(); 
	gEnv->printOn();
	// gEnv->writeBest();

	return 0;
}
