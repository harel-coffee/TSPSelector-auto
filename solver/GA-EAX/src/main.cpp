/*
 * main.cpp
 *   created on: April 24, 2013
 * last updated: June 13, 2013
 *       author: liushujia
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
	int maxNumOfTrial=1000;

	TEnvironment* gEnv = new TEnvironment();
	gEnv->fFileNameTSP=(char*)malloc(100);
	// ./GA-EAX tsp_file NPOP NCH optimum tmax
	// default: 100, 30, -1(unknown optimum), 3600
	if (argc != 6)
	{
		cout << "./GA-EAX tsp_file NPOP NCH optimum tmax";
		exit(-1);
	}
	gEnv->fFileNameTSP = argv[1];
	gEnv->Npop = atoi(argv[2]);
	gEnv->Nch = atoi(argv[3]);
	gEnv->optimum = atoi(argv[4]);
	gEnv->tmax = atoi(argv[5]);

	cout << "Initialization ..." << endl;
	gEnv->define();
	gEnv->doIt();
	// gEnv->writeBest();
	gEnv->printOn();
	//system("pause");
	return 0;
}
