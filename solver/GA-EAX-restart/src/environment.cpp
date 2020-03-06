/*
 * environment.cpp
 *   created on: April 24, 2013
 * last updated: June 13, 2013
 *       author: liushujia
 */

#ifndef __ENVIRONMENT__
#include "environment.h"
#endif

#include <math.h> 

extern int gBestValue;
extern TIndi gBest;

void MakeRandSol(TEvaluator* eval, TIndi& indi);
void Make2optSol(TEvaluator* eval, TIndi& indi);

TEnvironment::TEnvironment(){
	fEvaluator = new TEvaluator();
}

TEnvironment::~TEnvironment(){
	delete [] fIndexForMating;
	delete [] tCurPop;
	delete fEvaluator;
	delete tCross;

	int N = fEvaluator->Ncity;
	for( int i = 0; i < N; ++i ) delete [] fEdgeFreq[ i ];
	delete [] fEdgeFreq;
}

void TEnvironment::define()
{
	fEvaluator->setInstance( fFileNameTSP );
	gBestValue = -1;
	gBest.define(fEvaluator->Ncity);
	// int N = fEvaluator->Ncity;
	// fIndexForMating = new int [ Npop + 1 ];  
	// tCurPop = new TIndi [ Npop ];
	// for ( int i = 0; i < Npop; ++i ) tCurPop[i].define( N );

	// tBest.define( N );
	// tCross = new TCross( N );
	// tCross->eval = fEvaluator;                 
	// tCross->Npop = Npop;             

	// tKopt = new TKopt( N );
	// tKopt->eval = fEvaluator;
	// tKopt->setInvNearList();
	// fEdgeFreq = new int* [ N ]; 
	// for( int i = 0; i < N; ++i ) fEdgeFreq[ i ] = new int [ N ]; 
}

void TEnvironment::define_for_restart()
{
	int N = fEvaluator->Ncity;
	fIndexForMating = new int[Npop + 1];
	tCurPop = new TIndi[Npop];
	for (int i = 0; i < Npop; ++i)
		tCurPop[i].define(N);

	tBest.define(N);
	tCross = new TCross(N);
	tCross->eval = fEvaluator;
	tCross->Npop = Npop;

	tKopt = new TKopt(N);
	tKopt->eval = fEvaluator;
	tKopt->setInvNearList();
	fEdgeFreq = new int *[N];
	for (int i = 0; i < N; ++i) fEdgeFreq[i] = new int[N];
}

void TEnvironment::doIt()
{
	this->define();
	this->fTimeStart = clock();
	this->fTimeEnd = this->fTimeStart;
	bool flag = false; // find optimum
	int trial = 0; // num of trial

	while((int)((double)(this->fTimeEnd - this->fTimeStart)/(double)CLOCKS_PER_SEC) < tmax)
	{
		trial ++;
		this->define_for_restart();
		this->initPop();
		this->init();
		this->getEdgeFreq();
		while (1)
		{
			// each generation of GA-EAX
			this->setAverageBest();
			if(gBestValue == -1 || fBestValue < gBestValue)
			{
				gBestValue = fBestValue;
				gBest = tBest;
				printf("Trial %d, find better solution %d\n", trial, gBestValue);
				if(gBestValue<=optimum)
				{
					flag = true;
					printf("Trial %d, find optimum solution %d, exit\n", trial, gBestValue);
					break;
				}
			}
			if (fCurNumOfGen % 100 == 0)
			{
				printf("Trial %-5d Gen %-5d:\t%-10d\t%-.2f\n", trial, fCurNumOfGen, fBestValue, fAverageValue);
				this->fTimeEnd = clock();
				if((int)((double)(this->fTimeEnd - this->fTimeStart)/(double)CLOCKS_PER_SEC) >= tmax)
				{
					break;
				}
			}

			if (this->terminationCondition())
			{
				this->fTimeEnd = clock();
				break;
			}

			this->selectForMating();
			for (int s = 0; s < Npop; ++s)
				this->generateKids(s);

			++fCurNumOfGen;
		}

		if(flag) break;
	}
	this->fTimeEnd = clock();
}

void TEnvironment::init(){
	fAccumurateNumCh = 0;
	fCurNumOfGen = 0;
	fStagBest = 0;
	fMaxStagBest = 0;
	fStage = 1;				// 阶段1
	fFlagC[ 0 ] = 4;		// 维持种群多样性	1:Greedy, 2:---, 3:Distance, 4:Entropy
	fFlagC[ 1 ] = 1;		// Eset类型			1:Single-AB, 2:Block2
} 

bool TEnvironment::terminationCondition(){
	if ( fAverageValue - fBestValue < 0.001 )  return true;
	if( fStage == 1 ){
		if( fStagBest == int(1500/Nch) && fMaxStagBest == 0 )			// 1500/Nch
			fMaxStagBest =int( fCurNumOfGen / 10 );						// fMaxStagBest = G/10
		else if( fMaxStagBest != 0 && fMaxStagBest <= fStagBest ){ 
			fStagBest = 0;
			fMaxStagBest = 0;
			fCurNumOfGen1 = fCurNumOfGen;
			fFlagC[ 1 ] = 2; 
			fStage = 2;      
		}
		return false;
	}
	if( fStage == 2 ){
		if( fStagBest == int(1500/Nch) && fMaxStagBest == 0 )			// 1500/Nch
			fMaxStagBest = int( (fCurNumOfGen - fCurNumOfGen1) / 10 );	// fMaxStagBest = G/10
		else if( fMaxStagBest != 0 && fMaxStagBest <= fStagBest ) return true;
		return false;
	}

	return true;
}

void TEnvironment::setAverageBest(){
	int stockBest = tBest.fEvaluationValue;
	fAverageValue = 0.0;
	fBestIndex = 0;
	fBestValue = tCurPop[0].fEvaluationValue;
	for(int i = 0; i < Npop; ++i ){
		fAverageValue += tCurPop[i].fEvaluationValue;
		if( tCurPop[i].fEvaluationValue < fBestValue ){
			fBestIndex = i;
			fBestValue = tCurPop[i].fEvaluationValue;
		}
	}
	tBest = tCurPop[ fBestIndex ];
	fAverageValue /= (double)Npop;
	if( tBest.fEvaluationValue < stockBest ){
		fStagBest = 0;
		fBestNumOfGen = fCurNumOfGen;
		fBestAccumeratedNumCh = fAccumurateNumCh;
	}
	else ++fStagBest;
}

void TEnvironment::initPop(){
	for ( int i = 0; i < Npop; ++i ){ 
		tKopt->makeRandSol( tCurPop[ i ] );		// 设置一个随机路程
		tKopt->doIt( tCurPop[ i ] );			// 局部搜索(2-opt neighborhood) 
	}
}

void TEnvironment::selectForMating(){
	tRand->permutation( fIndexForMating, Npop, Npop ); 
	fIndexForMating[ Npop ] = fIndexForMating[ 0 ];
}

void TEnvironment::generateKids( int s ){
	// tCurPop[fIndexForMating[s]] 被替换为tCross->DoIt()中子代的一个最优解
	// fEdgeFreq[][] 同时被更新
	tCross->setParents( tCurPop[fIndexForMating[s]], tCurPop[fIndexForMating[s+1]], fFlagC, Nch );  
	tCross->doIt( tCurPop[fIndexForMating[s]], tCurPop[fIndexForMating[s+1]], Nch, 1, fFlagC, fEdgeFreq );
	fAccumurateNumCh += tCross->fNumOfGeneratedCh;
}

void TEnvironment::getEdgeFreq(){
	int  k0, k1, N = fEvaluator->Ncity;
	for( int j1 = 0; j1 < N; ++j1 )
		for( int j2 = 0; j2 < N; ++j2 )
			fEdgeFreq[ j1 ][ j2 ] = 0;

	for( int i = 0; i < Npop; ++i )
		for(int j = 0; j < N; ++j ){
			k0 = tCurPop[ i ].fLink[ j ][ 0 ];
			k1 = tCurPop[ i ].fLink[ j ][ 1 ];
			++fEdgeFreq[ j ][ k0 ];
			++fEdgeFreq[ j ][ k1 ];
		}
}

void TEnvironment::printOn()
{
	printf("Total time: %d\n ", (int)((double)(this->fTimeEnd - this->fTimeStart)/(double)CLOCKS_PER_SEC));
	printf("bestval = %d\n" , gBestValue);
	fflush(stdout);
}

void TEnvironment::writeBest(){
	FILE *fp;
	char filename[ 80 ];

	sprintf( filename, "bestSolution.txt" );
	fp = fopen( filename, "a");
	fEvaluator->writeTo( fp, gBest );
	fclose( fp );
}
