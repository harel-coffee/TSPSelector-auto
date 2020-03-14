/*
 * environment.h
 *   created on: April 24, 2013
 * last updated: June 13, 2013
 *       author: liushujia
 */

#ifndef __ENVIRONMENT__
#define __ENVIRONMENT__

#ifndef __INDI__
#include "indi.h"
#endif

#ifndef __RAND__
#include "randomize.h"
#endif

#ifndef __EVALUATOR__
#include "evaluator.h"
#endif

#ifndef __Cross__
#include "cross.h"
#endif

#ifndef __KOPT__
#include "kopt.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <time.h>

class TEnvironment{
public:
	TEnvironment(); 
	~TEnvironment();

	void define();						// ��ʼ��
	void doIt();						// GA������
	void init();						// GA��ʼ��
	bool terminationCondition();		// �˳�����          
	void setAverageBest();				// ������Ⱥƽ��TSP��������Ⱥ����TSP����        

	void initPop();						// ��ʼ����Ⱥ
	void selectForMating();				// ѡ�񸸱���ĸ��                
	void generateKids( int s );			// ������ѡ���Ӵ�            
	void getEdgeFreq();					// ������Ⱥ��ÿ���ߵ�Ƶ��                    

	void printOn();				// ������
	void writeBest();					// �������TSP·��

	TEvaluator* fEvaluator;				// �߾���
	TCross* tCross;						// �߼��Ͻ���
	TKopt* tKopt;						// �ֲ�����(2-opt neighborhood)
	char *fFileNameTSP;					// TSP�ļ���
	int optimum;
	int tmax;
	int Npop;							// ��Ⱥ����                   
	int Nch;							// ÿ������(ĸ��)�������Ӵ�����                      
	TIndi* tCurPop;						// ��ǰ��Ⱥ��Ա
	TIndi tBest;						// ��ǰ��Ⱥ���Ž�
	int fCurNumOfGen;					// ��ǰ��Ⱥ����
	long int fAccumurateNumCh;			// �Ӵ��ۼ���Ŀ             

	int fBestNumOfGen;					// ��ǰ���Ž����ڵĴ���                     
	long int fBestAccumeratedNumCh;		// ��ǰ���Ž���Ӵ��ۼ���Ŀ        
	int **fEdgeFreq;					// ��Ⱥ�ı�Ƶ��
	double fAverageValue;				// ��ȺTSP·����ƽ������                  
	int fBestValue;						// ��Ⱥ���Ž��·������                        
	int fBestIndex;						// ������Ⱥ���±�

	int* fIndexForMating;				// �����б�(r[])
	int fStagBest;						// �Ӵ����Ž�û���������ۼƴ���                         
	int fFlagC[ 10 ];					// EAX��ʽ��ѡ�����                      
	int fStage;							// ��ǰ�׶�
	int fMaxStagBest;					// fStagBest==fMaxStagBestʱִ����һ�׶�                      
	int fCurNumOfGen1;					// Stage I����ʱ����Ⱥ����                     

	clock_t fTimeStart, fTimeInit, fTimeEnd;	// �������ʱ��
};

#endif
