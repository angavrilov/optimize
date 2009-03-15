#include <stdio.h>
#include <stdlib.h>
#include <sys/timeb.h>

#if defined(_M_IX86_FP) && _M_IX86_FP >= 2
#define __SSE2__
#endif

#ifdef __SSE2__
#include <emmintrin.h>
#else
#ifndef NO_SSE
#error Please enable SSE2
#endif
#endif

#define KSTRIDE 512
#define MIN(a,b) (((a)<(b))?(a):(b))

#ifdef __SSE2__
#define SUM_LOOP(idx,limit,target,ptr,expr,exprsse) \
	{ \
		double temp[2]; \
		__m128d _s1 = _mm_set1_pd(0.0); \
		__m128d _s2 = _mm_set1_pd(0.0); \
		__m128d _s3 = _mm_set1_pd(0.0); \
		__m128d _s4 = _mm_set1_pd(0.0); \
		int _l2 = limit & ~1; \
		if (((long)ptr)&0xF) { \
			for(idx=0;idx<_l2-6;idx+=8) { \
				_s1 = _mm_add_pd(_s1, exprsse(_mm_loadu_pd,0)); \
				_s2 = _mm_add_pd(_s2, exprsse(_mm_loadu_pd,2)); \
				_s3 = _mm_add_pd(_s3, exprsse(_mm_loadu_pd,4)); \
				_s4 = _mm_add_pd(_s4, exprsse(_mm_loadu_pd,6)); \
			} \
			for(;idx<_l2;idx+=2) \
				_s1 = _mm_add_pd(_s1, exprsse(_mm_loadu_pd,0)); \
		} else { \
			for(idx=0;idx<_l2-6;idx+=8) { \
				_s1 = _mm_add_pd(_s1, exprsse(_mm_load_pd,0)); \
				_s2 = _mm_add_pd(_s2, exprsse(_mm_load_pd,2)); \
				_s3 = _mm_add_pd(_s3, exprsse(_mm_load_pd,4)); \
				_s4 = _mm_add_pd(_s4, exprsse(_mm_load_pd,6)); \
			} \
			for(;idx<_l2;idx+=2) \
				_s1 = _mm_add_pd(_s1, exprsse(_mm_load_pd,0)); \
		} \
		_mm_storeu_pd(temp, _mm_add_pd(_mm_add_pd(_s1,_s2),_mm_add_pd(_s3,_s4))); \
		target += temp[0]+temp[1]; \
		if (limit & 1) { \
			idx = limit-1; \
			target += expr; \
		} \
	}
#else
#define SUM_LOOP(idx,limit,target,ptr,expr,exprsse) \
	for (idx=0;idx<limit;idx++) target += expr;
#endif

double GetResult(double * LeftMatrix, double * RightMatrix, int N, int L, int M)
{
	// матрица LeftMatrix хранится по строкам
	// матрица RighttMatrix хранится по строкам
	// L ― число столбцов LeftMatrix и число строк RighttMatrix
	// N ― число строк LeftMatrix
	// M ― число столбцов RighttMatrix
	// Возвращаемый результат ― сумма всех элементов произведения LeftMatrix на RightMatrix слева направо

	int i,j,k,k0,ktop;
	double sum=0.0;
	double jrows[KSTRIDE];

	for(k0=0;k0<L;k0+=KSTRIDE) {
		ktop = MIN(KSTRIDE,L-k0);

		for(k=0;k<ktop;k++) {
			double jsum = 0.0;
			double *pright = RightMatrix + (k0+k)*M;
#define SSEXPR1(load,off) load(pright+j+off)
			SUM_LOOP(j,M,jsum,pright,pright[j],SSEXPR1)
			jrows[k]=jsum;
		}

		for(i=0;i<N;i++)
		{
			double *pleft = LeftMatrix + i*L + k0;
#define SSEXPR2(load,off) _mm_mul_pd(load(pleft+k+off),_mm_load_pd(jrows+k+off))
			SUM_LOOP(k,ktop,sum,pleft,pleft[k]*jrows[k],SSEXPR2)
		}
	}

	return sum;
}

int main(int argc, char* argv[])
{
	struct timeb tstartstruct,tstopstruct;

	int N1=2200;
	int M1=2300;
	int L1=2400;
	int i=0;
	double res=0;
	int seconds=0;
	int ms=0;

	double Range=1.0e+15;
	double * LM=new double[N1*L1];
	double * RM=new double[L1*M1];

	for(i=0;i<N1*L1;i++)
	{
		LM[i]=(2.0*Range*rand())/(1.0*RAND_MAX)-Range;
	}

	for(i=0;i<M1*L1;i++)
	{
		RM[i]=(2.0*Range*rand())/(1.0*RAND_MAX)-Range;
	}

	// Petrov Test
	ftime( &tstartstruct );

	res=GetResult(LM,RM,N1,L1,M1);

	ftime( &tstopstruct );

	seconds=tstopstruct.time-tstartstruct.time;
	ms=tstopstruct.millitm-tstartstruct.millitm;

	if(ms<0)
	{
		seconds--;
		ms+=1000;
	}

	if(ms >= 1000) {
		seconds += ms/1000;
		ms = ms % 1000;
	}

	printf(" Result is %e \n Work time is %i.%03i seconds \n",res,seconds,ms);
	// End Petrov Test

	delete[] LM;
	delete[] RM;
	return 0;
} 
