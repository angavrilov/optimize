#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <sys/timeb.h>

#ifdef __SSE2__
#include <xmmintrin.h>
#include <emmintrin.h>
#else
#ifndef NO_SSE
#error Please enable SSE2
#endif
#endif
 
#define L1_CACHE 32768
#define L2_CACHE (2*1048576)
#define TLB_SIZE 256
#define VALS_PER_LINE 8
#define VALS_PER_PAGE 512
#define MIN(a,b) (((a)<(b))?(a):(b))

#ifdef __GNUC__
#define ALIGN16(x) x __attribute__((aligned(16)))
#define _aligned_alloc(size,alignment) memalign(alignment,size)
#define _aligned_free(ptr) free(ptr)
#else
#define ALIGN16(x) __declspec(align(16)) x
#endif

double GetResult(double * LeftMatrix, double * RightMatrix, int N, int L, int M)
{
	// матрица LeftMatrix хранится по строкам
	// матрица RighttMatrix хранится по строкам
	// L ― число столбцов LeftMatrix и число строк RighttMatrix
	// N ― число строк LeftMatrix
	// M ― число столбцов RighttMatrix
	// Возвращаемый результат ― сумма всех элементов произведения LeftMatrix на RightMatrix слева направо

	int i=0;
	int j=0;
	int k=0;
	double Result=0.0;
	ALIGN16(double right[VALS_PER_PAGE]);
	
	int jstride = VALS_PER_PAGE;
	int istride = MIN(L2_CACHE*3/sizeof(double)/(jstride+L)/4, TLB_SIZE*3/4/2);

	if (istride > N) istride = N;
	if (jstride > M) jstride = M;
	
	double *sums = (double*)_aligned_alloc(istride*jstride*sizeof(double),16);
		
	{
		{
			for(int i0=0;i0<N;i0+=istride)
			{
				int itop = MIN(istride,N-i0);
				for(int j0=0;j0<M;j0+=jstride)
				{
					int jtop = MIN(jstride,M-j0);
					int jstep = (jtop+1)&~1;

					for(i=0;i<itop*jstep;i++)
						sums[i] = 0.0;
					for(k=0;k<L;k++)
					{
						double *pright = RightMatrix + k*M + j0;
						for(j=0;j<jtop;j++)
							right[j] = pright[j];
#ifdef __SSE2__						
						if (j < jstep)
							right[j] = 0.0;
#endif
						for(i=0;i<itop;i++)
						{
							double left = LeftMatrix[(i+i0)*L+k];
							double *psums = sums + i*jstep;
#ifdef __SSE2__
							__m128d left2 = _mm_set1_pd(left);

#define 					COMPUTE(d) _mm_store_pd(psums+j+d,\
									 _mm_add_pd(_mm_load_pd(psums+j+d),\
									 _mm_mul_pd(left2, _mm_load_pd(right+j+d))))
							for(j=0;j<jstep-14;j+=16) {
								COMPUTE(0);
								COMPUTE(2);
								COMPUTE(4);
								COMPUTE(6);
								COMPUTE(8);
								COMPUTE(10);
								COMPUTE(12);
								COMPUTE(14);
							}
							for(;j<jstep;j+=2)
								COMPUTE(0);
#else
							for(j=0;j<jtop;j++)
								psums[j] += left*right[j];
#endif
						}
					}
					for(i=0;i<itop*jstep;i++)
						Result += fabs(sums[i]);
				}
			}
		}
	}
	
	_aligned_free(sums);
	return(Result);
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
