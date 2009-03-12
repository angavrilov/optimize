#include <stdio.h>
#include <stdlib.h>
#include <sys/timeb.h>

#ifdef __SSE2__
#include <emmintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#define L2_CACHE (2*1048576)
#define TLB_SIZE 256
#define PAGE_SIZE 4096
#define MIN(a,b) (((a)<(b))?(a):(b))

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
	int k0,ktop;
	int leftindex=0;
	int rightindex=0;
	double sum=0.0;

#ifdef __SSE2__
	double temp[2];
	__m128d sum2 = _mm_set1_pd(0.0);
	__m128d sum3 = _mm_set1_pd(0.0);
	__m128d sum4 = _mm_set1_pd(0.0);
	int MX = (M&1) ? M : 0;
	int M2 = M & ~1;
#endif

	int kstride = MIN(L2_CACHE*3/L/sizeof(double)/4, TLB_SIZE*PAGE_SIZE*3/L/sizeof(double)/4);
	int istride = TLB_SIZE/4;

#pragma omp parallel private(i, j, k, k0, ktop, sum2, sum3, sum4, temp) reduction(+: sum)
    {
	for(k0=0;k0<L;k0+=kstride) {
		ktop = MIN(k0+kstride,L);
#ifdef _OPENMP
		for(int i0=omp_get_thread_num()*istride;i0<N;i0+=omp_get_num_threads()*istride)
#else
		for(int i0=0;i0<N;i0+=istride)
#endif
		{
			int itop = MIN(i0+istride,N);
			for(k=k0;k<ktop;k++)
			{
			    double *pright = RightMatrix + k*M;
			    for(i=i0;i<itop;i++)
			    {
				double left = LeftMatrix[i*L+k];
#ifdef __SSE2__
				__m128d left2 = _mm_set1_pd(left);
				if (((long)pright)&0xF) {
					for(j=0;j<M2;j+=6) {
						sum2 = _mm_add_pd(sum2, _mm_mul_pd(left2, _mm_loadu_pd(pright+j)));
						if ((j+2)>=M2) break;
						sum3 = _mm_add_pd(sum3, _mm_mul_pd(left2, _mm_loadu_pd(pright+j+2)));
						if ((j+4)>=M2) break;
						sum4 = _mm_add_pd(sum4, _mm_mul_pd(left2, _mm_loadu_pd(pright+j+4)));
					}
				} else {
					for(j=0;j<M2;j+=6) {
						sum2 = _mm_add_pd(sum2, _mm_mul_pd(left2, _mm_load_pd(pright+j)));
						if ((j+2)>=M2) break;
						sum3 = _mm_add_pd(sum3, _mm_mul_pd(left2, _mm_load_pd(pright+j+2)));
						if ((j+4)>=M2) break;
						sum4 = _mm_add_pd(sum4, _mm_mul_pd(left2, _mm_load_pd(pright+j+4)));
					}
				}
				if (MX)
					sum += left*pright[MX-1];
#else
				for(j=0;j<M;j++)
					sum+=left*pright[j];
#endif
			    }
			}
		}
	}

#ifdef __SSE2__
	_mm_storeu_pd(temp, _mm_add_pd(_mm_add_pd(sum2,sum3),sum4));
	sum += temp[0]+temp[1];
#endif
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
