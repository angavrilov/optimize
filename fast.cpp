#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/timeb.h>

#define L1_CACHE 32768
#define L2_CACHE (2*1048576)
#define TLB_SIZE 256
#define VALS_PER_LINE 8
#define VALS_PER_PAGE 512
#define JSTRIDE (L1_CACHE/sizeof(double)/2)
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
	double Result=0.0;
	double sums[JSTRIDE];

	int jstride = MIN(JSTRIDE,L2_CACHE/sizeof(double)/L/3);
	int istride = MIN(L2_CACHE/sizeof(double)/L/3, TLB_SIZE/2);
	
	if (jstride%VALS_PER_LINE)
		jstride += VALS_PER_LINE-(jstride%VALS_PER_LINE);
	if ((jstride%VALS_PER_PAGE) < VALS_PER_PAGE/4 && jstride > VALS_PER_PAGE)
		jstride -= jstride%VALS_PER_PAGE;
	
	{
		{
			for(int i0=0;i0<N;i0+=istride)
			{
				int itop = MIN(i0+istride,N);
				for(int j0=0;j0<M;j0+=jstride)
				{
					int jtop = MIN(jstride,M-j0);
					for(i=i0;i<itop;i++)
					{
						for(j=0;j<jtop;j++)
							sums[j] = 0.0;
						for(k=0;k<L;k++)
						{
							double left = LeftMatrix[i*L+k];
							double *pright = RightMatrix + k*M + j0;
							for(j=0;j<jtop;j++)
								sums[j] += left*pright[j];
						}
						for(j=0;j<jtop;j++)
							Result += fabs(sums[j]);
					}
				}
			}
		}
	}
	
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
