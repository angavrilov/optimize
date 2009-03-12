#include <stdio.h>
#include <stdlib.h>
#include <sys/timeb.h>

#define KSTRIDE 512
#define MIN(a,b) (((a)<(b))?(a):(b))

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
			for(j=0;j<M;j++)
				jsum+=pright[j];
			jrows[k]=jsum;
		}

		for(i=0;i<N;i++)
		{
			double *pleft = LeftMatrix + i*L + k0;
			for(k=0;k<ktop;k++)
			{
				sum += pleft[k]*jrows[k];
			}
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
