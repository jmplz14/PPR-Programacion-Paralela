#include "stdio.h"
#include <time.h>


#define NBLOCK 64
const int N=1000000;




__global__ void reduceSum(double *d_V, int N)
{
	extern __shared__ double sdata[NBLOCK];
	int tid = threadIdx.x;
	int i = blockIdx.x * (blockDim.x*2) + threadIdx.x;
	double suma = ((i < N) ? d_V[i] : 0);

	if (i + blockDim.x < N)
		suma += d_V[i+blockDim.x];
  
	sdata[tid] = suma; 

	__syncthreads();

	for (int s=blockDim.x/2; s>0; s>>=1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	if (tid == 0) d_V[blockIdx.x] = sdata[0];
}

__global__ void aproximarPi( double step, double *valores, int N)
{

int idHebra = blockIdx.x * blockDim.x + threadIdx.x;
if(idHebra < N)
{
	double x = (idHebra + 1 - 0.5) * step;
	valores[idHebra] = 4.0 / (1.0 + x * x);
}
}

int main()
{
/* pointers to host memory */
/* Allocate arrays A, B and C on host*/
double * valores = (double*) malloc(N*sizeof(double));


/* pointers to device memory */
double *valores_d;
/* Allocate arrays a_d, b_d and c_d on device*/
cudaMalloc ((void **) &valores_d, sizeof(double)*N);



clock_t begin_time = clock();

double step = 1.0 / (double)N;
int NumBlock = ceil((float)N/NBLOCK);
aproximarPi <<<NumBlock, NBLOCK>>> (step, valores_d, N);

reduceSum <<<NumBlock, NBLOCK/2>>> (valores_d, N);

/* Copy data from deveice memory to host memory */
cudaMemcpy(valores, valores_d, sizeof(double)*N, cudaMemcpyDeviceToHost);



/* Print c */
double suma = 0;
for (int i=0; i<NumBlock;i++)
	suma += valores[i];
double Tgpu = float(clock() - begin_time) / CLOCKS_PER_SEC;
printf("--------Datos Paralelos--------------\n");
printf(" El tiempo consumido es de %f segundos\n", Tgpu);

double pi = suma *step;
printf("El valor de pi es %f\n", pi);

begin_time = clock();

double suma2 = 0;
for (int i=1; i<=N; i++){
	double x = (i-0.5)*step;
	suma2 += 4.0 / ( 1.0 + x * x);	
	/*printf("----------------------------");
	printf("%f\n", suma2);
	printf("%f\n", valores[i]);*/


}

Tgpu = float(clock() - begin_time) / CLOCKS_PER_SEC;
printf("--------Datos secunenciales--------------\n");
printf(" El tiempo consumido es de %f segundos\n", Tgpu);
printf("El valor de pi sencuencial es %f", step*suma2);
/* Free the memory */
free(valores); 
cudaFree(valores_d);


}
