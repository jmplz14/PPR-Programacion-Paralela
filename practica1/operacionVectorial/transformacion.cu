#include <iostream>
#include <fstream>
#include <string.h>
#include <time.h>


// CUDA runtime
//#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
//#include <helper_functions.h>
//#include <helper_cuda.h>




using namespace std;

__device__ void reduce(float * d_suma, float * sdata_suma, float * d_max, float * sdata_max, int tid, int i)
{
	
	/*int tid = threadIdx.x;
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	sdata[tid] = ((i < N) ? d_V[i] : 0.0f);
	__syncthreads();*/

	for (int s=blockDim.x/2; s>0; s>>=1) {
		if (tid < s) {
			sdata_suma[tid] += sdata_suma[tid + s];
			
			int valor = sdata_max[tid + s];
			if( sdata_max[tid] < valor)
				sdata_max[tid] = valor;
		}
		
		__syncthreads();
	}

	if (tid == 0){
	 d_suma[blockIdx.x] = sdata_suma[0];
	 d_max[blockIdx.x] = sdata_max[0];
	}
	
}

__global__ void transformacionSinCompartida(float * A, float * B, float * C, int N) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < N) {
		int K = N / i;
		int posInicio = K * blockDim.x;
		float suma = 0;
		for (int j = 0; j < blockDim.x; j++) {
			int posActual = posInicio + j;
			float valorA = A[posActual] * i;
			if ( ( (int)ceil(valorA) % 2) == 0) {
				suma += valorA + B[posActual];
			}
			else {
				suma += valorA - B[posActual];
			}
		}
		C[i] = suma;
	}
	
}

__global__ void transformacionConCompartida(float * A, float * B, float * C, float * D_suma, float * D_max, int N) {
	extern __shared__ float sdata[];
	int tid = threadIdx.x;
	int i = tid + blockDim.x * blockIdx.x;
	
	float *sdata_A = sdata; 
	float *sdata_B = sdata + blockDim.x; 
	float *sdata_suma = sdata + blockDim.x*2;
	float *sdata_max = sdata + blockDim.x*3;

	sdata_A[tid] = A[i];
	sdata_B[tid] = B[i];
	
	__syncthreads();


	if (i < N) {


		float suma = 0;
		
		for (int j = 0; j < blockDim.x; j++) {
			float valorA = sdata_A[j] * i;
			if ( (int(ceil(valorA)) % 2) == 0) {
				suma += valorA + sdata_B[j];
			}
			else {
				suma += valorA - sdata_B[j];
			}
		}
	/*int posInicio = blockIdx.x * blockDim.x;
		float suma = 0;
		
		for (int j = 0; j < blockDim.x; j++) {
		
			
			int posActual = posInicio + j;
			float valorA = sdata_A[posActual] * i;
			printf("hola");
			if ( (int(ceil(valorA)) % 2) == 0) {
				suma += valorA + *(sdata_B+posActual);
			}
			else {
				suma += valorA - *(sdata_B+posActual);
			}
			
		}*/
		
		C[i] = suma;
		sdata_suma[tid] = suma;
		sdata_max[tid] = suma;
		
		__syncthreads();

		
		reduce(D_suma, sdata_suma, D_max, sdata_max, tid, i);
		
		
		
	}

}








int main(int argc, char *argv[]) {

	int blocksize, NBlocks;
	if (argc != 3)
	{
		cout << "Uso: transformacion Num_bloques Tam_bloque  " << endl;
		return(0);
	}
	else
	{
		NBlocks = atoi(argv[1]);
		blocksize = atoi(argv[2]);
	}

	const int   N = blocksize * NBlocks;

	float *A = new float[N];
	float *B = new float[N];
	float *C = new float[N];
	float *D = new float[NBlocks];
	float *D_suma = new float[blocksize];
	float *D_max = new float[blocksize];

	int devID;
	cudaError_t err;
	err = cudaGetDevice(&devID);
	if (err != cudaSuccess) {
		cout << "ERRORRR" << endl;
	}
	
	int size = N * sizeof(float);
	float * d_A = NULL, *d_B = NULL, *d_C = NULL, *d_suma = NULL, *d_max = NULL;

	err = cudaMalloc((void **)&d_A, size);
	if (err != cudaSuccess) {
		cout << "ERROR RESERVA A" << endl;
	}

	err = cudaMalloc((void **)&d_B, size);
	if (err != cudaSuccess) {
		cout << "ERROR RESERVA B" << endl;
	}

	err = cudaMalloc((void **)&d_C, size);
	if (err != cudaSuccess) {
		cout << "ERROR RESERVA B" << endl;
	}
	err = cudaMalloc((void **)&d_suma, NBlocks*sizeof(float));
	if (err != cudaSuccess) {
		cout << "ERROR RESERVA B" << endl;
	}
	err = cudaMalloc((void **)&d_max, NBlocks*sizeof(float));
	if (err != cudaSuccess) {
		cout << "ERROR RESERVA B" << endl;
	}
	
	for (int i = 0; i < N; i++)
	{
		/*A[i] = i;
		B[i] = i;*/
		A[i] = (float)(1 - (i % 100)*0.001);
		B[i] = (float)(0.5 + (i % 10) *0.1);
	}

	err = cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		cout << "ERROR COPIA A" << endl;
	}

	err = cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		cout << "ERROR COPIA B" << endl;
	}
	
	int threadsPerBlock = blocksize;
	int blocksPerGrid = ceil((float)N/threadsPerBlock);

	double  t1 = clock();

	transformacionConCompartida << <NBlocks, blocksize, 4 * blocksize * sizeof(float) >> > (d_A, d_B, d_C, d_suma, d_max, N);
	//transformacionConCompartida << <blocksPerGrid, threadsPerBlock>> > (d_A, d_B, d_C, N);
	err = cudaGetLastError();

	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to launch kernel! ERROR= %d\n", err);
		exit(EXIT_FAILURE);
	}
	cudaMemcpy(D_max, d_max, NBlocks*sizeof(float), cudaMemcpyDeviceToHost);
	cout <<"adfadf"<< endl;
	cudaMemcpy(D_suma, d_suma, NBlocks*sizeof(float), cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();
	
	double mayor = D_max[0];
	for (int k = 1; k < blocksize; k++)
		if (D_max[k] > mayor)
			mayor = D_max[k];
	cout << mayor << " este es el mayor" << endl;
	double Tgpu = (clock() - t1) / CLOCKS_PER_SEC;

	cout << "Tiempo gastado GPU " << Tgpu << endl << endl;

t1=clock();

  
float mx; 
// Compute C[i], d[K] and mx
for (int k=0; k<NBlocks;k++)
{ int istart=k*blocksize;
  int iend  =istart+blocksize;
  D[k]=0.0;
  for (int i=istart; i<iend;i++)
  { C[i]=0.0;
    for (int j=istart; j<iend;j++)
     { float a=A[j]*i;
       if ((int)ceil(a) % 2 ==0)
	C[i]+= a + B[j];
       else
 	C[i]+= a - B[j];
     }
   D[k]+=C[i];
   mx=(i==1)?C[0]:max(C[i],mx);
  }
}

  double t2=clock();
  t2=(t2-t1)/CLOCKS_PER_SEC;
  


//for (int i=0; i<N;i++)   cout<<"C["<<i<<"]="<<C[i]<<endl;
cout<<"................................."<<endl;
//for (int k=0; k<NBlocks;k++)    cout<<"D["<<k<<"]="<<D[k]<<endl;
cout<<"................................."<<endl<<"El valor máximo en C es:  "<<mx<<endl;

cout<<endl<<"N="<<N<<"= "<<blocksize<<"*"<<NBlocks<<"  ........  Tiempo gastado CPU= "<<t2<<endl<<endl;
}
