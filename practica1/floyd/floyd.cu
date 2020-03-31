#include <iostream>
#include <fstream>
#include <string.h>
#include <time.h>
#include "Graph.h"

// CUDA runtime
//#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
//#include <helper_functions.h>
//#include <helper_cuda.h>

#define blocksize 1024
#define blocksize2d 32

using namespace std;

//**************************************************************************
/*double cpuSecond()
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return((double)tp.tv_sec + (double)tp.tv_usec*1e-6);
}*/

//**************************************************************************
__global__ void floyd_kernel(int * M, const int nverts, const int k) {
	int ij = threadIdx.x + blockDim.x * blockIdx.x;
  if (ij < nverts * nverts) {
		int Mij = M[ij];
    int i= ij / nverts;
    int j= ij - i * nverts;
	//printf("%d:%d-", i, j);
    if (i != j && i != k && j != k) {
			int Mikj = M[i * nverts + k] + M[k * nverts + j];
    	Mij = (Mij > Mikj) ? Mikj : Mij;
    	M[ij] = Mij;
		}
  }
}

__global__ void floyd_kernel2d(int * M, const int nverts, const int k) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;  // Compute row index
	int i = blockIdx.y * blockDim.y + threadIdx.y;  // Compute column index

  //obtenemos la posicion del vector y vemos si es mayor que el tamaño total de la matriz
	if (i < nverts && j < nverts) {
		int ij = i * nverts + j;
		//printf("%d:%d-", i, j);
	  //obtenemos el vertice que corresponde
		int Mij = M[ij];


		//si no tenemos vertices repetidos
		if (i != j && i != k && j != k) {
			//obtenemos la suma de la distancias
			int Mikj = M[i * nverts + k] + M[k * nverts + j];
			//Nos quedamos la mejor
			Mij = (Mij > Mikj) ? Mikj : Mij;
			//Se escribe en la matriz
			M[ij] = Mij;
		}
	}
}

__global__ void reduce(int * input, int * output, int N)
{
	extern __shared__ int sdata[blocksize];
	int tid = threadIdx.x;
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	sdata[tid] = ((i < N) ? input[i] : 0.0f);
	__syncthreads();

	for (int s=blockDim.x/2; s>0; s>>=1) {
		if (tid < s) {
			int valor = sdata[tid + s];
			if( sdata[tid] < valor)
				sdata[tid] = valor;
		}
		__syncthreads();
	}
	if (tid == 0) output[blockIdx.x] = sdata[0];
}



int main (int argc, char *argv[]) {

	if (argc != 2) {
		cerr << "Sintaxis: " << argv[0] << " <archivo de grafo>" << endl;
		return(-1);
	}
	

  //Get GPU information
  int devID;
  cudaDeviceProp props;
  cudaError_t err;
  err = cudaGetDevice(&devID);
  if(err != cudaSuccess) {
		cout << "ERRORRR" << endl;
	}


cudaGetDeviceProperties(&props, devID);
  printf("Device %d: \"%s\" with Compute %d.%d capability\n", devID, props.name, props.major, props.minor);

	Graph G;
	G.lee(argv[1]);// Read the Graph

	//cout << "EL Grafo de entrada es:"<<endl;
	//G.imprime();
	const int nverts = G.vertices;
	const int niters = nverts;

	const int nverts2 = nverts * nverts;

	int *c_Out_M = new int[nverts2];
	int *c_Out_M2d = new int[nverts2];
	int *c_Out_reduce = new int[blocksize];
	int size = nverts2*sizeof(int);
	int * d_In_M = NULL, * d_In_M2d = NULL, * d_out_reduce = NULL;

	err = cudaMalloc((void **) &d_In_M, size);
	if (err != cudaSuccess) {
		cout << "ERROR RESERVA" << endl;
	}

	err = cudaMalloc((void **)&d_In_M2d, size);
	if (err != cudaSuccess) {
		cout << "ERROR RESERVA 2d" << endl;
	}

	err = cudaMalloc((void **)&d_out_reduce, blocksize*sizeof(int));
	if (err != cudaSuccess) {
		cout << "ERROR RESERVA reduce" << endl;
	}

	int *A = G.Get_Matrix();

//---------------------------------GPU phase 1d-------------------------------------
	double  t1 = clock();

	err = cudaMemcpy(d_In_M, A, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		cout << "ERROR COPIA A GPU" << endl;
	}

	for(int k = 0; k < niters; k++) {
		//printf("CUDA kernel launch \n");
	 	int threadsPerBlock = blocksize;
	 	int blocksPerGrid = (nverts2 + threadsPerBlock - 1) / threadsPerBlock;

	  floyd_kernel<<<blocksPerGrid,threadsPerBlock >>>(d_In_M, nverts, k);
	  err = cudaGetLastError();

	  if (err != cudaSuccess) {
	  	fprintf(stderr, "Failed to launch kernel! ERROR= %d\n",err);
	  	exit(EXIT_FAILURE);
		}
	}

	cudaMemcpy(c_Out_M, d_In_M, size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	double Tgpu = (clock()-t1) / CLOCKS_PER_SEC;

	cout << "Tiempo gastado GPU1d= " << Tgpu << endl << endl;

//-----------------------------------GPU phase 2d----------------------------------
	t1 = clock();

	err = cudaMemcpy(d_In_M2d, A, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		cout << "ERROR COPIA A GPU2d" << endl;


	}

	dim3 threadsPerBlock(blocksize2d, blocksize2d);
	dim3 numBlocks(ceil((float)nverts / blocksize2d), ceil((float)nverts / blocksize2d));
	for (int k = 0; k < niters; k++) {

		floyd_kernel2d << <numBlocks, threadsPerBlock >> > (d_In_M2d, nverts, k);
		err = cudaGetLastError();

		if (err != cudaSuccess) {
			fprintf(stderr, "Failed to launch kernel! ERROR= %d\n", err);
			exit(EXIT_FAILURE);
		}
	}

	cudaMemcpy(c_Out_M2d, d_In_M2d, size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	double Tgpu2d = (clock() - t1) / CLOCKS_PER_SEC;

	cout << "Tiempo gastado GPU2d= " << Tgpu2d << endl << endl;

//----------------------------------CPU phase-----------------------------------------
	t1 = clock();

	// BUCLE PPAL DEL ALGORITMO
	int inj, in, kn;
	for(int k = 0; k < niters; k++) {
          kn = k * nverts;
	  for(int i=0;i<nverts;i++) {
			in = i * nverts;
			for(int j = 0; j < nverts; j++)
	       			if (i!=j && i!=k && j!=k){
			 	    inj = in + j;
			 	    A[inj] = min(A[in+k] + A[kn+j], A[inj]);
	       }
	   }
	}

  double t2 = (clock() - t1) / CLOCKS_PER_SEC;
  cout << "Tiempo gastado CPU= " << t2 << endl << endl;

//---------------------------------reduce cuda------------------------------------------
t1 = clock();
reduce <<<ceil((float)nverts2/blocksize), blocksize>>> (d_In_M2d, d_out_reduce, nverts2);
err = cudaGetLastError();
if (err != cudaSuccess) {
	fprintf(stderr, "Failed to launch kernel! ERROR= %d\n", err);
	exit(EXIT_FAILURE);
}

cudaMemcpy(c_Out_reduce, d_out_reduce, sizeof(int)*blocksize, cudaMemcpyDeviceToHost);
cudaDeviceSynchronize();

int mayorGpu = c_Out_reduce[0];
for (int i=1; i<blocksize;i++)
	if (mayorGpu < c_Out_reduce[i])
		mayorGpu = c_Out_reduce[i];
double tReduceGpu = (clock() - t1) / CLOCKS_PER_SEC;
cout << "Tiempo gastado reduce GPU= " << tReduceGpu << endl;
cout << "MayorGpu = " << mayorGpu << endl << endl;	
//----------------------------------reduce cpu--------------------------------------------
t1 = clock();
int mayorCpu = c_Out_reduce[0];
for (int i=1; i<nverts2;i++)
	if (mayorCpu < A[i])
		mayorCpu = A[i];
double tReduceCpu = (clock() - t1) / CLOCKS_PER_SEC;
cout << "Tiempo gastado reduce CPU= " << tReduceCpu << endl;
cout << "MayorCpu = " << mayorCpu << endl << endl;

  
  cout << "Ganancia 1d sobre serie= " << t2 / Tgpu << endl;
  cout << "Ganancia 2d sobre serie= " << t2 / Tgpu2d << endl;
  cout << "Ganancia 2d sobre sobre 1d= " << Tgpu / Tgpu2d << endl;
  
  


  for(int i = 0; i < nverts; i++)
    for(int j = 0;j < nverts; j++)
       if (abs(c_Out_M[i*nverts+j] - G.arista(i,j)) > 0)
         cout << "Error 1d (" << i << "," << j << ")   " << c_Out_M[i*nverts+j] << "..." << G.arista(i,j) << endl;
  
  for (int i = 0; i < nverts; i++)
	  for (int j = 0; j < nverts; j++)
		  if (abs(c_Out_M2d[i*nverts + j] - G.arista(i, j)) > 0)
			  cout << "Error 2d (" << i << "," << j << ")   " << c_Out_M2d[i*nverts + j] << "..." << G.arista(i, j) << endl;

}
