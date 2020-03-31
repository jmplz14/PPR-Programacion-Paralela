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

#define blocksize 32

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
	int size = nverts2*sizeof(int);
	int * d_In_M = NULL;

	err = cudaMalloc((void **) &d_In_M, size);
	if (err != cudaSuccess) {
		cout << "ERROR RESERVA" << endl;
	}

	int *A = G.Get_Matrix();

	// GPU phase
	double  t1 = clock();

	err = cudaMemcpy(d_In_M, A, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		cout << "ERROR COPIA A GPU" << endl;
	}
	dim3 threadsPerBlock(blocksize, blocksize);
	dim3 numBlocks(ceil((float)nverts / blocksize), ceil((float)nverts / blocksize));
	for(int k = 0; k < niters; k++) {
		//printf("CUDA kernel launch \n");
	 	//int threadsPerBlock = blocksize;
	 	//int blocksPerGrid = (nverts2 + threadsPerBlock - 1) / threadsPerBlock;


	  floyd_kernel<<<numBlocks,threadsPerBlock >>>(d_In_M, nverts, k);
	  err = cudaGetLastError();

	  if (err != cudaSuccess) {
	  	fprintf(stderr, "Failed to launch kernel! ERROR= %d\n",err);
	  	exit(EXIT_FAILURE);
		}
	}

	cudaMemcpy(c_Out_M, d_In_M, size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	double Tgpu = clock()-t1;

	cout << "Tiempo gastado GPU= " << Tgpu << endl << endl;

	// CPU phase
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

  double t2 = clock() - t1;
  cout << "Tiempo gastado CPU= " << t2 << endl << endl;
  cout << "Ganancia= " << t2 / Tgpu << endl;


  for(int i = 0; i < nverts; i++)
    for(int j = 0;j < nverts; j++)
       if (abs(c_Out_M[i*nverts+j] - G.arista(i,j)) > 0)
         cout << "Error (" << i << "," << j << ")   " << c_Out_M[i*nverts+j] << "..." << G.arista(i,j) << endl;

}
