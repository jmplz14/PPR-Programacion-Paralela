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

//**************************************************************************
/*double cpuSecond()
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return((double)tp.tv_sec + (double)tp.tv_usec*1e-6);
}*/

//**************************************************************************
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

__global__ void transformacionConCompartida(float * A, float * B, float * C, int N) {
	extern __shared__ float sdata[];
	int tid = threadIdx.x;
	int i = tid + blockDim.x * blockIdx.x;
	
	float *sdata_A = sdata; 
	float *sdata_B = sdata + blockDim.x; 

	sdata_A[tid] = A[i];
	sdata_B[tid] = B[i];

	__syncthreads();

	if (i < N) {
		int K = N / i;
		int posInicio = K * blockDim.x;
		float suma = 0;

		for (int j = 0; j < blockDim.x; j++) {
			int posActual = posInicio + j;
			float valorA = sdata_A[posActual] * i;

			if (((int)ceil(valorA) % 2) == 0) {
				suma += valorA + sdata_B[posActual];
			}
			else {
				suma += valorA - sdata_B[posActual];
			}
		}
		C[i] = suma;
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

	int devID;
	cudaError_t err;
	err = cudaGetDevice(&devID);
	if (err != cudaSuccess) {
		cout << "ERRORRR" << endl;
	}
	
	int size = N * sizeof(float);
	float * d_A = NULL, *d_B = NULL, *d_C = NULL;

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

	err = cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		cout << "ERROR COPIA A" << endl;
	}

	err = cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		cout << "ERROR COPIA B" << endl;
	}

	err = cudaMemcpy(d_C, C, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		cout << "ERROR COPIA C" << endl;
	}

	for (int i = 0; i < N; i++)
	{
		A[i] = (float)(1 - (i % 100)*0.001);
		B[i] = (float)(0.5 + (i % 10) *0.1);
	}
	
	int threadsPerBlock = blocksize;
	int blocksPerGrid = ceil((float)N/threadsPerBlock);

	double  t1 = clock();

	transformacionConCompartida << <blocksPerGrid, threadsPerBlock, 2 * blocksize * sizeof(float) >> > (d_A, d_B, d_C, N);
	err = cudaGetLastError();

	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to launch kernel! ERROR= %d\n", err);
		exit(EXIT_FAILURE);
	}

	cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	double Tgpu = (clock() - t1) / CLOCKS_PER_SEC;

	cout << "Tiempo gastado GPU " << Tgpu << endl << endl;
}
