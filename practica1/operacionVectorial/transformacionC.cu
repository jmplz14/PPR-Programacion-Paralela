#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>

using namespace std;

//**************************************************************************
__global__ void transformacion_global(float * A, float * B, float * C, float * D, float * mx)
{

    int tid = threadIdx.x;
    int i = tid + blockDim.x * blockIdx.x;
    float c = 0.0; // valor a calcular

    extern __shared__ float sdata[]; // Memoria Compartida
    float *sdata_A = sdata; 	   // Apunta al primer valor de A
    float *sdata_B = sdata+blockDim.x;    // Apunta al primer valor de B
    float *sdata_C = sdata+blockDim.x*2;  // Apunta al primer valor de C
    float *sdata_aux = sdata+blockDim.x*3; // Apunta al primer valor de una copia de C


    // Paso a memoria compartida de A y B
    *(sdata_A+tid) = A[i];
    *(sdata_B+tid) = B[i];

    __syncthreads();

    /***** Calculo del vector C con memoria global  *****/

    int jini = blockIdx.x * blockDim.x;
    int jfin = jini + blockDim.x;
    for (int j = jini; j < jfin; j++){ 
        float resultado = A[j] * i ;
        int s = int(ceil(resultado))%2 == 0 ? 1 : -1;
        c += resultado + B[j] * s;
    }

    
    C[i] = c;
    *(sdata_C+tid) = c;
    *(sdata_aux+tid) = c;

    __syncthreads();

    /***** Calcula la reduccion de la suma y lo guarda en D y la reducción del mayor y lo guarda en mx *****/
    float v1, v2;
    for ( unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
        if (tid < s)
        {
            *(sdata_C+tid) += *(sdata_C+tid+s);
            v1 = *(sdata_aux+tid);
            v2 = *(sdata_aux+tid+s);
            *(sdata_aux+tid) = (v1 > v2) ? v1 : v2;
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        D[blockIdx.x] = *(sdata_C);
        mx[blockIdx.x] = *(sdata_aux);
    }
}
    
    //**************************************************************************
__global__ void transformacion_shared(float * A, float * B, float * C, float * D, float * mx)
{

    int tid = threadIdx.x;
    int i = tid + blockDim.x * blockIdx.x;
    float c = 0.0; // valor a calcular

    extern __shared__ float sdata[]; // Memoria Compartida
    float *sdata_A = sdata; 	   // Apunta al primer valor de A
    float *sdata_B = sdata+blockDim.x;    // Apunta al primer valor de B
    float *sdata_C = sdata+blockDim.x*2;  // Apunta al primer valor de C
    float *sdata_aux = sdata+blockDim.x*3; // Apunta al primer valor de una copia de C


    // Paso a memoria compartida de A y B
    *(sdata_A+tid) = A[i];
    *(sdata_B+tid) = B[i];

    __syncthreads();

    /***** Calculo del vector C con memoria compartida  *****/

    for (int j = 0; j < blockDim.x; j++){
        float resultado = *(sdata_A+j) * i + *(sdata_B+j);
        int s = int(ceil(resultado))%2 == 0 ? 1 : -1;
        c += resultado + B[j] * s;
    }

    C[i] = c;
    *(sdata_C+tid) = c;
    *(sdata_aux+tid) = c;

    __syncthreads();


    /***** Calcula la reduccion de la suma y lo guarda en D y la reducción del mayor y lo guarda en mx *****/
    float v1, v2;
    for ( unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
        if (tid < s)
        {
            *(sdata_C+tid) += *(sdata_C+tid+s);
            v1 = *(sdata_aux+tid);
            v2 = *(sdata_aux+tid+s);
            *(sdata_aux+tid) = (v1 > v2) ? v1 : v2;
        }
        __syncthreads();
    }

    if (tid == 0){
        D[blockIdx.x] = *(sdata_C);
        mx[blockIdx.x] = *(sdata_aux);
    }

}


//**************************************************************************
//**************************************************************************


int main (int argc, char *argv[]) {

    int Bsize, NBlocks;
    if (argc != 3)
    { cout << "Uso: transformacion Num_bloques Tam_bloque  "<<endl;
        return(0);
    }
    else
    {
        NBlocks = atoi(argv[1]);
        Bsize= atoi(argv[2]);
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

   

    const int   N = Bsize * NBlocks;

    //* pointers to host memory */
    float *h_A, *h_B, *h_C, *h_D, *h_D_global, *h_D_shared, h_mx,  *h_mx_global, *h_mx_shared;
    float *d_A, *d_B, *d_C,       *d_D_global, *d_D_shared,        *d_mx_global, *d_mx_shared;

    //* Allocate arrays a, b and c on host*/
    h_A = new float[N];
    h_B = new float[N];
    h_C = new float[N];
    h_D = new float[NBlocks];

    //variables para kernel global
    h_D_global = new float[NBlocks];
    h_mx_global= new float[NBlocks];

    //variables para kernel compartido
    h_D_shared = new float[NBlocks];
    h_mx_shared = new float[NBlocks];

    //reservar memoria para variables del device
    d_A = NULL; d_B = NULL; d_C = NULL;
    d_D_global = NULL; d_D_shared = NULL;
    d_mx_global = NULL; d_mx_shared = NULL;

	err = cudaMalloc((void **) &d_A, N*sizeof(float));
	if (err != cudaSuccess) {
		cout << "ERROR RESERVA A" << endl;
	}

	err = cudaMalloc((void **) &d_B, N*sizeof(float));
	if (err != cudaSuccess) {
		cout << "ERROR RESERVA B" << endl;
    }
    
    err = cudaMalloc((void **) &d_C, N*sizeof(float));
	if (err != cudaSuccess) {
		cout << "ERROR RESERVA C" << endl;
    }
    
    err = cudaMalloc((void **) &d_D_global, NBlocks*sizeof(float));
	if (err != cudaSuccess) {
		cout << "ERROR RESERVA D_global" << endl;
    }
    
    err = cudaMalloc((void **) &d_D_shared, NBlocks*sizeof(float));
	if (err != cudaSuccess) {
		cout << "ERROR RESERVA D_Shared" << endl;
    }
    
    err = cudaMalloc((void **) &d_mx_global, NBlocks*sizeof(float));
	if (err != cudaSuccess) {
		cout << "ERROR RESERVA mx_global" << endl;
    }
    
    err = cudaMalloc((void **) &d_mx_shared, NBlocks*sizeof(float));
	if (err != cudaSuccess) {
		cout << "ERROR RESERVA mx_shared" << endl;
	}

	//* Initialize arrays A and B */
    for (int i=0; i<N;i++)
    { 
        h_A[i]= (float) (1  -(i%100)*0.001);
        h_B[i]= (float) (0.5+(i%10) *0.1  );    
    }

	//************************  GPU PHASE TRANSFORMACION GLOBAL **********************************

	double  t1 = clock();

	err = cudaMemcpy(d_A, h_A, N*sizeof(float), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		cout << "ERROR COPIA EN A" << endl;
    }
    
    err = cudaMemcpy(d_B, h_B, N*sizeof(float), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		cout << "ERROR COPIA EN B" << endl;
    }

	dim3 threadsPerBlock(Bsize,1);
	dim3 numBloques(NBlocks,1);
	
	transformacion_global<<<numBloques, threadsPerBlock, Bsize*4*sizeof(float)>>>(d_A, d_B, d_C, d_D_global, d_mx_global);  
      
    err = cudaGetLastError();

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch transformacion global kernel!\n");
        exit(EXIT_FAILURE);
    }

    cudaMemcpy(h_D_global, d_D_global, NBlocks*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mx_global, d_mx_global, NBlocks*sizeof(float), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    
    // Obtenemos el valor de la reducción
    float mx_global = h_mx_global[0];
    for (int k = 1; k<NBlocks; k++)
        mx_global = (mx_global > h_mx_global[k]) ? mx_global : h_mx_global[k];

    double TGPUTransformacionGlobal = (clock()-t1)/CLOCKS_PER_SEC;
    


	//************************  GPU PHASE TRANSFORMACION SHARED **********************************

	t1 = clock();
	
	transformacion_shared<<<numBloques, threadsPerBlock, Bsize*4*sizeof(float)>>>(d_A, d_B, d_C, d_D_shared, d_mx_shared);
      
    err = cudaGetLastError();

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch transformacion shared kernel!\n");
        exit(EXIT_FAILURE);
    }

    cudaMemcpy(h_D_shared, d_D_shared, NBlocks*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mx_shared, d_mx_shared, NBlocks*sizeof(float), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    
    // Obtenemos el valor de la reducción
    float mx_shared = h_mx_shared[0];
    for (int k = 1; k<NBlocks; k++)
    mx_shared = (mx_shared > h_mx_shared[k]) ? mx_shared : h_mx_shared[k];

    double TGPUTransformacionShared = (clock()-t1)/CLOCKS_PER_SEC;
    

	//************************  CPU PHASE VERSION SECUENCIAL **********************************
    
    // Time measurement  
    t1=clock();

    // Compute C[i], d[K] and mx
    for (int k=0; k<NBlocks;k++)
    { 
        int istart=k*Bsize;
        int iend  =istart+Bsize;
        h_D[k]=0.0;
        for (int i=istart; i<iend;i++)
        { h_C[i]=0.0;
            for (int j=istart; j<iend;j++)
            { float a=h_A[j]*i;
            if ((int)ceil(a) % 2 ==0)
            h_C[i]+= a + h_B[j];
            else
            h_C[i]+= a - h_B[j];
            }
        h_D[k]+=h_C[i];
        h_mx=(i==1)?h_C[0]:max(h_C[i],h_mx);
        }
    }

  double t2=clock();
  t2=(t2-t1)/CLOCKS_PER_SEC;


	cout << "********** Valores máximos obtenidos **********" << endl;
	cout<<endl;
	cout << "Máximo de C version secuencial = " << h_mx << endl << endl;
	cout << "Máximo de C version global = " << mx_global << endl << endl;
	cout << "Máximo de C version compartida = " << mx_shared << endl;
	cout <<endl;
	
    cout << "********** Tiempos Obtenidos **********" << endl;
    cout<<endl;
    cout << "N =" << N << " = " << Bsize << " * " << NBlocks << endl;
    
	cout << "Tiempo gastado version secuencial = " << t2 << endl << endl;
	cout << "Tiempo gastado version global = " << TGPUTransformacionGlobal << endl << endl;
	cout << "Tiempo gastado version compartida  = " << TGPUTransformacionShared << endl;
	
	cout << "********** Ganancias **********" << endl;
	cout<<endl;
	cout << "Ganancia version global = " << t2/TGPUTransformacionGlobal << endl << endl;
	cout << "Ganancia version compartida = " << t2/TGPUTransformacionShared << endl << endl;
	
}
