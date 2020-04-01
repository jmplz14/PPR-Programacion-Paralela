#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>

using namespace std;

//**************************************************************************
int main(int argc, char *argv[])
//**************************************************************************
{
int Bsize, NBlocks;
if (argc != 3)
  { cout << "Uso: transformacion Num_bloques Tam_bloque  "<<endl;
    return(0);
  }
else
  {NBlocks = atoi(argv[1]);
   Bsize= atoi(argv[2]);
  }

const int   N=Bsize*NBlocks;
//* pointers to host memory */

float *A, *B, *C,*D;

//* Allocate arrays a, b and c on host*/
A = new float[N];
B = new float[N];
C = new float[N];
D = new float[NBlocks];
float mx; // maximum of C

//* Initialize arrays A and B */
for (int i=0; i<N;i++)
  { A[i]= (float) (1  -(i%100)*0.001);
    B[i]= (float) (0.5+(i%10) *0.1  );    
  }

// Time measurement  
double t1=clock();

  
  
// Compute C[i], d[K] and mx
for (int k=0; k<NBlocks;k++)
{ int istart=k*Bsize;
  int iend  =istart+Bsize;
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
for (int k=0; k<NBlocks;k++)    cout<<"D["<<k<<"]="<<D[k]<<endl;
cout<<"................................."<<endl<<"El valor máximo en C es:  "<<mx<<endl;

cout<<endl<<"N="<<N<<"= "<<Bsize<<"*"<<NBlocks<<"  ........  Tiempo gastado CPU= "<<t2<<endl<<endl;


//* Free the memory */
delete(A); 
delete(B); 
delete(C);
delete(D);

}