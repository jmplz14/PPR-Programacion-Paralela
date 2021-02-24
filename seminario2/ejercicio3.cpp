#include "mpi.h"
#include <math.h>
#include <vector>
#include <cstdlib>
#include <iostream>
using namespace std;
 
int main(int argc, char *argv[]) {
    int tama, rank, size;
 
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
 
    if (argc < 2) {
        if (rank == 0) {
            cout << "No se ha especificado numero de elementos, multiplo de la cantidad de entrada, por defecto sera " << size * 100;
            cout << "\nUso: <ejecutable> <cantidad>" << endl;
        }
        tama = size * 100;
    } else {
        tama = atoi(argv[1]);
        if (tama < size) tama = size;
        else {
            int i = 1, num = size;
            while (tama > num) {
                ++i;
                num = size*i;
            }
            if (tama != num) {
                if (rank == 0)
                    cout << "Cantidad cambiada a " << num << endl;
                tama = num;
            }
        }
    }
 
    // Creacion y relleno de los vectores
    vector<long> VectorA, VectorB, VectorALocal;
    VectorA.resize(tama, 0);
    VectorB.resize(tama, 0);
    VectorALocal.resize(tama/size, 0);
    if (rank == 0) {
        for (long i = 0; i < tama; ++i) {
            VectorA[i] = i + 1; // Vector A recibe valores 1, 2, 3, ..., tama
            //VectorB[i] = (i + 1)*10; // Vector B recibe valores 10, 20, 30, ..., tama*10
        }
    }
 
    // Repartimos los valores del vector A
    MPI_Scatter(&VectorA[0], // Valores a compartir
            tama / size, // Cantidad que se envia a cada proceso
            MPI_LONG, // Tipo del dato que se enviara
            &VectorALocal[0], // Variable donde recibir los datos
            tama / size, // Cantidad que recibe cada proceso
            MPI_LONG, // Tipo del dato que se recibira
            0,  // proceso principal que reparte los datos
            MPI_COMM_WORLD); // Comunicador (En este caso, el global)
    // Repartimos los valores del vector B
    /*MPI_Scatter(&VectorB[0],
            tama / size,
            MPI_LONG,
            &VectorBLocal[0],
            tama / size,
            MPI_LONG,
            0,
            MPI_COMM_WORLD);*/
    long numRepes = ceil((float)tama/size);
    long inicio = numRepes * rank ;
    long final = inicio + numRepes;

    //printf("%d:%ld:%ld\n",size,inicio,final);
    for (long i = inicio; i < final ; i++)
    {
        VectorB[i] = (i + 1)*10;
    }

    //printf("%d:%ld:%ld:%ld\n",size,inicio,final,VectorB[inicio]);
    
    // Calculo de la multiplicacion escalar entre vectores
    long producto = 0;
    for (long i = 0; i < tama / size; ++i) {
        producto += VectorALocal[i] * VectorB[i + inicio];
    }
    long total;
    // Reunimos los datos en un solo proceso, aplicando una operacion
    // aritmetica, en este caso, la suma.
    MPI_Reduce(&producto, // Elemento a enviar
            &total, // Variable donde se almacena la reunion de los datos
            1, // Cantidad de datos a reunir
            MPI_LONG, // Tipo del dato que se reunira
            MPI_SUM, // Operacion aritmetica a aplicar
            0, // Proceso que recibira los datos
            MPI_COMM_WORLD); // Comunicador
 
    if (rank == 0)
        cout << "Total = " << total << endl;

    MPI_Finalize();
    return 0;
}