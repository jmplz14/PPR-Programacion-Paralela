#include <iostream>
#include <fstream>
#include <string.h>
#include <sys/time.h>
#include "Graph.h"
#include "mpi.h"
#include "math.h"

using namespace std;

//**************************************************************************

int main(int argc, char *argv[])
{

        MPI::Init(argc, argv);

        if (argc != 2)
        {
                cerr << "Sintaxis: " << argv[0] << " <archivo de grafo>" << endl;
                return (-1);
        }

        Graph G;
        int nverts, rank, size;
        MPI_Comm COM_COLUMNAS, COM_FILAS;
        //obtenemos id proceso
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        //obtenemos numero de procesos del comunicador
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        // Cargamos los datos del fichero
        if (rank == 0)
        {
                G.lee(argv[1]);
                nverts = G.vertices;
                //G.imprime();
        }
        // Pasamos el numero de vertices a todos los procesos
        MPI_Bcast(&nverts, 1, MPI_INT, 0, MPI_COMM_WORLD);

        //iniciamos variables de tamaño necesarias
        /*const int bsize1d = nverts / size;
        const int bsize2d = bsize1d * nverts;*/
        const int num_matrices_fil_y_col = sqrt(size);
        const int tam_submatriz = nverts / num_matrices_fil_y_col;
        const int tam_total_Submatriz = tam_submatriz * tam_submatriz;
        
        //cogemos el puntero de la matriz de datos
        //int *A = G.Get_Matrix();

        //Process 0 scatters blocks of matrix A
        /*int matrizLocalA[tam_submatriz][tam_submatriz];	
	int * local_A= new int[bsize2d];*/

        //variables necesarias para enviar la parte de la matriz a cada proceso
        int *buf_envio = new int[nverts * nverts];
        MPI_Datatype MPI_BLOQUE;
        int posicion, fila_P, columna_P, comienzo;
        int *matriz = G.Get_Matrix();
        if (rank == 0)
        {

                //Definimos el tipo de bloque que nos sera necesario y se crea
                MPI_Type_vector(tam_submatriz, tam_submatriz, nverts, MPI_INT, &MPI_BLOQUE);
                MPI_Type_commit(&MPI_BLOQUE);
                //bucle encargado de enpaquetar los datos 
                posicion = 0;
                for (int i = 0; i < size; i++)
                {
                        //obtenemos los indices de la submatriz necesaria y la empaquetamos 
                        fila_P = i / num_matrices_fil_y_col;
                        columna_P = i % num_matrices_fil_y_col;
                        comienzo = (columna_P * tam_submatriz) + (fila_P * tam_total_Submatriz * num_matrices_fil_y_col);
                        MPI_Pack(matriz + comienzo, 1, MPI_BLOQUE, buf_envio, sizeof(int) * nverts * nverts, &posicion, MPI_COMM_WORLD);
                }
                
        }

        //enviamos a cada proceso su parte y esperamos que todos la tengan
        int buf_recep[tam_submatriz][tam_submatriz];
        MPI_Scatter(buf_envio, sizeof(int) * tam_total_Submatriz, MPI_PACKED, buf_recep, tam_total_Submatriz, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        

        //creamos los comunicadores nuevos
        const int rank_fila = rank / num_matrices_fil_y_col;
        const int rank_columna = rank - rank_fila * num_matrices_fil_y_col;
        
        MPI_Comm_split(MPI_COMM_WORLD, rank_fila, rank, &COM_COLUMNAS);
        MPI_Comm_split(MPI_COMM_WORLD, rank_columna, rank, &COM_FILAS);

        double t1 = MPI_Wtime();

        // obtenemos los indices locales
        const int local_i_start = 0;
        const int local_j_start = 0;
        const int local_i_end = nverts / num_matrices_fil_y_col;
        const int local_j_end = local_i_end;
        

        //calculamos los indices globales para comparar luego con los elementos de la fila k

        int global_i = rank_fila * tam_submatriz;
        int global_j = rank_columna * tam_submatriz;
        
        //creamos los vectores para almacenar la fila y la columna k
        int fila_k[nverts];
        int columna_k[nverts];

        //iniciamos el algortimos
        for (int k = 0; k < nverts; k++)
        {
                //calculamos la fila k y columna knecesaria y miramos si este proceso es el que la contiene para copiarla
                int proceso_tiene_k = k / tam_submatriz;
                if (proceso_tiene_k == rank_fila)
                {
                        memcpy(fila_k, &buf_recep[k % tam_submatriz][0], sizeof(int) * tam_submatriz);
                }
                if (proceso_tiene_k == rank_columna)
                {
                        int local_k = k % tam_submatriz;
                        for (int i = 0; i < tam_submatriz; i++)
                                columna_k[i] = buf_recep[i][local_k];
                }

                // Enviamos la fila y columna o la recivimos dependiendo del proceso que se sea.
                MPI_Bcast(fila_k, tam_submatriz, MPI_INT, proceso_tiene_k, COM_FILAS);
                MPI_Bcast(columna_k, tam_submatriz, MPI_INT, proceso_tiene_k, COM_COLUMNAS);


                //bucle que realizara los calculos de los nuevos valores
                for (int i = local_i_start; i < local_i_end; i++)
                {

                        for (int j = local_j_start; j < local_j_end; j++)
                        {
                                
                                if (global_i != global_j && global_i != k && global_j != k)
                                {
                                        int suma = fila_k[j] + columna_k[i];
                                        buf_recep[i][j] = min(buf_recep[i][j], suma);
                                }
                                global_j++;
                        }
                        global_i++;
                }
        }

        //obtenemos el tiempo para la medicion despues de que todos terminen
        MPI_Barrier(MPI_COMM_WORLD); 
        double t2 = MPI_Wtime() - t1;

        //enviamos la submantrices a el proceso 0
        MPI_Gather(buf_recep, tam_total_Submatriz, MPI_INT, buf_envio, sizeof(int) * tam_total_Submatriz, MPI_PACKED, 0, MPI_COMM_WORLD);

        

        //desempaquetamos las matrices
        int *m_Final = new int[nverts * nverts];
        int *matrizResultado = G.Get_Matrix();
        if (rank == 0)
        {
                posicion = 0;
                for (int i = 0; i < size; i++)
                {
                        //obtenemos la posicion que le corresponde
                        fila_P = i / num_matrices_fil_y_col;
                        columna_P = i % num_matrices_fil_y_col;
                        comienzo = (columna_P * tam_submatriz) + (fila_P * tam_total_Submatriz * num_matrices_fil_y_col);
                        //Desenpaquetamos la matriz
                        MPI_Unpack(buf_envio, sizeof(int) * nverts *nverts, &posicion,  matriz + comienzo, 1,   MPI_BLOQUE, MPI_COMM_WORLD);        
                    

                }
        }

 
        //mostramos el tiempo
        if (rank == 0)
        {
               // G.imprime();
                cout << "Tiempo gastado= " << t2 << endl;
        }

        MPI::Finalize();
}
