/* ******************************************************************** */
/*               Algoritmo Branch-And-Bound Secuencial                  */
/* ******************************************************************** */
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <mpi.h>
#include "libbb.h"

unsigned int NCIUDADES;
int rank, size;

main(int argc, char **argv)
{
        MPI::Init(argc, argv);
        
        switch (argc)
        {
        case 3:
                NCIUDADES = atoi(argv[1]);
                break;
        default:
                std::cerr << "La sintaxis es: bbseq <tama�o> <archivo>" << std::endl;
                exit(1);
                break;
        }

        //Calculamos el tamaño de la matriz
        int total_matriz = NCIUDADES * NCIUDADES;

        //marcara el final de la ejecución
        bool fin = false,
             nueva_U; // hay nuevo valor de c.s.

        int U = INFINITO; // valor de c.s. se inicializa a infinito

        //puntero a la matriz de datos
        int** tsp0 = reservarMatrizCuadrada(NCIUDADES);

        tNodo nodo,   // nodo a explorar
            lnodo,    // hijo izquierdo
            rnodo,    // hijo derecho
            solucion; // mejor solucion

        tPila pila; // pila de nodos a explorar

        int iteraciones = 0;
        
        //variables que marcan si tienes el token del fichero libbb.cc
        extern bool token_presente;

        // Comunicadore para la carga y la cota del fichero libbb.cc
        extern MPI_Comm comunicadorCarga, comunicadorCota;
        

        //obtenemos id proceso
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        //obtenemos numero de procesos del comunicador
        MPI_Comm_size(MPI_COMM_WORLD, &size);
       
        //variables que identifican el proceso siguiente y anterior del fichero libbb.cc
        extern int anterior, siguiente;
        
        //Proceso siguiente y anterior en el anillo
        siguiente = (rank + 1) % size;
        anterior = (rank - 1 + size) % size;

        //creamos los dos comunicadeores declarados anteriormente realizando una copia del principal.
        MPI_Comm_dup(MPI_COMM_WORLD, &comunicadorCarga);
        MPI_Comm_dup(MPI_COMM_WORLD, &comunicadorCota);
        
        //el proceso 0 se encarga de leer los datos y repartirlos
        if (rank == 0)
        {
                //Marcamos que el proceso 0 tiene el token
                token_presente = true;

                //Se carga el fichero que pasamos como párametro
                LeerMatriz(argv[2], tsp0);
                //repartimos la matriz entre todos los procesos
                MPI_Bcast(&tsp0[0][0], total_matriz, MPI_INT, 0, MPI_COMM_WORLD);
                //inicialializamos el nodo
                InicNodo(&nodo);
        }
       
       
        //Se realiza el equilibrado de la carga para el resto de procesos.
        if (rank != 0)
        {
                //Se recoge la pate de la matriz que le corresponde
                MPI_Bcast(&tsp0[0][0], total_matriz, MPI_INT, 0, MPI_COMM_WORLD);
                
                //marcamos el toquen como false para indicar que no lo tenemos
                token_presente = false;

                //Llamamos a la funcion de equilibar carga.
                equilibrarCarga(pila, fin, solucion);

                //Si no estamos al final sacamos un nodo de la pila
                if (!fin)
                {
                        pila.pop(nodo);
                }
        }
        double t=MPI_Wtime();
        while (!fin)
        { // ciclo del Branch&Bound
                Ramifica(&nodo, &lnodo, &rnodo, tsp0);
                nueva_U = false;
                if (Solucion(&rnodo))
                {
                        if (rnodo.ci() < U)
                        { // se ha encontrado una solucion mejor
                                U = rnodo.ci();
                                nueva_U = true;
                                CopiaNodo(&rnodo, &solucion);
                        }
                }
                else
                { //  no es un nodo solucion
                        if (rnodo.ci() < U)
                        { //  cota inferior menor que cota superior
                                if (!pila.push(rnodo))
                                {
                                        printf("Error: pila agotada\n");
                                        liberarMatriz(tsp0);
                                        exit(1);
                                }
                        }
                }
                if (Solucion(&lnodo))
                {
                        if (lnodo.ci() < U)
                        { // se ha encontrado una solucion mejor
                                U = lnodo.ci();
                                nueva_U = true;
                                CopiaNodo(&lnodo, &solucion);
                        }
                }
                else
                { // no es nodo solucion
                        if (lnodo.ci() < U)
                        { // cota inferior menor que cota superior
                                if (!pila.push(lnodo))
                                {
                                        printf("Error: pila agotada\n");
                                        liberarMatriz(tsp0);
                                        exit(1);
                                }
                        }
                }

                difusion_Cota_Superior(U, nueva_U);
                if (nueva_U){
                        pila.acotar(U);
                }
                        

                equilibrarCarga(pila, fin, solucion);
                if (!fin){
                        pila.pop(nodo);  
                }
                        
                iteraciones++;
        }


        t=MPI::Wtime()-t;
        
        MPI_Barrier(MPI_COMM_WORLD);
        if(rank == 0){
                printf ("Solucion: \n");
	        EscribeNodo(&solucion);
                std::cout<< "Tiempo gastado= "<<t<<std::endl;
        }
	MPI_Barrier(MPI_COMM_WORLD);
        
	std::cout << "Numero de iteraciones del proceso " << rank << " = " << iteraciones << std::endl;
	liberarMatriz(tsp0);
        MPI::Finalize();
}
