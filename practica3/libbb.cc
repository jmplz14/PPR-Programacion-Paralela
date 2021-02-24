/* ************************************************************************ */
/*  Libreria de funciones para el Branch-Bound y manejo de la pila          */
/* ************************************************************************ */

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include "libbb.h"
using namespace MPI;
extern unsigned int NCIUDADES;

// Tipos de mensajes que se env�an los procesos
const int PETICION = 0;
const int NODOS = 1;
const int TOKEN = 2;
const int FIN = 3;

// Estados en los que se puede encontrar un proceso
const int ACTIVO = 0;
const int PASIVO = 1;

// Colores que pueden tener tanto los procesos como el token
const int BLANCO = 0;
const int NEGRO = 1;

// Comunicadores que usar� cada proceso
MPI_Comm comunicadorCarga; // Para la distribuci�n de la carga
MPI_Comm comunicadorCota;  // Para la difusi�n de una nueva cota superior detectada

// Variables que indican el estado de cada proceso
extern int rank;           // Identificador del proceso dentro de cada comunicador (coincide en ambos)
extern int size;           // N�mero de procesos que est�n resolviendo el problema
int estado;                // Estado del proceso {ACTIVO, PASIVO}
int color;                 // Color del proceso {BLANCO,NEGRO}
int color_token;           // Color del token la �ltima vez que estaba en poder del proceso
bool token_presente;       // Indica si el proceso posee el token
int anterior;              // Identificador del anterior proceso
int siguiente;             // Identificador del siguiente proceso
bool difundir_cs_local;    // Indica si el proceso puede difundir su cota inferior local
bool pendiente_retorno_cs; // Indica si el proceso est� esperando a recibir la cota inferior de otro proceso

/* ********************************************************************* */
/* ****************** Funciones para el Branch-Bound  ********************* */
/* ********************************************************************* */

void LeerMatriz(char archivo[], int **tsp)
{
  FILE *fp;
  int i, j, r;

  if (!(fp = fopen(archivo, "r")))
  {
    printf("ERROR abriendo archivo %s en modo lectura.\n", archivo);
    exit(1);
  }
  printf("-------------------------------------------------------------\n");
  for (i = 0; i < NCIUDADES; i++)
  {
    for (j = 0; j < NCIUDADES; j++)
    {
      r = fscanf(fp, "%d", &tsp[i][j]);
      printf("%3d", tsp[i][j]);
    }
    r = fscanf(fp, "\n");
    printf("\n");
  }
  printf("-------------------------------------------------------------\n");
}

bool Inconsistente(int **tsp)
{
  int fila, columna;
  for (fila = 0; fila < NCIUDADES; fila++)
  { /* examina cada fila */
    int i, n_infinitos;
    for (i = 0, n_infinitos = 0; i < NCIUDADES; i++)
      if (tsp[fila][i] == INFINITO && i != fila)
        n_infinitos++;
    if (n_infinitos == NCIUDADES - 1)
      return true;
  }
  for (columna = 0; columna < NCIUDADES; columna++)
  { /* examina columnas */
    int i, n_infinitos;
    for (i = 0, n_infinitos = 0; i < NCIUDADES; i++)
      if (tsp[columna][i] == INFINITO && i != columna)
        n_infinitos++; /* increm el num de infinitos */
    if (n_infinitos == NCIUDADES - 1)
      return true;
  }
  return false;
}

void Reduce(int **tsp, int *ci)
{
  int min, v, w;
  for (v = 0; v < NCIUDADES; v++)
  {
    for (w = 0, min = INFINITO; w < NCIUDADES; w++)
      if (tsp[v][w] < min && v != w)
        min = tsp[v][w];
    if (min != 0)
    {
      for (w = 0; w < NCIUDADES; w++)
        if (tsp[v][w] != INFINITO && v != w)
          tsp[v][w] -= min;
      *ci += min; /* acumula el total restado para calc c.i. */
    }
  }
  for (w = 0; w < NCIUDADES; w++)
  {
    for (v = 0, min = INFINITO; v < NCIUDADES; v++)
      if (tsp[v][w] < min && v != w)
        min = tsp[v][w];
    if (min != 0)
    {
      for (v = 0; v < NCIUDADES; v++)
        if (tsp[v][w] != INFINITO && v != w)
          tsp[v][w] -= min;
      *ci += min; /* acumula cantidad restada en ci */
    }
  }
}

bool EligeArco(tNodo *nodo, int **tsp, tArco *arco)
{
  int i, j;
  for (i = 0; i < NCIUDADES; i++)
    if (nodo->incl()[i] == NULO)
      for (j = 0; j < NCIUDADES; j++)
        if (tsp[i][j] == 0 && i != j)
        {
          arco->v = i;
          arco->w = j;
          return true;
        }
  return false;
}

void IncluyeArco(tNodo *nodo, tArco arco)
{
  nodo->incl()[arco.v] = arco.w;
  if (nodo->orig_excl() == arco.v)
  {
    int i;
    nodo->datos[1]++;
    for (i = 0; i < NCIUDADES - 2; i++)
      nodo->dest_excl()[i] = NULO;
  }
}

bool ExcluyeArco(tNodo *nodo, tArco arco)
{
  int i;
  if (nodo->orig_excl() != arco.v)
    return false;
  for (i = 0; i < NCIUDADES - 2; i++)
    if (nodo->dest_excl()[i] == NULO)
    {
      nodo->dest_excl()[i] = arco.w;
      return true;
    }
  return false;
}

void PonArco(int **tsp, tArco arco)
{
  int j;
  for (j = 0; j < NCIUDADES; j++)
  {
    if (j != arco.w)
      tsp[arco.v][j] = INFINITO;
    if (j != arco.v)
      tsp[j][arco.w] = INFINITO;
  }
}

void QuitaArco(int **tsp, tArco arco)
{
  tsp[arco.v][arco.w] = INFINITO;
}

void EliminaCiclos(tNodo *nodo, int **tsp)
{
  int cnt, i, j;
  for (i = 0; i < NCIUDADES; i++)
    for (cnt = 2, j = nodo->incl()[i]; j != NULO && cnt < NCIUDADES;
         cnt++, j = nodo->incl()[j])
      tsp[j][i] = INFINITO; /* pone <nodo[j],i> infinito */
}

void ApuntaArcos(tNodo *nodo, int **tsp)
{
  int i;
  tArco arco;

  for (arco.v = 0; arco.v < NCIUDADES; arco.v++)
    if ((arco.w = nodo->incl()[arco.v]) != NULO)
      PonArco(tsp, arco);
  for (arco.v = nodo->orig_excl(), i = 0; i < NCIUDADES - 2; i++)
    if ((arco.w = nodo->dest_excl()[i]) != NULO)
      QuitaArco(tsp, arco);
  EliminaCiclos(nodo, tsp);
}

void InfiereArcos(tNodo *nodo, int **tsp)
{
  bool cambio;
  int cont, i, j;
  tArco arco;

  do
  {
    cambio = false;
    for (i = 0; i < NCIUDADES; i++) /* para cada fila i */
      if (nodo->incl()[i] == NULO)
      { /* si no hay incluido un arco <i,?> */
        for (cont = 0, j = 0; cont <= 1 && j < NCIUDADES; j++)
          if (tsp[i][j] != INFINITO && i != j)
          {
            cont++; /* contabiliza entradas <i,?> no-INFINITO */
            arco.v = i;
            arco.w = j;
          }
        if (cont == 1)
        { /* hay una sola entrada <i,?> no-INFINITO */
          IncluyeArco(nodo, arco);
          PonArco(tsp, arco);
          EliminaCiclos(nodo, tsp);
          cambio = true;
        }
      }
  } while (cambio);
}

void Reconstruye(tNodo *nodo, int **tsp0, int **tsp)
{
  int i, j;
  for (i = 0; i < NCIUDADES; i++)
    for (j = 0; j < NCIUDADES; j++)
      tsp[i][j] = tsp0[i][j];
  ApuntaArcos(nodo, tsp);
  EliminaCiclos(nodo, tsp);
  nodo->datos[0] = 0;
  Reduce(tsp, &nodo->datos[0]);
}

void HijoIzq(tNodo *nodo, tNodo *lnodo, int **tsp, tArco arco)
{
  int **tsp2 = reservarMatrizCuadrada(NCIUDADES);
  ;
  int i, j;

  CopiaNodo(nodo, lnodo);
  for (i = 0; i < NCIUDADES; i++)
    for (j = 0; j < NCIUDADES; j++)
      tsp2[i][j] = tsp[i][j];
  IncluyeArco(lnodo, arco);
  ApuntaArcos(lnodo, tsp2);
  InfiereArcos(lnodo, tsp2);
  Reduce(tsp2, &lnodo->datos[0]);
  liberarMatriz(tsp2);
}

void HijoDch(tNodo *nodo, tNodo *rnodo, int **tsp, tArco arco)
{
  int **tsp2 = reservarMatrizCuadrada(NCIUDADES);
  int i, j;

  CopiaNodo(nodo, rnodo);
  for (i = 0; i < NCIUDADES; i++)
    for (j = 0; j < NCIUDADES; j++)
      tsp2[i][j] = tsp[i][j];
  ExcluyeArco(rnodo, arco);
  ApuntaArcos(rnodo, tsp2);
  InfiereArcos(rnodo, tsp2);
  Reduce(tsp2, &rnodo->datos[0]);

  liberarMatriz(tsp2);
}

void Ramifica(tNodo *nodo, tNodo *lnodo, tNodo *rnodo, int **tsp0)
{
  int **tsp = reservarMatrizCuadrada(NCIUDADES);
  tArco arco;
  Reconstruye(nodo, tsp0, tsp);
  EligeArco(nodo, tsp, &arco);
  HijoIzq(nodo, lnodo, tsp, arco);
  HijoDch(nodo, rnodo, tsp, arco);

  liberarMatriz(tsp);
}

bool Solucion(tNodo *nodo)
{
  int i;
  for (i = 0; i < NCIUDADES; i++)
    if (nodo->incl()[i] == NULO)
      return false;
  return true;
}

int Tamanio(tNodo *nodo)
{
  int i, cont;
  for (i = 0, cont = 0; i < NCIUDADES; i++)
    if (nodo->incl()[i] == NULO)
      cont++;
  return cont;
}

void InicNodo(tNodo *nodo)
{
  nodo->datos[0] = nodo->datos[1] = 0;
  for (int i = 2; i < 2 * NCIUDADES; i++)
    nodo->datos[i] = NULO;
}

void CopiaNodo(tNodo *origen, tNodo *destino)
{
  for (int i = 0; i < 2 * NCIUDADES; i++)
    destino->datos[i] = origen->datos[i];
}

void EscribeNodo(tNodo *nodo)
{
  int i;
  printf("ci=%d : ", nodo->ci());
  for (i = 0; i < NCIUDADES; i++)
    if (nodo->incl()[i] != NULO)
      printf("<%d,%d> ", i, nodo->incl()[i]);
  if (nodo->orig_excl() < NCIUDADES)
    for (i = 0; i < NCIUDADES - 2; i++)
      if (nodo->dest_excl()[i] != NULO)
        printf("!<%d,%d> ", nodo->orig_excl(), nodo->dest_excl()[i]);
  printf("\n");
}

/* ********************************************************************* */
/* **********         Funciones para manejo de la pila  de nodos        *************** */
/* ********************************************************************* */

bool tPila::push(tNodo &nodo)
{
  if (llena())
    return false;

  // Copiar el nodo en el tope de la pila
  for (int i = 0; i < 2 * NCIUDADES; i++)
    nodos[tope + i] = nodo.datos[i];

  // Modificar el tope de la pila
  tope += 2 * NCIUDADES;
  return true;
}

bool tPila::pop(tNodo &nodo)
{
  if (vacia())
    return false;

  // Modificar el tope de la pila
  tope -= 2 * NCIUDADES;

  // Copiar los datos del nodo
  for (int i = 0; i < 2 * NCIUDADES; i++)
    nodo.datos[i] = nodos[tope + i];

  return true;
}

bool tPila::divide(tPila &pila2)
{

  if (vacia() || tamanio() == 1)
    return false;

  int mitad = tamanio() / 2;

  if (tamanio() % 2 == 0)
  { // La pila se puede dividir en dos partes iguales
    for (int i = 0; i < mitad; i++)
      for (int j = 0; j < 2 * NCIUDADES; j++)
        pila2.nodos[i * 2 * NCIUDADES + j] = nodos[(mitad + i) * 2 * NCIUDADES + j];
    tope = pila2.tope = mitad * 2 * NCIUDADES;
  }
  else
  { // La pila no se puede dividir en dos partes iguales
    for (int i = 0; i < mitad; i++)
      for (int j = 0; j < 2 * NCIUDADES; j++)
        pila2.nodos[i * 2 * NCIUDADES + j] = nodos[(mitad + i + 1) * 2 * NCIUDADES + j];
    tope = (mitad + 1) * 2 * NCIUDADES;
    pila2.tope = mitad * 2 * NCIUDADES;
  }

  return true;
}

void tPila::acotar(int U)
{
  int tope2 = 0;
  for (int i = 0; i < tope; i += 2 * NCIUDADES)
    if (nodos[i] <= U)
    {
      for (int j = i; j < 2 * NCIUDADES; j++)
        nodos[tope2 + j] = nodos[i + j];
      tope2 += 2 * NCIUDADES;
    }
  tope = tope2;
}

/* ******************************************************************** */
//         Funciones de reserva dinamica de memoria
/* ******************************************************************** */

// Reserva en el HEAP una matriz cuadrada de dimension "orden".
int **reservarMatrizCuadrada(unsigned int orden)
{
  int **m = new int *[orden];
  m[0] = new int[orden * orden];
  for (unsigned int i = 1; i < orden; i++)
  {
    m[i] = m[i - 1] + orden;
  }

  return m;
}

// Libera la memoria dinamica usada por matriz "m"
void liberarMatriz(int **m)
{
  delete[] m[0];
  delete[] m;
}

/* ******************************************************************** */
//         Funciones nuevas para la practica 3
/* ******************************************************************** */
void equilibrarCarga(tPila &pila, bool & fin, tNodo & solu)
{

  //información sobre operaciones de recepción de mensajes
  MPI_Status estadoMPI;

  int proceso_comunica;
  tNodo solucion_recibida;

  //colocamos el color en blanco
  color = BLANCO;

  //se pide trabajo si no se tiene y no ha finizaliado aun.
  if (pila.vacia())
  {
    
    //Se pide al proceso siguiente del comunicador de carga
    MPI_Send(&rank, 1, MPI_INT, siguiente, PETICION, comunicadorCarga);

    while (pila.vacia() && !fin)
    {
      //Espera la respuesta
      MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, comunicadorCarga, &estadoMPI);

      //Comprobamos el tipo de respuesta que obtenemos.
      switch (estadoMPI.MPI_TAG)
      {
      //recibe un mensaje de petición de nodos
      case PETICION:
        
        //obtenemos los datos de la petición del proceso anterior a este
        MPI_Recv(&proceso_comunica, 1, MPI_INT, anterior, PETICION, comunicadorCarga, &estadoMPI);

        //enviamos la peticion al siguiente proceso
        MPI_Send(&proceso_comunica, 1, MPI_INT, siguiente, PETICION, comunicadorCarga);

        //En el caso de que seamos nosotros mismo es que nadie respondio nuestra peticion y dio toda la vuelta
        if (proceso_comunica == rank)
        {

          //Marcamos que estamos en modo pasivo
          estado = PASIVO;

          //si temos el toquen reiniciamos la detección de fin
          if (token_presente)
          {

            //Si somos el el proceso 0 ponemos el toque blanco para iniciar el posible final y si no lo dejamos con nuestro color
            if (rank == 0)
            {
              color_token = BLANCO;
            }
            else
            {
              color_token = color;
            }

            //enviamos el mensaje al proceso anterior.
            MPI_Send(NULL, 0, MPI_INT, anterior, TOKEN, comunicadorCarga);

            //marcamos el token como que ya no esta presente y colocamos nuestro color a blanco
            token_presente = false;
            color = BLANCO;
          }
        }
        break;

      //Tendremos nodos con los que trabajar
      case NODOS:
        int numero_nodos;

        //obtenemos el numero de nodos
        MPI_Get_count(&estadoMPI, MPI_INT, &numero_nodos);

        //obtenemos el id de proceso del que nos contesto
        proceso_comunica = estadoMPI.MPI_SOURCE;

        //hacemos la petición para que nos envíe los nodos y marcamos en la pila el numero de nodos que recibimos
        MPI_Recv(pila.nodos, numero_nodos, MPI_INT, proceso_comunica, NODOS, comunicadorCarga, &estadoMPI);
        pila.tope = numero_nodos;
        estado = ACTIVO;
        break;

      //se recibe un tipo token
      case TOKEN:
        //recivimos el tipo token del proceso siguiente puesto que estos se envian al contrario
        MPI_Recv(NULL, 0, MPI_INT, siguiente, TOKEN, comunicadorCarga, &estadoMPI);

        //marcamos que tenemos el toquen
        token_presente = true;

        //si recibimos el toquen en estado pasivo
        if (estado == PASIVO)
        {
          //si es el proceso cero y recibimos el token blanco teniendo el color blanco este termina
          if (rank == 0 && color == BLANCO && color_token == BLANCO)
          {
            //marcamos la variable que indica el final a verdadero
            fin = true;

            //enviamos el mensaje de fin al proceso siguiente
            MPI_Send(solu.datos, NCIUDADES * 2, MPI_INT, siguiente, FIN, comunicadorCarga);

            //esperamos para recibir el final del proceso anterior que se corresponde con el utlimo
            MPI_Recv(solucion_recibida.datos, NCIUDADES * 2, MPI_INT, anterior, FIN, comunicadorCarga, &estadoMPI);

            //Si la solucion recibida es mejor no quedamos esta
            if (solucion_recibida.ci() < solu.ci())
              CopiaNodo(&solucion_recibida, &solu);
          }

          else
          {
            //si estamos en el proceso 0 pero no tenemos blanco se cambia a blanco y si no es el p0 enviamos nos quedamos el color que tenemos
            if (rank == 0)
            {
              color_token = BLANCO;
            }
            else
            {
              color_token = color;
            }
            // enviamos el mensaje token al anterior
            MPI_Send(NULL, 0, MPI_INT, anterior, TOKEN, comunicadorCarga);

            //marcamos que no tenemos el token y cambiamos nuestro color a blanco
            token_presente = false;
            color = BLANCO;
          }
        }
        break;
        //se recibe un mensaje tipo fin

      case FIN:
        //Marcamos fin como verdadero para indicar que este proceso ha finalizado
        fin = true;

        //Se recibe la solucion del nodo anterior
        MPI_Recv(solucion_recibida.datos, NCIUDADES * 2, MPI_INT, anterior, FIN, comunicadorCarga, &estadoMPI);

        //Nos quedamos con la mejor solucion
        if (solucion_recibida.ci() < solu.ci())
          CopiaNodo(&solucion_recibida, &solu);

        //enviamos la mejor solucion al siguiente
        MPI_Send(solu.datos, NCIUDADES * 2, MPI_INT, siguiente, FIN, comunicadorCarga);
        break;
      }
    }
  }

  //esta parte comprueba si tenemos alguna peticion de nodos o token si no ha finalizado aun
  //Si se puede atender se atiende y sino lo pasa al siguiente proceso
  if (!fin)
  {
    int flag;
    //Comprobamos si tenemos peticion
    
    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, comunicadorCarga, &flag, &estadoMPI);
    
    //Si se tiene alguna peticion
    while (flag > 0)
    { 
      
      //Miramos el estado de la peticion recibida
      switch (estadoMPI.MPI_TAG)
      {
      //Si es una peticion de nodo
      case PETICION:
        //recibimos la petición del proceso anterior
        MPI_Recv(&proceso_comunica, 1, MPI_INT, anterior, PETICION, comunicadorCarga, &estadoMPI);

        //Miramos si tenemos elementos para poder repartir
        if (pila.tamanio() > 1)
        {
          
          //creamos una nueva pila con la que enviar los datos
          tPila pila_enviar;

          //dividimos los datos de nuestra pila para pasar al proceso
          pila.divide(pila_enviar);

          //enviamos la pila creada al proceso que nos realizo la peticion.
          MPI_Send(pila_enviar.nodos, pila_enviar.tope, MPI_INT, proceso_comunica, NODOS, comunicadorCarga);

          //marcamos que se enviaron nodos cambiando nuestro color a negro si lo enviamos a un proceso menor que actual
          if (rank < proceso_comunica)
          {
            color = NEGRO;
          }
        }
        else
        {
          //enviamos al siguiente si nosotros no podemos repartir nodos
          MPI_Send(&proceso_comunica, 1, MPI_INT, siguiente, PETICION, comunicadorCarga);
        }
        break;
      //Se tiene una peticion de token
      case TOKEN:
        //recibimos el token y lo marcamos como que esta presente
        MPI_Recv(NULL, 0, MPI_INT, estadoMPI.MPI_SOURCE, TOKEN, comunicadorCarga, &estadoMPI);
        token_presente = true;

        break;
      }

      //volvemos a comprobar si se recibieron mas peticiones mientras procesamos la actual.
      
      MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, comunicadorCarga, &flag, &estadoMPI);
    }
  }
}


void difusion_Cota_Superior(int & cs_actual, bool & actualizar_cs){

  //Iniciamos los datos locales
  difundir_cs_local = actualizar_cs;
  pendiente_retorno_cs = false;
  MPI_Status estadoMPI;
  
  //enviamos el valor al siguiente proceso del anillo
  if (difundir_cs_local && !pendiente_retorno_cs){
    //Envías al siguiente proceso el valor local
    MPI_Send(&cs_actual, 1, MPI_INT, siguiente, rank, comunicadorCota);
    pendiente_retorno_cs = true;
    difundir_cs_local = false;
  }

  //obtenemos si hay peticiones que atender sobre la cota
  int numero_peticiones,proceso_comunica, cota_recibida;
  
  MPI_Iprobe(anterior, MPI_ANY_TAG, comunicadorCota, &numero_peticiones, &estadoMPI);
  proceso_comunica = estadoMPI.MPI_TAG;

  //se atiende a las peticiones
  while ( numero_peticiones > 0 ){ 

    //reciBimos la primera petición a espera deL proceso anterior
    MPI_Recv(&cota_recibida, 1, MPI_INT, anterior, proceso_comunica, comunicadorCota, &estadoMPI);
    
    //vemos si la petición mejora la actual
    if (cota_recibida < cs_actual){
      cs_actual = cota_recibida;
      actualizar_cs = true;
    }

    //si el proceso es el actual y  se ha mejorado la cota enviamos
    if (proceso_comunica == rank && difundir_cs_local){ 
        //se envia la cota nueva
        MPI_Send(&cs_actual, 1, MPI_INT, siguiente, rank, comunicadorCota);
        //se actualizan los datos para indicar que se envio un cota
        pendiente_retorno_cs = true;
        difundir_cs_local = false;

    //si el proceso es el actual y  se no ha mejorado la cota no la enviamos
    }else if (proceso_comunica == rank && !difundir_cs_local){
      pendiente_retorno_cs = false;
    //si no conincide con el mismo proceso se envia 
    }else{ 
      MPI_Send(&cs_actual, 1, MPI_INT, siguiente, proceso_comunica, comunicadorCota);
    }

    //se comprubea si se tienen mas envios en espera.
    
    MPI_Iprobe(anterior, MPI_ANY_TAG, comunicadorCota, &numero_peticiones, &estadoMPI );
    proceso_comunica = estadoMPI.MPI_TAG;
  }
  
 
}
