#include "mpi.h"
#include <math.h>
int main(int argc, char **argv)
{

    int n, myid, numprocs, i;
    double mypi, pi, h, sum, x;
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    if (myid == 0)
    {
        printf("Number of intervals: ");
        scanf("%d", &n);
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    h = 1.0 / (double)n;
    int numRepes = ceil((float)n/numprocs);
    int inicio = numRepes * myid + 1;
    int final = inicio + numRepes-1;
    final = final < n ? final : n;
    sum = 0.0;

    for (i = inicio; i <= final ; i++)
    {
        x = h * ((double)i - 0.5);
        sum += 4.0 / (1.0 + x * x);
    }
    mypi = h * sum;

    MPI_Reduce(&mypi, &pi, 1, MPI_DOUBLE, MPI_SUM,
               0, MPI_COMM_WORLD);
    if (myid == 0)
        printf("pi is approximately %.16f, \n", pi);
    MPI_Finalize();
    return 0;
}