#include "utils.hpp"


using namespace std;
double time_elapsed(struct timespec &start, struct timespec &end)
{
    double time = end.tv_sec - start.tv_sec;
    time = time*1000 + (double)(end.tv_nsec - start.tv_nsec)/1000000;
    return time;
}

double *init_zeros_mat(int n)
{
    double *A = (double *)malloc(n*n*sizeof(double));
    for (int i=0; i<n*n; i++){
        A[i] = 0;
    }
    return A;
}

double *init_rand_mat(int n, int seed)
{
    double *A = (double*)malloc(n*n*sizeof(double));
    srand48(seed);
    for (int i=0; i<n*n; i++){
        A[i] = drand48();
    }
    return A;
}

double *init_zeros_vec(int n)
{
    double *A = (double *)malloc(n*sizeof(double));
    for (int i=0; i<n; i++){
        A[i] = 0;
    }
    return A;
}

double *init_rand_vec(int n, int seed)
{
    double *A = (double *)malloc(n*sizeof(double));
    srand48(seed);
    for (int i=0; i<n; i++){
        A[i] = drand48();
    }
    return A;
}


double *cp_mat(double *A, int n)
{
    double *B = (double *)malloc(n*n*sizeof(double));
    memcpy(B, A, n*n*sizeof(double));
    return B;
}

double *cp_vec(double *A, int n)
{
    double *B = (double *)malloc(n*sizeof(double));
    memcpy(B, A, n*sizeof(double));
    return B;
}


void print_mat(double *A, int n)
{
    cout << setprecision(3) << fixed;
    for (int i=0; i<n; i++){
        for (int j=0; j<n; j++){
            cout << A[i*n+j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

void print_vec(double *A, int n)
{
    cout << setprecision(3) << fixed;
    for (int i=0; i<n; i++){
        cout << A[i] << endl;
    }
    cout << endl;
}

