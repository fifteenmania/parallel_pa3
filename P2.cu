#include <iostream>
#include <unistd.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include "utils.hpp"
#define SINGLE_SECTION 64

using namespace std;

// random matries
fp *A, *A0 = NULL;
// results
fp *b, *c, *b0 = NULL;
// consts
int n = 0, p = 0;


inline int get_pivot(int col)
{
    int pivot_row = col;
    fp pivot_max = fabs(A[col*n+col]);
    for (int i=col+1; i<n; i++){
        if (fabs(A[i*n+col])>pivot_max){
            pivot_max = fabs(A[i*n+col]);
            pivot_row = i;
        }
    }
    return pivot_row;
}

inline void swap_vec(int i, int j)
{
    fp temp = b[i];
    b[i] = b[j];
    b[j] = temp;
}

void GE_single()
{
    for (int i=0; i<n-1; i++){
        int piv = get_pivot(i);
        swap_ranges(A+i*n, A+i*n+n, A+piv*n);
        swap_vec(i, piv);
        for (int j=i+1; j<n; j++){
            fp ratio = A[j*n+i]/A[i*n+i];
            for (int k=i; k<n; k++){
                A[j*n+k] -= ratio*A[i*n+k];
            }
            b[j] -= ratio*b[i];
        }
    }
    return;
}



void GE_cuda()
{
    return;
}

/*
void GE_omp()
{
    for (int i=0; i<n-1-SINGLE_SECTION; i++){
        int piv = get_pivot(i);
        swap_ranges(A+i*n, A+i*n+n, A+piv*n);
        swap_vec(i, piv);
        double piv_val = A[i*n+i];
        #pragma omp parallel for num_threads(p)
        for (int j=i+1; j<n; j++){
            double ratio = A[j*n+i]/piv_val;
            for (int k=i; k<n; k++){
                A[j*n+k] -= ratio*A[i*n+k];
            }
            b[j] -= ratio*b[i];
        }
    }
    // Small sections are calculated with a single thread 
    // to prevent false sharing.
    for (int i=n-1-SINGLE_SECTION; i<n-1; i++){
        int piv = get_pivot(i);
        swap_ranges(A+i*n, A+i*n+n, A+piv*n);
        swap_vec(i, piv);
        for (int j=i+1; j<n; j++){
            double ratio = A[j*n+i]/A[i*n+i];
            for (int k=i; k<n; k++){
                A[j*n+k] -= ratio*A[i*n+k];
            }
            b[j] -= ratio*b[i];
        }
    }
    return;
}*/

void backsub()
{
    for (int i=n-1; i>=0; i--){
        double ratio = b[i]/A[i*n+i];
        b[i] = ratio;
        for (int j=0; j<i; j++){
            b[j] -= ratio*A[j*n+i];
        }
    }
    return;
}

void vmult()
{
    for (int i=0; i<n; i++){
        for (int j=0; j<n; j++){
            c[i] += A0[i*n+j] * b[j];
        }
    }
}

int main(int argc, char **argv)
{
    if (argc !=3){
        cout << "usage: [n]:size of input [p]:number of threads" << endl;
        return 0;
    }
    n = atoi(argv[1]);
    p = atoi(argv[2]);

    int seed = time(NULL);
    A = init_rand_mat(n, seed);
    A0 = cp_mat(A, n);
    b = init_rand_vec(n, seed+1);
    b0 = cp_vec(b, n);
    c = init_zeros_vec(n);
   
    struct timespec begin, end;
    // multi thread
    clock_gettime(CLOCK_MONOTONIC, &begin);
    GE_single();
    
    backsub();
    clock_gettime(CLOCK_MONOTONIC, &end);
    cout << "Parallel: " << time_elapsed(begin, end) << " ms" << endl;
    
    // c = Ab
    vmult();
    // correctness
    fp resid = v_l2_norm(c, b0, n);
    cout << "Residual: " << resid << endl;
    cout << "Correcct: " << (resid<SIM_THRES) << endl;

    return 0;
}


