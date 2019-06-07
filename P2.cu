#include <iostream>
#include <unistd.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include "utils.hpp"
#define SINGLE_SECTION 64
#define TILE_WIDTH 16

using namespace std;

inline int get_pivot(float *A, int n, int col)
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

inline void swap_vec(float *b, int i, int j)
{
    fp temp = b[i];
    b[i] = b[j];
    b[j] = temp;
}

void GE_single(float *A, float *b, int n)
{
    for (int i=0; i<n-1; i++){
        int piv = get_pivot(A, n, i);
        swap_ranges(A+i*n, A+i*n+n, A+piv*n);
        swap_vec(b, i, piv);
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

__global__
void RowSubKernel(float *d_A, float *d_b, int n, int col_idx)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if ((row < n) && (col && n)){
        float pivot_val = d_A[col_idx*n+col_idx];
        float rowhd_val = d_A[row*n+col_idx];
        float row1_val = d_A[col_idx*n+col];
        float row2_val = d_A[row*n+col];
        d_A[row*n+col] = row2_val - (row1_val * rowhd_val / pivot_val);
    }
}

void row_sub(float *A, float *b, float *d_A, float *d_b, int n, int col_idx)
{
    int size_A = n * n * sizeof(float);
    int size_b = n * sizeof(float);
    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size_b, cudaMemcpyHostToDevice);
    
    int grid_width = (n + TILE_WIDTH - 1)/TILE_WIDTH;
    dim3 dimGrid = (grid_width, grid_width, 1);
    dim3 dimBlock = (TILE_WIDTH, TILE_WIDTH, 1);

    RowSubKernel<<<dimGrid, dimBlock>>>(d_A, d_b, n, col_idx);

    cudaMemcpy(A, d_A, size_A, cudaMemcpyDeviceToHost);
    cudaMemcpy(b, d_b, size_b, cudaMemcpyDeviceToHost);
}

void GE_cuda(float *A, float *b, int n)
{
    float *d_A, *d_b;
    int size_A = n * n * sizeof(float);
    int size_b = n * sizeof(float);
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_b, size_b);
    
    for (int i=0; i<n-1; i++){
        int piv = get_pivot(A, n, i);
        swap_ranges(A+i*n, A+i*n+n, A+piv*n);
        swap_vec(b, i, piv);
        row_sub(A, b, d_A, d_b, n, i);
    }
    cudaFree(d_A);
    cudaFree(d_b);
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

void backsub(float *A, float *b, int n)
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

void vmult(float *A, float *b, float *c, int n)
{
    for (int i=0; i<n; i++){
        for (int j=0; j<n; j++){
            c[i] += A[i*n+j] * b[j];
        }
    }
}

int main(int argc, char **argv)
{
    if (argc !=2){
        cout << "usage: [n]:size of input" << endl;
        return 0;
    }
    int n;
    float *A, *b, *c;
    float *A0, *b0;
    
    n = atoi(argv[1]);

    int seed = time(NULL);
    A = init_rand_mat(n, seed);
    A0 = cp_mat(A, n);
    b = init_rand_vec(n, seed+1);
    b0 = cp_vec(b, n);
    c = init_zeros_vec(n);
   
    struct timespec begin, end;
    // multi thread
    clock_gettime(CLOCK_MONOTONIC, &begin);
    GE_cuda(A, b, n);
    backsub(A, b, n);
    clock_gettime(CLOCK_MONOTONIC, &end);
    cout << "Parallel: " << time_elapsed(begin, end) << " ms" << endl;
    
    // c = Ab
    vmult(A0, b, c, n);
    // correctness
    fp resid = v_l2_norm(c, b0, n);
    cout << "Residual: " << resid << endl;
    cout << "Correcct: " << (resid<SIM_THRES) << endl;

    return 0;
}


