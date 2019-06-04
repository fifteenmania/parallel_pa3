#include <iostream>
#include <unistd.h>
#include <time.h>
#include <stdlib.h>
#include "utils.hpp"

#define TILE_WIDTH 16

using namespace std;

// random matries
double *A, *B = NULL;
// results
double *C, *D = NULL;
// consts
int n = 0;

void multiply_single()
{
    for (int i=0; i<n; i++){
        for (int j=0; j<n; j++){
            for (int k=0; k<n; k++){
                C[i*n+k] += A[i*n+j]*B[n*j+k];
            }
        }
    }
}

__global__
void MatrixMulKernel(double *d_A, double *d_B, double *d_C, int width)
{
    int row_idx = blockIdx.y*blockDim.y + threadIdx.y;
    int col_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if ((row_idx < width) && (col_idx < width)){
        double Cval = 0;
        for (int k=0; k<width; k++){
            double Aval = d_A[row_idx*width+k];
            double Bval = d_B[k*width+col_idx];
            Cval += Aval*Bval;
        }
        d_C[row_idx*width+col_idx] = Cval;
    }
}

void multiply_cuda()
{
    int size = n*n*sizeof(double);
    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    
    cudaMalloc(&d_B, size);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    cudaMalloc(&d_C, size);  

    // Kernel invoke
    int grid_width = (n+TILE_WIDTH-1)/TILE_WIDTH;
    dim3 dimGrid(grid_width, grid_width, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    MatrixMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, n);

    // Copy result
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    
    // Free device matrices
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

bool mat_equal()
{
    for (int i=0; i<n*n; i++){
        if (C[i] != D[i])
            return false;
    }
    return true;
}

int main(int argc, char **argv)
{
    if (argc !=2){
        cout << "usage: [n]:size of input" << endl;
        return 0;
    }
    n = atoi(argv[1]);

    int seed = time(NULL);
    A = init_rand_mat(n, seed);
    B = init_rand_mat(n, seed+1);
    C = (double *)init_zeros_mat(n);
    D = (double *)init_zeros_mat(n);
    
    struct timespec begin, end;
    // single thread
#ifndef BENCH
    clock_gettime(CLOCK_MONOTONIC, &begin);
    multiply_single();
    clock_gettime(CLOCK_MONOTONIC, &end);
    cout << "Single:   " << time_elapsed(begin, end) << " ms" << endl;
#endif

    // multi thread
    //
    clock_gettime(CLOCK_MONOTONIC, &begin);
    multiply_cuda();
    clock_gettime(CLOCK_MONOTONIC, &end);
    cout << "Parallel: " << time_elapsed(begin, end) << " ms" << endl;

#ifndef BENCH
    // correctness
    cout << "Correct:  " << mat_equal() << endl;
#endif
    return 0;
}

