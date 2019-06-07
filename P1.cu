#include <iostream>
#include <unistd.h>
#include <time.h>
#include <stdlib.h>
#include <limits>
#include "utils.hpp"

#define TILE_WIDTH 16
//#define BENCH 1

using namespace std;

// fp == float
// random matries
fp *A, *B = NULL;
// results
fp *C, *D = NULL;
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
void MatrixMulKernel(fp *d_A, fp *d_B, fp *d_C, int width)
{
    int row_idx = blockIdx.y*blockDim.y + threadIdx.y;
    int col_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if ((row_idx < width) && (col_idx < width)){
        fp Cval = 0;
        for (int k=0; k<width; k++){
            fp Aval = d_A[row_idx*width+k];
            fp Bval = d_B[k*width+col_idx];
            Cval += Aval*Bval;
        }
        d_C[row_idx*width+col_idx] = Cval;
    }
}

__global__
void MatrixMulKernelS(float *d_A, float *d_B, float *d_C, int width)
{
    __shared__ float subTileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float subTileB[TILE_WIDTH][TILE_WIDTH];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row_idx = by*TILE_WIDTH + ty;
    int col_idx = bx*TILE_WIDTH + tx;
    int tile_max = (width+TILE_WIDTH-1)/TILE_WIDTH;
    if ((row_idx < width) && (col_idx < width)){
        float Cval = 0;
        for (int m=0; m<tile_max; m++){
            int idx_Ax = m*TILE_WIDTH + tx;
            int idx_A = row_idx*width + idx_Ax;
            int idx_By = m*TILE_WIDTH + ty;
            int idx_B = (idx_By)*width + col_idx;
            if ((row_idx < width) && (idx_Ax < width)){
                subTileA[ty][tx] = d_A[idx_A];
            } else{
                subTileA[ty][tx] = 0;
            }
            if ((idx_By < width) && (col_idx < width)){
                subTileB[ty][tx] = d_B[idx_B];
            } else{
                subTileB[ty][tx] = 0;
            }
            __syncthreads();
            for (int k=0; k<TILE_WIDTH; k++){
                Cval += subTileA[ty][k] * subTileB[k][tx];
            }
            __syncthreads();
        }
        d_C[row_idx*width+col_idx] = Cval;
    }
}

void multiply_cuda()
{
    int size = n*n*sizeof(fp);
    fp *d_A, *d_B, *d_D;
    cudaMalloc(&d_A, size);
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    
    cudaMalloc(&d_B, size);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    cudaMalloc(&d_D, size);  

    // Kernel invoke
    int grid_width = (n+TILE_WIDTH-1)/TILE_WIDTH;
    dim3 dimGrid(grid_width, grid_width, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    MatrixMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_D, n);

    // Copy result
    cudaMemcpy(D, d_D, size, cudaMemcpyDeviceToHost);
    
    // Free device matrices
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_D);
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
    //A = init_ones_mat(n);
    //B = init_ones_mat(n);
    C = (fp *)init_zeros_mat(n);
    D = (fp *)init_zeros_mat(n);
    
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
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &end);
    cout << "Parallel: " << time_elapsed(begin, end) << " ms" << endl;

    //print_mat(A, n);
    //print_mat(B, n);
    //print_mat(C, n);
    //print_mat(D, n);
#ifndef BENCH
    // correctness
    max_norm(C, D, n);
    //cout << "residue:  " << norm << endl;
    //cout << "Correct:  " << (norm<SIM_THRES) << endl;
#endif
    return 0;
}

