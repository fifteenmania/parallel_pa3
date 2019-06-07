#include "mmreader.hpp"
#include <time.h>
#include <iostream>
#include <sys/time.h>
#include <unistd.h>
#include "utils.hpp"
#define TILE_WIDTH 16

//#define BENCH 1

bool
SCsrMatrixfromFile(struct sparse_mtx *A, const char* filePath)
{
    // Check that the file format is matrix market; the only format we can read right now
    // This is not a complete solution, and fails for directories with file names etc...
    // TODO: Should we use boost filesystem?
    std::string strPath( filePath );
    if( strPath.find_last_of( '.' ) != std::string::npos )
    {
        std::string ext = strPath.substr( strPath.find_last_of( '.' ) + 1 );
        if( ext != "mtx" )
        {
            std::cout << "Reading file name error" << std::endl;
            return false;
        }
    }
    else
        return false;

    // Read data from a file on disk into buffers
    // Data is read natively as COO format with the reader
    MatrixMarketReader mm_reader;
    if( mm_reader.MMReadFormat(filePath) )
        return false;

    // JPA: Shouldn't that just be an assertion check? It seems to me that
    // the user have to call clsparseHeaderfromFile before calling this function,
    // otherwise the whole pCsrMatrix will be broken;
    A->nrow = mm_reader.GetNumRows( );
    A->ncol = mm_reader.GetNumCols( );
    A->nnze = mm_reader.GetNumNonZeroes( );

    A->row = (int32_t *)malloc((A->nrow + 1) * sizeof(int32_t));
    A->val = (float *)malloc(A->nnze * sizeof(float));
    A->col = (int32_t *)malloc(A->nnze * sizeof(int32_t));

    if(A->row == NULL || A->col == NULL || A->val == NULL)
    {
        if(A->row == NULL)
            free((void *)A->row);
        if(A->col == NULL)
            free((void *)A->col);
        if(A->val == NULL)
            free((void *)A->val);
        return false;
    }

    //  The following section of code converts the sparse format from COO to CSR
    Coordinate* coords = mm_reader.GetUnsymCoordinates( );

    std::sort( coords, coords + A->nnze, CoordinateCompare );

    int32_t current_row = 1;

    A->row[ 0 ] = 0;

    for (int32_t i = 0; i < (int32_t)A->nnze; i++)
    {
        A->col[ i ] = coords[ i ].y;
        A->val[ i ] = coords[ i ].val;

        while( coords[ i ].x >= current_row )
            A->row[ current_row++ ] = i;
    }

    A->row[ current_row ] = A->nnze;

    while( current_row <= (int32_t)A->nrow )
        A->row[ current_row++ ] = A->nnze;

    return true;
}

void multiply_single(struct sparse_mtx *A, struct dense_mtx *B, struct dense_mtx *C)
{
    if(C->val == NULL)
        return;
    for(int32_t i = 0; i < (int32_t)A->nrow; i++)
    {
        int32_t A_col_start = A->row[i];
        int32_t A_col_stop = A->row[i + 1];
        
        for(int32_t j = A_col_start; j < A_col_stop; j++)
        {
            int32_t B_row = A->col[j];

            for(int32_t k = 0; k < (int32_t)B->ncol; k++)
                C->val[i * C->ncol + k] += A->val[j] * B->val[B_row * B->ncol + k];
        }
    }
}

__global__
void SparseMMKernel(const int A_nrow, const int A_ncol, 
        const int B_ncol, 
        int32_t *d_Arow, int32_t *d_Acol, 
        float *d_Aval, float *d_Bval,
        float *d_Cval)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if ((row < A_nrow) && (col < B_ncol)){
        int col_start = d_Arow[row];
        int col_end = d_Arow[row + 1];
        float Cval = 0;
        for (int j=col_start; j<col_end; j++){
            int B_row = d_Acol[j];
            Cval += d_Aval[j] * d_Bval[B_row*B_ncol + col]; 
        }
        d_Cval[row*B_ncol + col] = Cval;
    }
}

void multiply_cuda(struct sparse_mtx *A, struct dense_mtx *B, struct dense_mtx *C)
{
    float *d_Aval, *d_Bval, *d_Cval;
    int32_t *d_Arow, *d_Acol; 
    
    // Copy to device
    int size_Aval = (int) A->nnze * sizeof(float);
    int size_Arow = ((int) A->nrow + 1) * sizeof(int32_t);
    int size_Acol = (int) A->nnze * sizeof(int32_t);
    int size_Bval = (int) B->nrow * (int) B->ncol * sizeof(float);
    int size_Cval = (int) A->nrow * (int) B->ncol * sizeof(float);
    cudaMalloc(&d_Aval, size_Aval);
    cudaMalloc(&d_Arow, size_Arow);
    cudaMalloc(&d_Acol, size_Acol);
    cudaMalloc(&d_Bval, size_Bval);
    cudaMalloc(&d_Cval, size_Cval);

    cudaMemcpy(d_Aval, A->val, size_Aval, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Arow, A->row, size_Arow, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Acol, A->col, size_Acol, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Bval, B->val, size_Bval, cudaMemcpyHostToDevice);
    
    // Kernel invoke
    int grid_width = (B->ncol + TILE_WIDTH - 1)/TILE_WIDTH;
    int grid_height = (A->nrow + TILE_WIDTH - 1)/TILE_WIDTH;
    dim3 dimGrid(grid_width, grid_height, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    SparseMMKernel<<<dimGrid, dimBlock>>>(A->nrow, A->ncol, 
            B->ncol, 
            d_Arow, d_Acol, 
            d_Aval, d_Bval, 
            d_Cval);

    // Copy results
    cudaMemcpy(C->val, d_Cval, size_Cval, cudaMemcpyDeviceToHost);

    cudaFree(d_Aval);
    cudaFree(d_Arow);
    cudaFree(d_Acol);
    cudaFree(d_Bval);
    cudaFree(d_Cval);
}

float max_norm(struct dense_mtx *C1, struct dense_mtx *C2, int num_round)
{
    float max_error = 0;
    float error = 0;
    float max_abs_error = 0;
    float abs_error = 0;
    int max_idx = 0;
    for (uint32_t i=0; i<C1->nrow*C1->ncol; i++){
        abs_error = fabs(C1->val[i] - C2->val[i]);
        error = abs_error / max(fabs(C1->val[i]), std::numeric_limits<float>::min());
        if (error > max_error){
            max_error = error;
            max_idx = i;
        }
        if (abs_error > max_abs_error)
            max_abs_error = abs_error;
    }
    //std::cout.precision(std::numeric_limits<float>::max_digits10);
    //std::cout << "max diff " << C1->val[max_idx] << " and " << C2->val[max_idx] << std::endl
    //    << "max error " << max_error << std::endl;
    std::cout << "------   Correctness Test Result   ------" << std::endl;
    std::cout << "policy        : 'max_rel_err < 5.0e-7' " << std::endl;
    std::cout << "num_op        : " << num_round << std::endl;
    std::cout << "max_entry     : " << C1->val[max_idx] << ",  " << C2->val[max_idx] << std::endl;
    std::cout << "max_abs_err   : " << max_abs_error << std::endl;
    std::cout << "max_rel_err   : " << max_error << std::endl;
    std::cout << "correctness   : " << std::boolalpha <<(max_error<5.0e-7) << std::endl << std::endl;
    return max_error;
}

bool mat_equal(struct dense_mtx *C1, struct dense_mtx *C2)
{
    for (uint32_t i=0; i<C1->nrow*C1->ncol; i++){
        if (C1->val[i] != C2->val[i])
            return false;
    }
    return true;
}


uint64_t GetTimeStamp() {
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return tv.tv_sec*(uint64_t)1000000+tv.tv_usec;
}

int main(int argc, char **argv)
{
    struct sparse_mtx A;
    if(!SCsrMatrixfromFile(&A, argv[1]))
    {
        std::cout << "read failed." << std::endl;
        return 0;
    }
    std::cout << "Matrix: " << argv[1] << std::endl;

    struct dense_mtx B;
    B.nrow = A.ncol;
    B.ncol = atoi(argv[2]);
    B.val = (float *)malloc(sizeof(float) * B.nrow * B.ncol);

    srand((unsigned int)time(NULL));
    for(int i = 0; i < (int)B.nrow; i++)
    {
        for(int j = 0; j < (int)B.ncol; j++)
        {
            B.val[B.ncol * i + j] = ((float)rand()/(float)(RAND_MAX)) * ((rand() % 2) ? 1.0f : -1.0f);
        }
    }

    struct dense_mtx C1, C2;
    C1.val = NULL;
    C2.val = NULL;
    
    C1.nrow = A.nrow;
    C1.ncol = B.ncol;
    C1.val = (float *)malloc(C1.nrow * C1.ncol * sizeof(float));
    
    C2.nrow = A.nrow;
    C2.ncol = B.ncol;
    C2.val = (float *)malloc(C2.nrow * C2.ncol * sizeof(float));
    
    struct timespec start, end;
    #ifndef BENCH
    std::cout << "Single Thread Computation Start" << std::endl;
    clock_gettime(CLOCK_MONOTONIC, &start);
    multiply_single(&A, &B, &C1);
    clock_gettime(CLOCK_MONOTONIC, &end);
    std::cout << "Single Thread Computation End: " << time_elapsed(start, end)  << " ms." << std::endl;
    #endif

    std::cout << "CUDA Computation Start" << std::endl;
    clock_gettime(CLOCK_MONOTONIC, &start);
    multiply_cuda(&A, &B, &C2);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &end);
    std::cout << "CUDA Computation End: " << time_elapsed(start, end) << " ms." << std::endl << std::endl;

    // TODO: Testing Code by comparing C1 and C2

    //std::cout << C1.val[145725055] << std::endl;
    //std::cout << C2.val[145725055] << std::endl;
    
    #ifndef BENCH
    max_norm(&C1, &C2, A.ncol);
    //std::cout << "max_err/op  : " << max_error << std::endl;
    //std::cout << "correctness : " << (max_error<5.0e-7) << std::endl;
    #endif

    free(A.row);
    free(A.col);
    free(A.val);
    free(B.val);
    if(C1.val != NULL)
        free(C1.val);
    if(C2.val != NULL)
        free(C2.val);
    
    return 0;
}
