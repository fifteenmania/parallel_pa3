#include "utils.hpp"


using namespace std;
double time_elapsed(struct timespec &start, struct timespec &end)
{
    double time = end.tv_sec - start.tv_sec;
    time = time*1000 + (double)(end.tv_nsec - start.tv_nsec)/1000000;
    return time;
}

fp *init_zeros_mat(int n)
{
    fp *A = (fp *)malloc(n*n*sizeof(fp));
    for (int i=0; i<n*n; i++){
        A[i] = 0;
    }
    return A;
}

fp *init_rand_mat(int n, int seed)
{
    fp *A = (fp*)malloc(n*n*sizeof(fp));
    srand48(seed);
    for (int i=0; i<n*n; i++){
        A[i] = (fp)drand48();
    }
    return A;
}

fp *init_zeros_vec(int n)
{
    fp *A = (fp *)malloc(n*sizeof(fp));
    for (int i=0; i<n; i++){
        A[i] = 0;
    }
    return A;
}

fp *init_ones_mat(int n)
{
    fp *A = (fp *)malloc(n*n*sizeof(fp));
    for (int i=0; i<n*n; i++){
        A[i] = 1;
    }
    return A;
}

fp *init_rand_vec(int n, int seed)
{
    fp *A = (fp *)malloc(n*sizeof(fp));
    srand48(seed);
    for (int i=0; i<n; i++){
        A[i] = (fp)drand48();
    }
    return A;
}


fp *cp_mat(double *A, int n)
{
    fp *B = (fp *)malloc(n*n*sizeof(fp));
    memcpy(B, A, n*n*sizeof(fp));
    return B;
}

fp *cp_vec(fp *A, int n)
{
    fp *B = (fp *)malloc(n*sizeof(fp));
    memcpy(B, A, n*sizeof(fp));
    return B;
}


void print_mat(fp *A, int n)
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

void print_vec(fp *A, int n)
{
    cout << setprecision(3) << fixed;
    for (int i=0; i<n; i++){
        cout << A[i] << endl;
    }
    cout << endl;
}

fp l2_norm(fp *A, fp *B, int n)
{
    fp error;
    fp cum_error = 0;
    for (int i=0; i<n*n; i++){
        error = (A[i] - B[i]);
        cum_error += error * error;
    }
    return sqrt(cum_error/n/n);
}

fp v_l2_norm(fp *A, fp *B, int n)
{
    fp cum_error = 0;
    for (int i=0; i<n; i++){
        fp error = A[i] - B[i];
        cum_error += error * error;
    }
    return sqrt(cum_error/n);
}
/*
fp max_norm(fp *A, fp *B, int n)
{
    fp error;
    fp max_error = 0;
    for (int i=0; i<n*n; i++){
        error = fabs(A[i] - B[i]);
        if (error > max_error){
            max_error = error;
        }
    }
    return max_error/n;
}*/

float max_norm(float *A, float *B, int n)
{
    float max_error = 0;
    float error = 0;
    float max_abs_error = 0;
    float abs_error = 0;
    int max_idx = 0;
    for (int i=0; i<n*n; i++){
        abs_error = fabs(A[i] - B[i]);
        error = abs_error / max((float)fabs(A[i]), numeric_limits<float>::min());
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
    std::cout << "num_op        : " << n << std::endl;
    std::cout << "max_entry     : " << A[max_idx] << ",  " << B[max_idx] << std::endl;
    std::cout << "max_abs_err   : " << max_abs_error << std::endl;
    std::cout << "max_rel_err   : " << max_error << std::endl;
    std::cout << "correctness   : " << std::boolalpha <<(max_error<5.0e-7) << std::endl << std::endl;
    return max_error;
}

bool mat_equal(fp *A, fp *B, int n)
{
    return max_norm(A, B, n) < SIM_THRES;
}
