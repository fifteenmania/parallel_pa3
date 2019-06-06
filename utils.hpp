#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <iostream>
#include <stdlib.h>
#include <iomanip>
#include <cstring>
#include <math.h>
#define SIM_THRES 5.0e-7

typedef float fp;

double time_elapsed(struct timespec &, struct timespec &);
fp *init_zeros_mat(int);
fp *init_rand_mat(int, int);
fp *init_ones_mat(int);
fp *init_zeros_vec(int);
fp *init_rand_vec(int, int);
void print_mat(fp *, int);
void print_vec(fp *, int);
fp *cp_mat(fp *, int);
fp *cp_vec(fp *, int);
fp l2_norm(fp *, fp *, int);
fp max_norm(fp *, fp *, int);
bool mat_equal(fp *, fp *, int);

