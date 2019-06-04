#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <iostream>
#include <stdlib.h>
#include <iomanip>
#include <cstring>


double time_elapsed(struct timespec &, struct timespec &);
double *init_zeros_mat(int);
double *init_rand_mat(int, int);
double *init_zeros_vec(int);
double *init_rand_vec(int, int);
void print_mat(double *, int);
void print_vec(double *, int);
double *cp_mat(double *, int);
double *cp_vec(double *, int);


