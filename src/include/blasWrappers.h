#pragma once
#include <iostream>

double dot(int dim, double *x, int incx, double *y, int incy);
void axpy(int dim, double alpha, double *x, int incx, double *y, int incy);
double nrm2(int dim, double *x,int incx);
void scal(int dim, double alpha, double *x, int incx);
//void dot();

