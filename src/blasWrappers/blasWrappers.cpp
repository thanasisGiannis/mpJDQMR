#include "../include/blasWrappers.h"
#include <iostream>
#include <mkl.h>

double dot(int dim, double *x, int incx, double *y, int incy){
	//return cblas_ddot(dim,x,incx,y,incy);
	//std::cout << "Hello from dot!-()" << std::endl; return 0.1;
	
	return cblas_ddot(dim,x,incx,y,incy);
	
}

void axpy(int dim, double alpha, double *x, int incx, double *y, int incy){
		cblas_daxpy(dim,alpha,x,incx,y,incy);
}

double nrm2(int dim, double *x,int incx){

	return cblas_dnrm2(dim,x,incx);
}

void scal(int dim, double alpha, double *x, int incx){
	cblas_dscal(dim,alpha,x,incx);

}

