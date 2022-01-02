#include <iostream>
#include <mkl.h>
#include "blasWrappers.h"

mpjd::LinearAlgebra::LinearAlgebra(target_t target_) 
: target(target_)
{
	/* future implementation will use target to select cpu or gpu target machine */	
}


double mpjd::LinearAlgebra::dot(int dim, double *x, int incx, double *y, int incy){
	return cblas_ddot(dim,x,incx,y,incy);
	
}

void mpjd::LinearAlgebra::axpy(int dim, double alpha, double *x, int incx, double *y, int incy){
		cblas_daxpy(dim,alpha,x,incx,y,incy);
}

double mpjd::LinearAlgebra::nrm2(int dim, double *x,int incx){

	return cblas_dnrm2(dim,x,incx);
}

void mpjd::LinearAlgebra::scal(int dim, double alpha, double *x, int incx){
	cblas_dscal(dim,alpha,x,incx);

}

std::string mpjd::LinearAlgebra::getTarget(){
	if(target == target_t::HOST){
		return "HOST";
	}else {
		return "Nothing";
	}
}

