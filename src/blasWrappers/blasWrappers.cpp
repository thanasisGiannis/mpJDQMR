#include <iostream>
#include <mkl.h>
#include "blasWrappers.h"

mpjd::LinearAlgebra::LinearAlgebra()
{
	/* future implementation will use target to select cpu or gpu target machine */	
}

void mpjd::LinearAlgebra::gemm(char transa,char  transb, int m, int n, int k, 
											double alpha, double* a, int lda,
											double* b,int  ldb,
											double beta,double* c,int ldc){
	CBLAS_TRANSPOSE ta;										
	switch (transa){
		case 'T':
			ta = CblasTrans;
			break;
		case 'N':
			ta = CblasNoTrans;
			break;
		default:
			exit(-1);
	}

	CBLAS_TRANSPOSE tb;										
	switch (transb){
		case 'T':
			tb = CblasTrans;
			break;
		case 'N':
			tb = CblasNoTrans;
			break;
		default:
		  std::cout << "Fault" << std::endl;
			exit(-1);
	}

//	std::cout << "== " << ta << " " << tb << std::endl;
	MKL_INT mm = static_cast<MKL_INT>(m);
	MKL_INT nn = static_cast<MKL_INT>(n);
	MKL_INT kk = static_cast<MKL_INT>(k);

	cblas_dgemm(CblasColMajor, ta, tb,mm,nn,kk, alpha,a,lda, b,ldb,beta, c,ldc);

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


void mpjd::LinearAlgebra::eig(int n, double* a, int  lda,  double *l, int numEvals, eigenTarget_t target){

	//l should contains eigenvalues in descending order
	LAPACKE_dsyev(LAPACK_COL_MAJOR, 'V', 'U', n, a, lda, l );
}


