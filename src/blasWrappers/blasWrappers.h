#pragma once
#include <iostream>
#include <string>
namespace mpjd {

enum class eigenTarget_t {
							/* Target of wanted eigenvalues */
							SM // smallest by absolute value
							/* future implementations */
							//BM // bigest by absolute value 
							//SR // smallest real
							//BR // bigest real
							
						};

enum class target_t
		 {  
		 		// following future targets 
		 		// DEVICE, //[=GPU]
		 		HOST
		 	};


class LinearAlgebra {

	target_t target;	
public:
	
	LinearAlgebra(target_t target_);

	std::string getTarget();
	
	void gemm(char transa, char  transb, 
						int m, int n, int k,
						double alpha,double* a, int  lda,
						double* b, int  ldb,
						double beta, double* c,int ldc);
						
	double dot(int dim, double *x, int incx, double *y, int incy);
	void axpy(int dim, double alpha, double *x, int incx, double *y, int incy);
	double nrm2(int dim, double *x,int incx);
	void scal(int dim, double alpha, double *x, int incx);
	void eig(int n, double *a, int  lda,int numEvals, eigenTarget_t target);
	
};
}
//void dot();

