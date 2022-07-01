#pragma once
#include <iostream>
#include <string>
#include "../../include/half.hpp"
//TODO: FILE FORMAT
using half_float::half;

namespace mpjd {

enum class eigenTarget_t {
							/* Target of wanted eigenvalues */
							SM, // smallest by absolute value
							LM // bigest by largest absolute value 
							/* future implementations */
							//SR // smallest real
							//BR // bigest real
							
						};



class LinearAlgebra {
public:
	
	LinearAlgebra();

	std::string getTarget();
	
	
	void eig(int n, double *a, int  lda, double *l, int numEvals, eigenTarget_t target);
	
	
	/* NEEDED IN MORE THAN ONE PRECISION FOR THE INNER SOLVER */
  /* DOUBLE */
	void gemm(char transa, char  transb, 
						int m, int n, int k,
						double alpha,double* a, int  lda,
						double* b, int  ldb,
						double beta, double* c,int ldc);

	double dot(int dim, double *x, int incx, double *y, int incy);
	void   axpy(int dim, double alpha, double *x, int incx, double *y, int incy);
	double nrm2(int dim, double *x,int incx);
	void   scal(int dim, double alpha, double *x, int incx);



  /* FLOAT */
	void gemm(char transa, char  transb, 
						int m, int n, int k,
						float alpha,float* a, int  lda,
						float* b, int  ldb,
						float beta, float* c,int ldc);

	float dot(int dim, float *x, int incx, float *y, int incy);
	void  axpy(int dim, float alpha, float *x, int incx, float *y, int incy);
	float nrm2(int dim, float *x,int incx);
	void  scal(int dim, float alpha, float *x, int incx);
	
	/* HALF */
	void gemm(char transa, char  transb, 
					int M, int N, int K,
					half alpha,half* A, int  ldA,
					half* B, int  ldB,
					half beta, half* C,int ldC);

	half dot(int dim, half *x, int incx, half *y, int incy);
	void  axpy(int dim, half alpha, half *x, int incx, half *y, int incy);
	half nrm2(int dim, half *x,int incx);
	void  scal(int dim, half alpha, half *x, int incx);

	private:
	void gemmAB(int M, int N, int K,
    					half alpha,half* A, int  ldA,
				      half* B, int  ldB,
				      half beta, half* C,int ldC);
					    
	void gemmATB(int M, int N, int K,
				      half alpha,half* A, int  ldA,
				      half* B, int  ldB,
				      half beta, half* C,int ldC);
};
}
//void dot();

