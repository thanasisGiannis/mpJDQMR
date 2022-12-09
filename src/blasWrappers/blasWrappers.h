#pragma once
#include <iostream>
#include <string>
#include "../../include/half.hpp"

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

	  void eig(const int n, double *a, const int  lda, 
        double *l, const int numEvals, eigenTarget_t target);

	  /* NEEDED IN MORE THAN ONE PRECISION FOR THE INNER SOLVER */
    /* DOUBLE */
	  void gemm(const char transa, const char  transb,
      const  int m, const int n, const int k, 
      const double alpha, const double* a, const int lda,
      const double* b, const int ldb,
      const double beta, double* c, const int ldc); 

	  double dot(const int dim, const double *x, const int incx,
	      const double *y, const int incy);
	  void axpy(const int dim, const double alpha, 
	      const double *x, const int incx, double *y, const int incy);
	  double nrm2(const int dim, const double *x, const int incx);
	  void scal(const int dim, const double alpha, 
	      double *x, const int incx);

    void geam(const char transa, const char transb,
              int m, int n,
              const double alpha, const double *A, int lda,
              const double beta, const double *B, int ldb,
              double *C, int ldc);
    
    void trsm(const char Layout, const char side,
                               const char uplo, const char transa,
                               const char diag, 
                               const int dim, const int nrhs,
                               const double alpha, 
                               const double *A, const int ldA, 
                               double *B , const int ldB);    

    /* FLOAT */
	  void gemm(const char transa, const char  transb, 
	      const int m, const int n, const int k,
	      const float alpha, const float* a, const int  lda,
	      const float* b, const int ldb,
	      const float beta, float* c, const int ldc);

	  float dot(const int dim, const float *x, const int incx, 
	      const float *y, const int incy);
	  void axpy(const int dim, const float alpha, const float *x, const int incx,
        float *y, const int incy);
	  float nrm2(const int dim, const float *x, const int incx);
	  void scal(const int dim, const float alpha, float *x, const int incx);
	  void geam(const char transa, const char transb,
              int m, int n,
              const float alpha, const float *A, int lda,
              const float beta, const float *B, int ldb,
              float *C, int ldc);
    void trsm(const char Layout, const char side,
                               const char uplo, const char transa,
                               const char diag, 
                               const int dim, const int nrhs,
                               const float alpha, 
                               const float *A, const int ldA, 
                               float *B , const int ldB); 
                                         
	  /* HALF */
	  void gemm(const char transa, const char  transb, 
					  const int M, const int N, const int K,
					  const half_float::half alpha, 
					  const half_float::half* A, const int ldA,
					  const half_float::half* B, const int ldB,
					  const half_float::half beta, half_float::half* C, const int ldC);
	  half_float::half dot(const int dim, 
	      const half_float::half *x, const int incx, 
	      const half_float::half *y, const int incy);
	  void axpy(const int dim, const half_float::half alpha, 
	      const half_float::half *x, const int incx,
	      half_float::half *y, const int incy);
	  half_float::half nrm2(const int dim, 
	      const half_float::half *x, const int incx);
	  void scal(const int dim, const half_float::half alpha, 
	      half_float::half *x, const int incx);
	  void geam(const char transa, const char transb,
              int m, int n,
              const half_float::half alpha,
              const half_float::half *A, int lda,
              const half_float::half beta, 
              const half_float::half *B, int ldb,
              half_float::half *C, int ldc);
    void trsm(const char Layout, const char side,
                               const char uplo, const char transa,
                               const char diag, 
                               const int M, const int N,
                               const half_float::half alpha, 
                               const half_float::half *A, const int ldA, 
                               half_float::half *B , const int ldB); 

  private:
	  void gemmAB(const int M, const int N, const int K,
      					const half_float::half alpha, const half_float::half* A, const int ldA,
				        const half_float::half* B, const int ldB,
				        const half_float::half beta, half_float::half* C, const int ldC);
					      
	  void gemmATB(const int M, const int N, const int K,
				        const half_float::half alpha, const half_float::half* A, const int ldA,
				        const half_float::half* B, const int ldB,
				        const half_float::half beta, half_float::half* C, const int ldC);
		void gemmABT(const int M, const int N, const int K,
                const half_float::half alpha, const half_float::half* A, const int ldA,
                const half_float::half* B, const int ldB,
                const half_float::half beta, half_float::half* C,int ldC);
};
}
//void dot();

