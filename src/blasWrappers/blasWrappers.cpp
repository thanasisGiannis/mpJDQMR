#include "blasWrappers.h"

#include <iostream>
#include <vector>

#include <mkl.h>
#include <omp.h>

#define UNUSED(expr) static_cast<void>(expr)

#pragma omp declare reduction(+: half_float::half : omp_out += omp_in)


mpjd::LinearAlgebra::LinearAlgebra()
{
	/* TODO:future implementation will use target to select cpu or gpu target machine */	
}


void mpjd::LinearAlgebra::eig(const int n, double* a, const int  lda,  double *l, 
    const int numEvals, eigenTarget_t target) {

  UNUSED(numEvals);
  UNUSED(target);
  
	//l should contains eigenvalues in descending order
	LAPACKE_dsyev(LAPACK_COL_MAJOR, 'V', 'U', n, a, lda, l );


}

void mpjd::LinearAlgebra::gemm(const char transa, const char  transb,
    const  int m, const int n, const int k, 
    const double alpha, const double* a, const int lda,
    const double* b, const int ldb,
    const double beta, double* c, const int ldc) {
    
	CBLAS_TRANSPOSE ta;										
	switch (transa) {
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
	switch (transb) {
		case 'T':
			tb = CblasTrans;
			break;
		case 'N':
			tb = CblasNoTrans;
			break;
		default:
			exit(-1); // error
	}

	MKL_INT mm = static_cast<MKL_INT>(m);
	MKL_INT nn = static_cast<MKL_INT>(n);
	MKL_INT kk = static_cast<MKL_INT>(k);

	cblas_dgemm(CblasColMajor, ta, tb, mm, nn, kk, alpha, a, lda, 
	    b, ldb, beta, c, ldc);
}


double mpjd::LinearAlgebra::dot(const int dim, const double *x, const int incx, 
    const double *y, const int incy) {
    
	return cblas_ddot(dim, x, incx, y, incy);
}

void mpjd::LinearAlgebra::axpy(const int dim, const double alpha, 
    const double *x, int incx, double *y, const int incy) {
    
		cblas_daxpy(dim, alpha, x, incx, y, incy);
}

double mpjd::LinearAlgebra::nrm2(const int dim, const double *x, const int incx) {

	//return cblas_dnrm2(dim, x, incx);
	MKL_INT dim_  = static_cast<MKL_INT>(dim);
	MKL_INT incx_ = static_cast<MKL_INT>(incx);
	return cblas_dnrm2(dim_, x, incx_);
}

void mpjd::LinearAlgebra::scal(const int dim, const double alpha, 
    double *x, const int incx) {

	cblas_dscal(dim, alpha, x, incx);
}




void mpjd::LinearAlgebra::gemm(const char transa, const char  transb,
    const int m, const int n, const int k, 
    const float alpha, const float* a, const int lda,
    const float* b, const int  ldb,
    const float beta, float* c, int ldc) {
    
	CBLAS_TRANSPOSE ta;										
	switch (transa) {
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
			exit(-1);
	}

	MKL_INT mm = static_cast<MKL_INT>(m);
	MKL_INT nn = static_cast<MKL_INT>(n);
	MKL_INT kk = static_cast<MKL_INT>(k);

	cblas_sgemm(CblasColMajor, ta, tb, mm, nn, kk, alpha, a, lda, 
	    b, ldb, beta, c, ldc);
}


float mpjd::LinearAlgebra::dot(const int dim, const float *x, const int incx,
    const float *y, const int incy) {

	return cblas_sdot(dim, x, incx, y, incy);
}


void mpjd::LinearAlgebra::axpy(const int dim, const float alpha, 
    const float *x, const int incx, float *y, const int incy) {
    
  cblas_saxpy(dim,alpha,x,incx,y,incy);
}

float mpjd::LinearAlgebra::nrm2(const int dim, const float *x, const int incx) {

	MKL_INT dim_  = static_cast<MKL_INT>(dim);
	MKL_INT incx_ = static_cast<MKL_INT>(incx);
	return cblas_snrm2(dim_, x, incx_);
}

void mpjd::LinearAlgebra::scal(const int dim, const float alpha, 
    float *x, const int incx) {

	cblas_sscal(dim, alpha, x, incx);
}


/* HALF */
void mpjd::LinearAlgebra::gemm(const char transa, const char  transb, 
    const int M, const int N, const int K,
    const half_float::half alpha, const half_float::half* A, const int ldA,
    const half_float::half* B, const int  ldB,
    const half_float::half beta, half_float::half* C, int ldC) {
  
  if (transb == 'T') {
      std::cout << "ERROR GEMM" << std::endl;
      exit(0);
  }

  switch (transa)
  {
    case 'N':
      gemmAB(M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
      break;
    case 'T':
      gemmATB(M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
      break;
    default:
      exit(0);
  }
}


void mpjd::LinearAlgebra::gemmAB(const int M, const int N, const int K,
    const half_float::half alpha, const half_float::half* A, const int ldA,
    const half_float::half* B, const int ldB,
    const half_float::half beta, half_float::half* C,int ldC) {

  if(M==1 && N ==1){
    C[0] = dot(K, A,1, B, 1);
    return;
  }

  int blockM = std::min(M,2048);
  int blockN = std::min(N,16);
  int blockK = std::min(K,16);

  #pragma omp parallel if(blockM>1 || blockN>1)
  {
    half_float::half miniC[blockM*blockN];
    half_float::half miniA[blockM*blockK];
    half_float::half miniB[blockK*blockN];
    
    #pragma omp for collapse(2) //if(blockM>1 && blockN>1)
    for (int i = 0; i < M; i += blockM) {
      for (int j = 0; j < N; j += blockN) {
      
          /* initialize miniC */
          memset(miniC, 0, sizeof(half_float::half)*blockM*blockN);
          
          
          for (int k = 0; k < K; k += blockK) {
            /* prefetch A */
            //int sA    = std::min(M-i, blockM);
            int kkEnd = std::min(blockK, K-k);
            int iiEnd = std::min(blockM, M-i);
            
            for (int ii = 0; ii < iiEnd; ii++) {
              #pragma omp simd
              for (int kk = 0; kk < kkEnd; kk++){
                miniA[kk + ii*blockK ] = A[(i+ii)+(k+kk)*ldA];
              }
            }
            
            /* prefetch B */
            int sB    = std::min(K-k, blockK);
            int jjEnd = std::min(blockN, N-j);

            #pragma omp simd
            for (int jj = 0; jj < jjEnd; jj++) {
                memcpy(&miniB[0 + jj*blockM], &B[k+(j+jj)*ldB],
                    sB*sizeof(half_float::half));
            }
            
            #pragma omp parallel for collapse(2) if(blockM>1 || blockN>1)
            for (int ii = 0; ii < blockM; ii++) {
              for (int jj = 0; jj < blockN; jj++) {
                half_float::half cc = static_cast<half_float::half>(0.0);
                #pragma omp simd reduction (+:cc)
                for (int kk = 0; kk < blockK; kk++) {
                       cc += alpha*miniA[ii*blockK+kk]*miniB[kk+jj*blockK];
                  }
                miniC[ii+jj*blockM] = cc;  
              }
            }
          }// for (int k = 0; k < K; k += blockK) 
          
          
          /* store miniC to C */
          for (int jj = 0; (jj < blockN) & (j+jj < N); jj++) {
            auto iiEnd = std::min(blockM,M-i);
            #pragma omp simd
            for (int ii = 0; ii < iiEnd; ii++) {
                C[i+ii+(j+jj)*ldC] = beta*C[i+ii + (j+jj)*ldC];
            }
            #pragma omp simd
            for (int ii = 0; ii < iiEnd; ii++) {
                C[i+ii+(j+jj)*ldC] += miniC[ii+jj*blockM];
            }
          }
      } // for (int j = 0; j < N; j += blockN)
    } // for (int i = 0; i < M; i += blockM)
  } //  #pragma omp parallel if(blockM>1 || blockN>1)        
}
						
void mpjd::LinearAlgebra::gemmATB(const int M, const int N, const int K,
    const half_float::half alpha, const half_float::half* A, const int ldA,
    const half_float::half* B, const int ldB,
    const half_float::half beta, half_float::half* C, int ldC) {

  /* 
    Both A and B are accessed by column major 
    dot products should be cache efficient
    using simd and reduction in order to accelerate
  */
  #pragma omp parallel for collapse(2) if(M>1 || N>1)
  for (int jj = 0; jj < N; jj++) {
     for (int ii = 0; ii < M; ii++) {
          half_float::half cc = static_cast<half_float::half>(0.0);// C[ii+jj*ldC];
          
          //#pragma omp simd reduction(+:cc)
          #pragma omp simd reduction(+:cc)
          for (int kk = 0; kk < K; kk++) {
              cc += A[kk+ii*ldA]*B[kk+jj*ldB] ;
          }
          C[ii+jj*ldC] = beta*C[ii+jj*ldC] + alpha*cc;
      }
  }
}

half_float::half mpjd::LinearAlgebra::dot(const int dim, 
    const half_float::half *x, const int incx,
    const half_float::half *y, const int incy) {

  half_float::half res=static_cast<half_float::half>(0.0);
  #pragma omp simd reduction(+:res)
  for (int i = 0; i < dim; i++) {
    res += x[i*incx]*y[i*incy];
  }

  return res;
}

void mpjd::LinearAlgebra::axpy(const int dim, const half_float::half alpha, 
    const half_float::half *x, const int incx, 
    half_float::half *y, const int incy) {
    
  //y = y + alpha * x
  #pragma omp simd
  for(int i=0;i<dim;i++){
      y[i*incy] += alpha*x[i*incx];
  }
}

half_float::half mpjd::LinearAlgebra::nrm2(const int dim, 
    const half_float::half *x, const int incx) { 

  half_float::half res = dot(dim, x, incx, x, incx);
  return half_float::sqrt(res);
}

void mpjd::LinearAlgebra::scal(const int dim, const half_float::half alpha, 
    half_float::half *x, const int incx) {

  //#pragma omp parallel for
  for (int i = 0; i < dim; i++) {
      x[i*incx] *= alpha;
  }
}






















