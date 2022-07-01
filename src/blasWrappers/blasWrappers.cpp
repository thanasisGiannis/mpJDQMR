// TODO: ADD const TO INPUT DATA
#include "blasWrappers.h"

#include <iostream>
#include <vector>

#include <mkl.h>
#include <omp.h>

#pragma omp declare reduction(+: half : omp_out += omp_in)

mpjd::LinearAlgebra::LinearAlgebra()
{
	/* future implementation will use target to select cpu or gpu target machine */	
}


void mpjd::LinearAlgebra::eig(int n, double* a, int  lda,  double *l, 
    int numEvals, eigenTarget_t target) {

	//l should contains eigenvalues in descending order
	LAPACKE_dsyev(LAPACK_COL_MAJOR, 'V', 'U', n, a, lda, l );
}

void mpjd::LinearAlgebra::gemm(char transa, char  transb, int m, int n, int k, 
    double alpha, double* a, int lda,
    double* b, int  ldb,
    double beta, double* c, int ldc) {
    
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


double mpjd::LinearAlgebra::dot(int dim, double *x, int incx, 
    double *y, int incy) {
    
	return cblas_ddot(dim, x, incx, y, incy);
}

void mpjd::LinearAlgebra::axpy(int dim, double alpha, double *x, int incx, 
    double *y, int incy) {
    
		cblas_daxpy(dim, alpha, x, incx, y, incy);
}

double mpjd::LinearAlgebra::nrm2(int dim, double *x,int incx) {

	return cblas_dnrm2(dim, x, incx);
}

void mpjd::LinearAlgebra::scal(int dim, double alpha, double *x, int incx) {

	cblas_dscal(dim, alpha, x, incx);
}




void mpjd::LinearAlgebra::gemm(char transa,char  transb, int m, int n, int k, 
    float alpha, float* a, int lda,
    float* b,int  ldb,
    float beta,float* c,int ldc) {
    
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


float mpjd::LinearAlgebra::dot(int dim, float *x, int incx, float *y, int incy) {

	return cblas_sdot(dim, x, incx, y, incy);
}


void mpjd::LinearAlgebra::axpy(int dim, float alpha, float *x, int incx, 
    float *y, int incy) {
    
  cblas_saxpy(dim,alpha,x,incx,y,incy);
}

float mpjd::LinearAlgebra::nrm2(int dim, float *x, int incx) {

	return cblas_snrm2(dim, x, incx);
}

void mpjd::LinearAlgebra::scal(int dim, float alpha, float *x, int incx) {

	cblas_sscal(dim, alpha, x, incx);
}


/* HALF */
void mpjd::LinearAlgebra::gemm(char transa, char  transb, 
    int M, int N, int K,
    half alpha,half* A, int  ldA,
    half* B, int  ldB,
    half beta, half* C,int ldC) {
  
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


void mpjd::LinearAlgebra::gemmAB(int M, int N, int K,
    half alpha,half* A, int  ldA,
    half* B, int  ldB,
    half beta, half* C,int ldC){

  if(M==1 && N ==1){
    C[0] = dot(K, A,1, B, 1);
    return;
  }

  int blockM = std::min(M,2048);
  int blockN = std::min(N,16);
  int blockK = std::min(K,16);

  #pragma omp parallel if(blockM>1 || blockN>1)
  {
    half miniC[blockM*blockN];
    half miniA[blockM*blockK];
    half miniB[blockK*blockN];
    
    #pragma omp for collapse(2) //if(blockM>1 && blockN>1)
    for (int i = 0; i < M; i += blockM) {
      for (int j = 0; j < N; j += blockN) {
      
          /* initialize miniC */
          memset(miniC, 0, sizeof(half)*blockM*blockN);
          
          
          for (int k = 0; k < K; k += blockK) {
            /* prefetch A */
            int sA    = std::min(M-i, blockM);
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
                    sB*sizeof(half));
            }
            
            #pragma omp parallel for collapse(2) if(blockM>1 || blockN>1)
            for (int ii = 0; ii < blockM; ii++) {
              for (int jj = 0; jj < blockN; jj++) {
                half cc = static_cast<half>(0.0);
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
						
void mpjd::LinearAlgebra::gemmATB(int M, int N, int K,
    half alpha, half* A, int  ldA,
    half* B, int  ldB,
    half beta, half* C, int ldC){

  /* 
    Both A and B are accessed by column major 
    dot products should be cache efficient
    using simd and reduction in order to accelerate
  */
  #pragma omp parallel for collapse(2) if(M>1 || N>1)
  for (int jj = 0; jj < N; jj++) {
     for (int ii = 0; ii < M; ii++) {
          half cc = static_cast<half>(0.0);// C[ii+jj*ldC];
          
          //#pragma omp simd reduction(+:cc)
          #pragma omp simd reduction(+:cc)
          for (int kk = 0; kk < K; kk++) {
              cc += A[kk+ii*ldA]*B[kk+jj*ldB] ;
          }
          C[ii+jj*ldC] = beta*C[ii+jj*ldC] + alpha*cc;
      }
  }
}

half mpjd::LinearAlgebra::dot(int dim, half *x, int incx, half *y, int incy){

  half res=static_cast<half>(0.0);
  #pragma omp simd reduction(+:res)
  for (int i = 0; i < dim; i++) {
    res += x[i]*y[i];
  }

  return res;
}

void mpjd::LinearAlgebra::axpy(int dim, half alpha, half *x, int incx,
    half *y, int incy){
    
  //y = y + alpha * x
  #pragma omp simd
  for(int i=0;i<dim;i++){
      y[i] += alpha*x[i];
  }

}

half mpjd::LinearAlgebra::nrm2(int dim, half *x, int incx){ 

  half res = dot(dim, x, incx, x, incx);
  return half_float::sqrt(res);
}

void mpjd::LinearAlgebra::scal(int dim, half alpha, half *x, int incx){

  //#pragma omp parallel for
  for (int i = 0; i < dim; i++) {
      x[i] *= alpha;
  }
}






















