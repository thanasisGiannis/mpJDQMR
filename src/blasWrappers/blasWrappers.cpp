#include "blasWrappers.h"

#include <iostream>
#include <vector>

#include <mkl.h>
#include <mkl_trans.h>
#include <omp.h>

#define UNUSED(expr) static_cast<void>(expr)

#pragma omp declare reduction(+: half_float::half : omp_out += omp_in)


mpjd::LinearAlgebra::LinearAlgebra()
{
    /* TODO:future implementation
     *  will use target to select cpu or gpu target machine
    */
}


void mpjd::LinearAlgebra::eig(const int n, double* a, const int  lda,
                              double *l, const int numEvals,
                              eigenTarget_t target)
{

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

void mpjd::LinearAlgebra::geam(const char transa, const char transb,
              int m, int n,
              const double alpha, const double *A, int lda,
              const double beta, const double *B, int ldb,
              double *C, int ldc) {

    MKL_INT mm = static_cast<MKL_INT>(m);
    MKL_INT nn = static_cast<MKL_INT>(n);

    mkl_domatadd('c', transa, transb, mm, nn, alpha, A, lda, beta, B, ldb, C, ldc);

}


void mpjd::LinearAlgebra::trsm(const char Layout, const char side,
                               const char uplo, const char transa,
                               const char diag,
                               const int dim, const int nrhs,
                               const double alpha,
                               const double *A, const int ldA,
                               double *B , const int ldB ) {

  CBLAS_LAYOUT cblas_Layout;
  CBLAS_SIDE cblas_side;
  CBLAS_UPLO cblas_uplo;
  CBLAS_TRANSPOSE cblas_transa;
  CBLAS_DIAG cblas_diag;

    switch (Layout) {
        case 'C':
            cblas_Layout = CblasColMajor;
            break;
        case 'R':
            cblas_Layout = CblasRowMajor;
            break;
        default:
            exit(-1); // error
    }


    switch (side) {
        case 'L':
            cblas_side = CblasLeft;
            break;
        case 'R':
            cblas_side = CblasRight;
            break;
        default:
            exit(-1); // error
    }


    switch (uplo) {
        case 'U':
            cblas_uplo = CblasUpper;
            break;
        case 'L':
            cblas_uplo = CblasLower;
            break;
        default:
            exit(-1); // error
    }

    switch (transa) {
        case 'N':
            cblas_transa = CblasNoTrans;
            break;
        case 'T':
            cblas_transa = CblasTrans;
            break;
        default:
            exit(-1); // error
    }

    switch (diag) {
        case 'N':
            cblas_diag = CblasNonUnit;
            break;
        case 'U':
            cblas_diag = CblasUnit;
            break;
        default:
            exit(-1); // error
    }


  MKL_INT mkl_dim  = static_cast<MKL_INT>(dim);
  MKL_INT mkl_nrhs = static_cast<MKL_INT>(nrhs);
  MKL_INT mkl_ldA  = static_cast<MKL_INT>(ldA);
  MKL_INT mkl_ldB  = static_cast<MKL_INT>(ldB);

  cblas_dtrsm(cblas_Layout, cblas_side, cblas_uplo, cblas_transa, cblas_diag,
              mkl_dim, mkl_nrhs, alpha, A, mkl_ldA, B, mkl_ldB);

}


/* FLOAT */
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

void mpjd::LinearAlgebra::geam(const char transa, const char transb,
              int m, int n,
              const float alpha, const float *A, int lda,
              const float beta, const float *B, int ldb,
              float *C, int ldc) {

    MKL_INT mm = static_cast<MKL_INT>(m);
    MKL_INT nn = static_cast<MKL_INT>(n);

    mkl_somatadd('c', transa, transb, mm, nn, alpha, A, lda, beta, B, ldb, C, ldc);

}



void mpjd::LinearAlgebra::trsm(const char Layout, const char side,
                               const char uplo, const char transa,
                               const char diag,
                               const int dim, const int nrhs,
                               const float alpha,
                               const float *A, const int ldA,
                               float *B , const int ldB ) {

  CBLAS_LAYOUT cblas_Layout;
  CBLAS_SIDE cblas_side;
  CBLAS_UPLO cblas_uplo;
  CBLAS_TRANSPOSE cblas_transa;
  CBLAS_DIAG cblas_diag;

    switch (Layout) {
        case 'C':
            cblas_Layout = CblasColMajor;
            break;
        case 'R':
            cblas_Layout = CblasRowMajor;
            break;
        default:
            exit(-1); // error
    }


    switch (side) {
        case 'L':
            cblas_side = CblasLeft;
            break;
        case 'R':
            cblas_side = CblasRight;
            break;
        default:
            exit(-1); // error
    }


    switch (uplo) {
        case 'U':
            cblas_uplo = CblasUpper;
            break;
        case 'L':
            cblas_uplo = CblasLower;
            break;
        default:
            exit(-1); // error
    }

    switch (transa) {
        case 'N':
            cblas_transa = CblasNoTrans;
            break;
        case 'T':
            cblas_transa = CblasTrans;
            break;
        default:
            exit(-1); // error
    }

    switch (diag) {
        case 'N':
            cblas_diag = CblasNonUnit;
            break;
        case 'U':
            cblas_diag = CblasUnit;
            break;
        default:
            exit(-1); // error
    }


  MKL_INT mkl_dim  = static_cast<MKL_INT>(dim);
  MKL_INT mkl_nrhs = static_cast<MKL_INT>(nrhs);
  MKL_INT mkl_ldA  = static_cast<MKL_INT>(ldA);
  MKL_INT mkl_ldB  = static_cast<MKL_INT>(ldB);

  cblas_strsm(cblas_Layout, cblas_side, cblas_uplo, cblas_transa, cblas_diag,
              mkl_dim, mkl_nrhs, alpha, A, mkl_ldA, B, mkl_ldB);

}




/* HALF */
void mpjd::LinearAlgebra::gemm(const char transa, const char  transb,
    const int M, const int N, const int K,
    const half_float::half alpha, const half_float::half* A, const int ldA,
    const half_float::half* B, const int  ldB,
    const half_float::half beta, half_float::half* C, const int ldC) {


  if( transa == 'N' && transb == 'N' ) {
    gemmAB(M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
  } else if( transa == 'N' && transb == 'T') {
    gemmABT(M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
  } else if( transa == 'T' && transb == 'N' ) {
    gemmATB(M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
  } else {
    std::cout << "ERROR GEMM" << std::endl;
    exit(0);
 }

}

void mpjd::LinearAlgebra::gemmABT(const int M, const int N, const int K,
    const half_float::half alpha, const half_float::half* A, const int ldA,
    const half_float::half* B, const int ldB,
    const half_float::half beta, half_float::half* C, const int ldC) {
// TODO: Find a better performance wise
//       implementation of this mat-mat

  //  #pragma omp parallel for collapse(2) if(M>1 && N>1)
  for(int i=0; i<M; i++){
    for(int j=0; j<N; j++){

     half_float::half c = static_cast<half_float::half>(0.0);
      for(int k=0; k<K; k++){
         c = c + A[i+k*ldA]*B[j+k*ldB];
      }
      C[i+j*ldC] = beta*C[i+j*ldC] + alpha*c;

    }
  }
}

void mpjd::LinearAlgebra::gemmAB(const int M, const int N, const int K,
    const half_float::half alpha, const half_float::half* A, const int ldA,
    const half_float::half* B, const int ldB,
    const half_float::half beta, half_float::half* C, const int ldC) {


// TODO: Find a better performance wise
//       implementation of this mat-mat

 //#pragma omp parallel for collapse(2) if(M>1 && N>1)
 for(int i=0; i<M; i++){
    for(int j=0; j<N; j++){
      half_float::half c = static_cast<half_float::half>(0.0);
      for(int k=0; k<K; k++){
         c = c + A[i+k*ldA]*B[k+j*ldB];
      }
      C[i+j*ldC] = beta*C[i+j*ldC] + alpha*c;
    }
  }

}
void mpjd::LinearAlgebra::gemmATB(const int M, const int N, const int K,
    const half_float::half alpha, const half_float::half* A, const int ldA,
    const half_float::half* B, const int ldB,
    const half_float::half beta, half_float::half* C, const int ldC) {

  /*
    Both A and B are accessed by column major
    dot products should be cache efficient
  */

  //#pragma omp parallel for collapse(2) if(M>1 && N>1)
  for(int i=0; i<M; i++){
    for(int j=0; j<N; j++){

     half_float::half c = static_cast<half_float::half>(0.0);
      for(int k=0; k<K; k++){
         c = c + A[k+i*ldA]*B[k+j*ldB];
      }
      C[i+j*ldC] = beta*C[i+j*ldC] + alpha*c;
    }
  }

}


half_float::half mpjd::LinearAlgebra::dot(const int dim,
    const half_float::half *x, const int incx,
    const half_float::half *y, const int incy) {

  half_float::half res = static_cast<half_float::half>(0.0);
  #pragma omp parallel for reduction(+:res) num_threads(256)
  for (int i = 0; i < dim; i++) {
    res += x[i*incx]*y[i*incy];
  }
  return res;
}

void mpjd::LinearAlgebra::axpy(const int dim, const half_float::half alpha,
    const half_float::half *x, const int incx,
    half_float::half *y, const int incy) {

  #pragma omp parallel for num_threads(256)
        for(int i=0;i<dim;i++){
      y[i*incy] += alpha*x[i*incx];
  }
}

half_float::half mpjd::LinearAlgebra::nrm2(const int dim,
    const half_float::half *x, const int incx) {

        /*
                This is not the proper way to calculate nrm2
                This is prone to some bad floating point arithmetic errors
                Need a better algorithm. This maybe a problem to the fast convergance
                of some algorithms
                Calculated norm does not approach the actual value by fp error standards
         */
  half_float::half res = dot(dim, x, incx, x, incx);
  return half_float::sqrt(res);
}

void mpjd::LinearAlgebra::scal(const int dim, const half_float::half alpha,
    half_float::half *x, const int incx) {

  #pragma omp parallel for num_threads(256)
  for (int i = 0; i < dim; i++) {
      x[i*incx] *= alpha;
  }
}

void mpjd::LinearAlgebra::geam(const char transa, const char transb,
          int m, int n,
          const half_float::half  alpha, const half_float::half *A, int lda,
          const half_float::half  beta, const half_float::half *B, int ldb,
          half_float::half *C, int ldc) {
        static int blockI = 8;
        static int blockJ = 8;

  if( transa == 'N' && transb == 'N') {

    #pragma omp parallel for collapse(2) num_threads(256)
    for (int i=0; i<m; i++) {
      for (int j=0; j<n; j++) {
          C[i+j*ldc] = alpha*A[i+j*lda] + beta*B[i+j*ldb];
      }
    }
  } else if(transa == 'T' && transb == 'N' ) {

    #pragma omp parallel for collapse(2) num_threads(256)
    for (int i=0; i<m; i+=blockI) {
      for (int j=0; j<n; j+=blockJ) {
                                for(int ii=0; ii<blockI && i+ii<m; ii++) {
                                        for(int jj=0; jj<blockJ && j+jj<n; jj++) {
                                                    C[i+ii+(j+jj)*ldc] = alpha*A[(j+jj)+(i+ii)*lda] + beta*B[i+ii+(j+jj)*ldb];
                                        }
                                }
      }
    }
  } else if (transa == 'N' && transb == 'T' ) {

    #pragma omp parallel for collapse(2) num_threads(256)
    for (int i=0; i<m; i++) {
      for (int j=0; j<n; j++) {
          C[i+j*ldc] = alpha*A[i+j*lda] + beta*B[j+i*ldb];
      }
    }
  } else if (transa == 'T' && transb == 'T' ) {

    #pragma omp parallel for collapse(2) num_threads(256)
    for (int j=0; j<n; j++) {
            for (int i=0; i<m; i++) {
          C[i+j*ldc] = alpha*A[j+i*lda] + beta*B[j+i*ldb];
      }
    }
  } else {
    exit(-1);
  }


}



void mpjd::LinearAlgebra::trsm(const char Layout, const char side,
                           const char uplo, const char transa,
                           const char diag,
                           const int M, const int N,
                           const half_float::half alpha,
                           const half_float::half *A, const int ldA,
                           half_float::half *B , const int ldB ) {

  if(Layout == 'C' &&
     side   == 'R' &&
     uplo   == 'U' &&
     transa == 'N' &&
     diag   == 'N') {
     // WE ACTUALLY USE THIS SETTING INSIDE TH
     // BLOCK SQMR - IF ANYTHING CHANGES IN THE
     // FUTURE- MORE SHOULD BE IMPLEMENTED
     // A = N x N
     // B = M x N

    for( int i=0; i<N; i++ ) {
      for( int j=i-1; j>=0; j-- ) {
        // b(:,i) = b(:,i) - A(j,i).*b(:,j);
        axpy(M, -A[j+i*ldA], &B[0+j*ldB], 1,&B[0+i*ldB], 1);
      }
      //b(:,i) = b(:,i)./A(i,i);
      scal(M, alpha/A[i+i*ldA], &B[0+i*ldB], 1);
    }
  } else {
    exit(-1);
  }

}





















