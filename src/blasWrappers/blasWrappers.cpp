#include <iostream>
#include <mkl.h>
#include "blasWrappers.h"
#include <vector>

mpjd::LinearAlgebra::LinearAlgebra()
{
	/* future implementation will use target to select cpu or gpu target machine */	
}


void mpjd::LinearAlgebra::eig(int n, double* a, int  lda,  double *l, int numEvals, eigenTarget_t target){

	//l should contains eigenvalues in descending order
	LAPACKE_dsyev(LAPACK_COL_MAJOR, 'V', 'U', n, a, lda, l );
}

/* NEEDED IN MORE THAN ONE PRECISION FOR THE INNER SOLVER */

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
		  //std::cout << "Fault" << std::endl;
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




void mpjd::LinearAlgebra::gemm(char transa,char  transb, int m, int n, int k, 
											float alpha, float* a, int lda,
											float* b,int  ldb,
											float beta,float* c,int ldc){
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
		  //std::cout << "Fault" << std::endl;
			exit(-1);
	}

//	std::cout << "== " << ta << " " << tb << std::endl;
	MKL_INT mm = static_cast<MKL_INT>(m);
	MKL_INT nn = static_cast<MKL_INT>(n);
	MKL_INT kk = static_cast<MKL_INT>(k);

	cblas_sgemm(CblasColMajor, ta, tb,mm,nn,kk, alpha,a,lda, b,ldb,beta, c,ldc);

}


float mpjd::LinearAlgebra::dot(int dim, float *x, int incx, float *y, int incy){
	return cblas_sdot(dim,x,incx,y,incy);
	
}

void mpjd::LinearAlgebra::axpy(int dim, float alpha, float *x, int incx, float *y, int incy){
		cblas_saxpy(dim,alpha,x,incx,y,incy);
}

float mpjd::LinearAlgebra::nrm2(int dim, float *x,int incx){

	return cblas_snrm2(dim,x,incx);
}

void mpjd::LinearAlgebra::scal(int dim, float alpha, float *x, int incx){
	cblas_sscal(dim,alpha,x,incx);

}


/* HALF */
// TODO: add declarations to those functions
void mpjd::LinearAlgebra::gemm(char transa, char  transb, 
					int M, int N, int K,
					half alpha,half* A, int  ldA,
					half* B, int  ldB,
					half beta, half* C,int ldC){
  
    if(transb == 'T'){
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

    int blockM = std::min(M,100);
    int blockN = std::min(N,100);
    int blockK = std::min(K,100);

     #pragma omp parallel       
     {
                
        half miniC[blockM*blockN];
        half miniA[blockM*blockK];
        half miniB[blockK*blockN];
        
        #pragma omp parallel for collapse(2) 
        for(int i=0;i<M;i+=blockM){
          for(int j=0;j<N;j+=blockN){
          
              /* initialize miniC */
              #pragma omp parallel for 
              for(int ii=0; ii<blockM; ii++){
                for(int jj=0; jj<blockN; jj++){
                    miniC[ii+jj*blockM] = static_cast<half>(0.0);
                    
                }
              }
              
              
              for(int k=0;k<K;k+=blockK){
              
                  /* prefetch A */
                  #pragma omp parallel for 
                  for(int kk=0; kk<blockK; kk++){
                      for(int ii=0; ii<blockM; ii++){
                          if( (i+ii < M) && (k+kk < K))
                            miniA[ii + kk*blockM] = A[i+ii + (k+kk)*ldA];
                          else
                            break;//miniA[ii + kk*blockM] = static_cast<half>(0.0);
                            
                      }
                  }
                  /* prefetch B */
                  #pragma omp parallel for 
                  for(int jj=0; jj<blockN; jj++){
                      for(int kk=0; kk<blockK; kk++){
                          if( (k+kk < K) && (j+jj < N))
                              miniB[kk + jj*blockK] = B[k+kk + (j+jj)*ldB];
                          else
                            break;//miniB[kk + jj*blockK] = static_cast<half>(0.0);    
                      }
                  }
        
                  /* multiply */
                  #pragma omp parallel for collapse(2) 
                  for(int ii=0; ii<std::min(M,blockM); ii++){
                     for(int kk=0; kk< std::min(blockK,K); kk++){
                          half aa = miniA[ii+kk*blockM];
                          for(int jj=0; jj< std::min(blockN,N); jj++){
                              miniC[ii+jj*blockM] += alpha*aa*miniB[kk+jj*blockK];
                          }
                      }
                  }
    
              }
              
              
              /* store miniC to C */
              #pragma omp parallel for 
              for(int jj=0; jj<blockN; jj++){
                for(int ii=0; ii<blockM; ii++){
                    if( (i+ii < M) && ( j+jj<N))
                        C[i+ii+(j+jj)*ldC] = miniC[ii+jj*blockM] + beta*C[i+ii + (j+jj)*ldC];
                    else
                      break;
                }
              }

             
          }
        }

    }        
}
						
void mpjd::LinearAlgebra::gemmATB(int M, int N, int K,
    					half alpha,half* A, int  ldA,
				      half* B, int  ldB,
				      half beta, half* C,int ldC){
    int blockM = std::min(M,100);
    int blockN = std::min(N,100);
    int blockK = std::min(K,100);

     #pragma omp parallel       
     {
                
        half miniC[blockM*blockN];
        half miniA[blockM*blockK];
        half miniB[blockK*blockN];
        
        #pragma omp parallel for collapse(2) 
        for(int i=0;i<M;i+=blockM){
          for(int j=0;j<N;j+=blockN){
          
              /* initialize miniC */
              #pragma omp parallel for 
              for(int ii=0; ii<blockM; ii++){
                for(int jj=0; jj<blockN; jj++){
                    miniC[ii+jj*blockM] = static_cast<half>(0.0);
                    
                }
              }
              
              
              for(int k=0;k<K;k+=blockK){
              
                  /* prefetch A */
                  #pragma omp parallel for 
                  for(int kk=0; kk<blockK; kk++){
                      for(int ii=0; ii<blockM; ii++){
                          if( (i+ii < M) && (k+kk < K))
                            miniA[kk + ii*blockM] = A[k+kk + (i+ii)*ldA];
                          else
                            break;//miniA[kk + ii*blockM] = static_cast<half>(0.0);
                            
                      }
                  }
                  /* prefetch B */
                  #pragma omp parallel for 
                  for(int jj=0; jj<blockN; jj++){
                      for(int kk=0; kk<blockK; kk++){
                          if( (k+kk < K) && (j+jj < N))
                            miniB[kk + jj*blockK] = B[k+kk + (j+jj)*ldB];
                          else
                            break;//miniB[kk + jj*blockK] = static_cast<half>(0.0);    
                      }
                  }
        
                  /* multiply */
                  #pragma omp parallel for collapse(2) 
                  for(int ii=0; ii<blockM; ii++){
                     for(int kk=0; kk<std::min(K,blockK); kk++){
                          half aa = miniA[kk+ii*blockM];
                          for(int jj=0; jj<std::min(N,blockN); jj++){
                              miniC[ii+jj*blockM] += alpha*aa*miniB[kk+jj*blockK];
                          }
                      }
                  }
    
              }
              
              
              /* store miniC to C */
              #pragma omp parallel for 
              for(int jj=0; jj<blockN; jj++){
                for(int ii=0; ii<blockM; ii++){
                    if( (i+ii < M) && ( j+jj<N))
                        C[i+ii+(j+jj)*ldC] = miniC[ii+jj*blockM] + beta*C[i+ii + (j+jj)*ldC];
                    else
                        break;    
                }
              }

             
          }
        }

    }        

}

half mpjd::LinearAlgebra::dot(int dim, half *x, int incx, half *y, int incy){
    
    half res=static_cast<half>(0.0);
    #pragma omp parallel for reduction(+:res)
    for(int i=0; i<dim; i++){
      res+= x[i]*y[i];
    }
    
    return res;
}

void mpjd::LinearAlgebra::axpy(int dim, half alpha, half *x, int incx, half *y, int incy){

  //y = y + alpha * x
  #pragma omp parallel for
  for(int i=0;i<dim;i++){
      y[i] += alpha*x[i];
  }

}

half mpjd::LinearAlgebra::nrm2(int dim, half *x,int incx){ 
    half res = dot(dim,x,incx,x,incx);
    return half_float::sqrt(res);
}

void mpjd::LinearAlgebra::scal(int dim, half alpha, half *x, int incx){

  #pragma omp parallel for
  for(int i=0;i<dim;i++){
      x[i] *= alpha;
  }
}






















