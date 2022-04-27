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

    int blockM = std::min(M,20);
    int blockN = std::min(N,20);
    int blockK = std::min(K,80);

     #pragma omp parallel if(blockM>1 && blockN>1)
     {
                
        half miniC[blockM*blockN];
        half miniA[blockM*blockK];
        half miniB[blockK*blockN];
        
        #pragma omp for collapse(2) //if(blockM>1 && blockN>1)
        for(int i=0;i<M;i+=blockM){
          for(int j=0;j<N;j+=blockN){
          
              /* initialize miniC */
              memset(miniC,0,sizeof(half)*blockM*blockN);
              
              
              for(int k=0;k<K;k+=blockK){
              
                  /* prefetch A */
                  int sA = std::min(M-i,blockM);
                  for(int kk=0; kk<blockK; kk++){
                      memcpy(&miniA[0 + kk*blockM],&A[i+(k+kk)*ldA],sA*sizeof(half));
                  }
                  
                  /* prefetch B */
                  int sB = std::min(K-k,blockK);
                  for(int jj=0; (jj<blockN) && (j+jj < N); jj++){
                      memcpy(&miniB[0 + jj*blockM],&B[k+(j+jj)*ldB],sB*sizeof(half));
                  }
        
                  /* multiply */
                  #pragma omp for collapse(2) 
                  for(int ii=0; ii<blockM; ii++){
                     for(int kk=0; kk< blockK; kk++){
                          half aa = miniA[ii+kk*blockM];
                          #pragma omp unroll
                          for(int jj=0; jj< blockN; jj++){
                              miniC[ii+jj*blockM] += alpha*aa*miniB[kk+jj*blockK];
                          }
                      }
                  }
    
              }
              
              
              /* store miniC to C */
              #pragma omp for 
              for(int jj=0; (jj<blockN) & (j+jj<N); jj++){
                #pragma omp unroll
                for(int ii=0; (ii<blockM) && (i+ii < M); ii++){
                    C[i+ii+(j+jj)*ldC] = miniC[ii+jj*blockM] + beta*C[i+ii + (j+jj)*ldC];
                    
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

    /* 
      Both A and B are accessed by column major 
      dot products should be cache efficient
      using simd and reduction in order to accelerate
    */
    #pragma omp parallel for collapse(2) if(M>1 && N>1)
    for(int jj=0; jj< N; jj++){
       for(int ii=0; ii<M; ii++){
           half cc = static_cast<half>(0.0);// C[ii+jj*ldC];
            #pragma omp simd reduction(+:cc)
            for(int kk=0; kk<K; kk++){
                cc += A[kk+ii*ldA]*B[kk+jj*ldB] ;
            }
            
            C[ii+jj*ldC] = beta*C[ii+jj*ldC] + alpha*cc;
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






















