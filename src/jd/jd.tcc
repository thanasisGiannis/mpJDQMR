#pragma omp once

#include <iostream>
#include <random>
#include <algorithm>
#include <iterator>
//#include <Accelerate/Accelerate.h>

#include "../include/blasWrappers.h"
template<class fp,class sfp>
void mpJDQMR::JD<fp,sfp>::eigenSolve(){

	/* MAIN JD ALGORITHM FUNCTION */
	
	// V = rand(dim,numEvals)
	matRand(V,ldV,numEvals);
	
	// V = orth(V)
	orth(V,ldV,dim,numEvals);
	
	//T = V'AV
	
	//[Q,L] = eig(T,V,numEVals, target)
	
	//while no convergence
		
		//V = Q
		
		//for j=2...m
			
			//R = AQ-QL
			
			// CHECK CONVERGENCE
			
			// for i=1...numEvals
				// Solve Mv(:,i) = R(:,i)
			//end for i
			
			// V = orht(V,v)
			//[Q,L] = eig(T,V,numEVals, target)
			
		//end for j
	
		
	
	//end while
	
}


template<class fp,class sfp>
void mpJDQMR::JD<fp,sfp>::intitT(){

	// update projected matrix using projection matrix V
	// update projected matrix using v vector
	// we follow a C style implemetation of the projected matrix update
	// using pointer arithmetic
 
	/*		
	
		T =|VtAV 0|
			 |0    0|
	*/
 
 	 
 
}


template<class fp,class sfp>
void mpJDQMR::JD<fp,sfp>::orth(std::vector<fp> *V,int ldV, int rows, int cols)const{

	/* Basic Gram-Schmidt orthogonalization */
	auto VV = static_cast<fp*>(V->data());
	
	for(auto j=0; j < cols ; j++){
		for(auto i=j-1; i >= 0 ; i--){
				auto alpha {-dot(rows,&VV[0+i*ldV],1,&VV[0+j*ldV],1)};// alpha = V(i)'V(j)
				axpy(rows,alpha,&VV[0+i*ldV],1,&VV[0+j*ldV],1);     	// V(j) = V(j) - V(i)*alpha
		}	
		auto alpha {nrm2(rows,&VV[0+j*ldV],1)}	;
		scal(dim,static_cast<fp>(1.0/alpha),&VV[0+j*ldV],1);    // V(j) = V(j)/norm(V(j))
	}

}
	
template<class fp,class sfp>
void mpJDQMR::JD<fp,sfp>::orth(std::vector<fp> *V,int ldV, int rows, int cols, std::vector<fp> v){

	auto VV = static_cast<fp*>(V->data());
	auto vv = static_cast<fp*>(v->data());
	
		for(auto i=0; i < cols ; i++){
				auto alpha {-dot(rows,&VV[0+i*ldV],1,&vv[0+0*ldV],1)};// alpha = V(i)'v
				axpy(rows,alpha,&VV[0+i*ldV],1,&vv[0+0*ldV],1);     	// v = v - V(i)*alpha
		}	
		auto alpha {nrm2(rows,&vv[0+0*ldV],1)}	;
		scal(dim,static_cast<fp>(1.0/alpha),&vv[0+0*ldV],1);    // v = v/norm(v)

}

template<class fp,class sfp>
void mpJDQMR::JD<fp,sfp>::matRand(std::vector<fp> *V, int rows, int cols){
	// fill with random values a matrix
		
	std::random_device rand_dev;
	std::mt19937 			 generator(rand_dev());
	std::uniform_int_distribution<int> distr(-100,100);

	for(auto i=0;i<rows*cols;i++){
		V->push_back(static_cast<fp>(distr(generator))/static_cast<fp>(100));
	}

}



template <class fp,class sfp>
mpJDQMR::JD<fp,sfp>::JD(Matrix<fp> *_Mat,int const _dim, int const _numEvals,
												 int const _maxBasis, int const _resSize)
: Mat{_Mat}, 
	dim{_dim}, 
	numEvals{_numEvals},
	maxBasis{_maxBasis},
	resSize{numEvals}
	
{
	// We follow a column wise matrix representation

	// projected matrix    : numEvals*maxBasis x numEvals*maxBasis
	T = new std::vector<fp>((numEvals*maxBasis)*(numEvals*maxBasis),0);
	ldT = numEvals*maxBasis;
	T->reserve((numEvals*maxBasis)*(numEvals*maxBasis));
	
	// rayleigh ritz basis : dim x numEvals*maxBasis
	V = new std::vector<fp>(dim*numEvals*maxBasis,0); ldV = dim;
	V->reserve(dim*numEvals*maxBasis);
	
	// ritz vectors        : dim x numEvals
	Q = new std::vector<fp>(dim*numEvals,0); ldQ = dim;
	Q->reserve(dim*numEvals);
	
	// ritz values         : numEvals	
	L = new std::vector<fp>(numEvals,0);
	L->reserve(numEvals);

	// residual vectors    : dim x numEvals
	R = new std::vector<fp>(dim*numEvals,0); ldR = dim;
	R->reserve(dim*numEvals);
	
}



template <class fp,class sfp>
mpJDQMR::JD<fp,sfp>::~JD()
{

	delete T;
	delete V;
	delete Q;
	delete L;
	delete R;

}

