#include <iostream>
#include <random>
#include <algorithm>
#include <iterator>
#include "../blasWrappers/blasWrappers.h"

//#include "../include/helper.h"

template<class fp>
void Rand_Vec_Init(std::vector<fp> &v,int dimv);


template<class fp>
mpjd::Subspace<fp>::	Subspace(Matrix<fp> &mat_, int dim_,int numEvals_,int maxBasis_, LinearAlgebra &la_,
				 std::vector<fp> &w_, int &ldw_,	
				 std::vector<fp> &Q_, int &ldQ_, std::vector<fp> &L_, 
				 std::vector<fp> &R_, int &ldR_, 
				 std::vector<fp> &Qlocked_, int &ldQlocked_,std::vector<fp> &Llocked_,
				 std::vector<fp> &Rlocked_, int &ldRlocked_)

:
	la(la_),
	mat(mat_),
	dim(dim_),
	numEvals(numEvals_),
	maxBasis(maxBasis_),
	blockSize(numEvals_),
	lockedNumEvals(0),
	w(w_),
	ldw(ldw_),
	Q(Q_),ldQ(ldQ_),
	L(L_),
	R(R_),ldR(ldR_),
	Qlocked(Qlocked_),ldQlocked(ldQlocked_),
	Llocked(Llocked_),
	Rlocked(Rlocked_),ldRlocked(ldRlocked_)
{

	V.reserve(dim*maxBasis*numEvals); ldV = dim; // subspace basis
	T.reserve(maxBasis*numEvals*maxBasis*numEvals); ldT = maxBasis*numEvals;// projected matrix
	
	Q.reserve(dim*numEvals); ldQ = dim; // locked eigenvectors
	L.reserve(numEvals); 				        // locked eigenvalues
	R.reserve(dim*numEvals); ldR = dim; // eigen residual 
	
	Qlocked.reserve(dim*numEvals); ldQlocked = dim; // locked eigenvectors
	Llocked.reserve(numEvals); 								      // locked eigenvalues
	Rlocked.reserve(dim*numEvals); ldRlocked = dim; // locked eigen residual 
	
	Aw.reserve(dim*numEvals); ldAw = dim;  // tmp vector to used in Av
	w.reserve(dim*numEvals);  ldw  = dim;  // new subspace direction
	
}



template<class fp>
void mpjd::Subspace<fp>::Subspace_init(){

	/*
		Initialize basis V with random vectors
		Then orthogonalize them if in blocking mode
	 */

	// size(V(:,1)) == dim * blockSize
	Rand_Vec_Init(V,dim*numEvals); // initialize subspace
	basisSize = 1;									// basis subspace init
	Subspace_orth_basis();					// orthogonalize basis vectors

}


template<class fp>
void mpjd::Subspace<fp>::Subspace_orth_basis(std::vector<fp> v){

	auto VV = static_cast<fp*>(V->data());
	auto vv = static_cast<fp*>(v->data());

	int rows = dim;
	int cols = basisSize*blockSize;

	for(auto i=0; i < cols ; i++){
			auto alpha {-la.dot(rows,&VV[0+i*ldV],1,&vv[0+0*ldV],1)};// alpha = V(i)'v
			la.axpy(rows,alpha,&VV[0+i*ldV],1,&vv[0+0*ldV],1);     	// v = v - V(i)*alpha
	}	
	auto alpha {la.nrm2(rows,&vv[0+0*ldV],1)}	;
	la.scal(dim,static_cast<fp>(1.0/alpha),&vv[0+0*ldV],1);    // v = v/norm(v)
}


template<class fp>
void mpjd::Subspace<fp>::Subspace_orth_basis(){
	/* orthogonalize basis V */
	auto VV = static_cast<fp*>(V.data());
	int rows = dim;
	int cols = basisSize*blockSize;
	
	for(auto j=0; j < cols ; j++){
		for(auto i=j-1; i >= 0 ; i--){
				auto alpha {-la.dot(rows,&VV[0+i*ldV],1,&VV[0+j*ldV],1)};// alpha = V(i)'V(j)
				la.axpy(rows,alpha,&VV[0+i*ldV],1,&VV[0+j*ldV],1);     	// V(j) = V(j) - V(i)*alpha
		}	
		auto alpha {la.nrm2(rows,&VV[0+j*ldV],1)}	;
		la.scal(dim,static_cast<fp>(1.0/alpha),&VV[0+j*ldV],1);    // V(j) = V(j)/norm(V(j))
	}
}

template<class fp>
void mpjd::Subspace<fp>::Subspace_update_basis(std::vector<fp> v){}

template<class fp>
void Rand_Vec_Init(std::vector<fp> &v,int dimv){
	std::random_device rand_dev;
	std::mt19937 			 generator(rand_dev());
	std::uniform_int_distribution<int> distr(-100,100);

	for(auto i=0;i<dimv;i++){
		v.push_back(static_cast<fp>(distr(generator))/static_cast<fp>(100));
	}

}

