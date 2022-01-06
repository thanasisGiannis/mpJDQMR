#include <iostream>
#include <random>
#include <algorithm>
#include <iterator>
#include "../blasWrappers/blasWrappers.h"

//#include "../include/helper.h"

template<class fp>
void init_vec_zeros(std::vector<fp> &v, int nVals){

	for (auto j=0;j<nVals; j++){
		v.push_back(static_cast<fp>(0));
	}

} 


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
	basisSize(0),
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
//	init_vec_zeros(V,dim*maxBasis*numEvals);
	
	T.reserve(maxBasis*numEvals*maxBasis*numEvals); ldT = maxBasis*numEvals;// projected matrix
	init_vec_zeros(T,maxBasis*numEvals*maxBasis*numEvals);
	
	Q.reserve(dim*numEvals); ldQ = dim; // locked eigenvectors
//	init_vec_zeros(Q,dim*numEvals);

	L.reserve(numEvals); 				        // locked eigenvalues
//	init_vec_zeros(L,numEvals);


	R.reserve(dim*numEvals); ldR = dim; // eigen residual 
//	init_vec_zeros(R,dim*numEvals);

	Qlocked.reserve(dim*numEvals); ldQlocked = dim; // locked eigenvectors
//	init_vec_zeros(Qlocked,dim*numEvals);

	Llocked.reserve(numEvals); 								      // locked eigenvalues
//	init_vec_zeros(Llocked,numEvals);

	Rlocked.reserve(dim*numEvals); ldRlocked = dim; // locked eigen residual 
	
	Aw.reserve(dim*numEvals); ldAw = dim;  // tmp vector to used in Av
	init_vec_zeros<fp>(Aw,dim*numEvals);

	w.reserve(dim*numEvals);  ldw  = dim;  // new subspace direction
	
}



template<class fp>
void mpjd::Subspace<fp>::Subspace_init_direction(){

	/*
		Initialize basis V with random vectors
		Then orthogonalize them if in blocking mode
	*/

	std::random_device rand_dev;
	std::mt19937 			 generator(rand_dev());
	std::uniform_int_distribution<int> distr(-100,100);

	for(auto i=0;i<dim*numEvals;i++){
		w.push_back(static_cast<fp>(distr(generator))/static_cast<fp>(100));
	}

}


template<class fp>
void mpjd::Subspace<fp>::Subspace_orth_direction(){

	auto VV = static_cast<fp*>(V.data());
	auto vv = static_cast<fp*>(w.data());

	int rows = dim;
	int cols = basisSize*blockSize;

	for(int j=0; j < blockSize; j++){
		for(int i=0; i < cols ; i++){
				fp alpha {-la.dot(rows,&VV[0+i*ldV],1,&vv[0+j*ldV],1)};// alpha = V(i)'v
				la.axpy(rows,alpha,&VV[0+i*ldV],1,&vv[0+j*ldV],1);     	// v = v - V(i)*alpha
		}	
	}

	for(int j=0; j < blockSize; j++){
		for(int i=0; i < j ; i++){
				auto alpha {-la.dot(rows,&vv[0+i*ldV],1,&vv[0+j*ldV],1)};// alpha = V(i)'v
				la.axpy(rows,alpha,&vv[0+i*ldV],1,&vv[0+j*ldV],1);     	// v = v - V(i)*alpha
		}	
		auto alpha {la.nrm2(rows,&vv[0+j*ldV],1)}	;
		la.scal(dim,static_cast<fp>(1.0/alpha),&vv[0+j*ldV],1);    // v = v/norm(v)

	}


}

template<class fp> 
void printMat(std::vector<fp> A,int ldA, int rows,int cols,std::string name){
	fp *A_ = A.data();
	std::cout << name << std::endl;
	
	for(auto i=0;i<rows;i++){	
	
		for(auto j=0;j<cols; j++){
			std::cout << "\% " << name << "("<< i << "," << j << ")=" << A_[i+j*ldA] << "        ";
		}
		std::cout << std::endl;
	}
}


template<class fp>
void mpjd::Subspace<fp>::Subpsace_project_at_new_direction(){
	mat.matVec(w,ldw, Aw, ldAw, blockSize); // Aw = A*w
	
	fp *Aw_ = Aw.data();
	fp *w_  =  w.data();
	fp *T_  =  T.data();
	fp *V_  =  V.data();
		
	fp one  = static_cast<fp>(1.0);
	fp zero = static_cast<fp>(0.0);

	/**
		T = [	T 0; 0 w'Aw]
	**/
	la.gemm('T','N', blockSize, blockSize, dim, one, w_, ldw,
						Aw_, ldAw, zero, &T_[basisSize*blockSize + basisSize*blockSize*ldT],ldT);	
						
	/* only the diagonal part is of interest */					
	if(basisSize < 1) return;
							
	/**
		T = [	T V'Aw; 0 w'Aw]
	**/
	la.gemm('T','N', blockSize, basisSize*blockSize, dim, one, V_, ldV,
						Aw_, ldAw, zero, &T_[basisSize*blockSize + (basisSize+1)*blockSize*ldT],ldT);	

	/**
		T = [	T V'Aw; w'AV w'Aw]
	**/
	la.gemm('T','N', blockSize, basisSize*blockSize, dim, one, Aw_, ldAw,
						V_, ldV, zero, &T_[(basisSize+1)*blockSize + basisSize*blockSize*ldT],ldT);	
}

template<class fp>
void mpjd::Subspace<fp>::Subspace_update_basis(std::vector<fp> v){}


















