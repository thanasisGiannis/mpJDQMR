#pragma once
#include "../matrix/matrix.h"
#include "../blasWrappers/blasWrappers.h"

namespace mpjd{
template<class fp>
class Subspace {

	std::vector<fp> V;  int ldV; 	// subspace basis
	std::vector<fp> T;  int ldT;  // projected matrix
	std::vector<fp> Aw; int ldAw; //
	std::vector<fp> AV; int ldAV;
	/* Next vectors are manipulated by JD class */
	std::vector<fp> &w; 			int &ldw;			  // new subspace direction
	
	std::vector<fp> &Q; int &ldQ; // ritz vectors
	std::vector<fp> q;  int  ldq; // eigenvectors of T
	
	std::vector<fp> &L; 					// ritz values
	std::vector<fp> &R; int &ldR; // eigenresidual 
	
	std::vector<fp> &Qlocked; int &ldQlocked; // locked eigenvectors
	std::vector<fp> &Llocked; 								// locked eigenvalues
	std::vector<fp> &Rlocked; int &ldRlocked; // locked eigen residual 
	
	
	int blockSize;		// non locked blockSize [=numEvals-lockedNumEvals]
	int maxBasis;		  // maximum basis size basisSize <= maxBasisSize ]
	int basisSize;    // current size of basis [size(V,2) = basisSize * blockSize]
	
	int numEvals; 			// number of eigenvalues wanted
	int lockedNumEvals; // number of converged eigenvalues
	int dim;
	eigenTarget_t eigTarget;
	
	Matrix<fp>    &mat;
	LinearAlgebra &la;
		
public:
	Subspace()=delete;
	Subspace(Matrix<fp> &mat_, int dim_,int numEvals_, eigenTarget_t eigTarget_, 
				 int maxBasis_, LinearAlgebra &la_,
				 std::vector<fp> &w_, int &ldw_,	
				 std::vector<fp> &Q_, int &ldQ_, std::vector<fp> &L_, 
				 std::vector<fp> &R_, int &ldR_, 
				 std::vector<fp> &Qlocked_, int &ldQlocked_,std::vector<fp> &Llocked_,
				 std::vector<fp> &Rlocked_, int &ldRlocked_);

	
//	void Subspace_init();
	void Subspace_init_direction();
	void Subspace_orth_direction();
	void Subspace_project_at_new_direction();
	void Subspace_update_basis();
	void Subspace_projected_mat_eig();
};

}

#include "subspace.tcc"
