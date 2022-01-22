#pragma once

#include "../blasWrappers/blasWrappers.h"
#include "../matrix/matrix.h"
#include "../matrix/smatrix.h"
#include "../subspace/subspace.h"
#include "../solver/sqmr.h"

namespace mpjd{



template<class fp, class sfp>
class JD {
	
	eigenTarget_t  eigTarget;   // Smallest by absolute value 
	//int 				   numEvals;    // number of wanted eigenvalues
	int 					 maxIters;
	int						 dim;
	fp						 tol;
	
	/* Shared variables between classes */
	std::vector<fp> w; int ldw; // common with basis and sqmr
															// next subspace direction
														  // reserved at subspace class
														  // will need to be used by inner solver
														  
	std::vector<fp> Q;			 int ldQ; 			// common with basis and sqmr
	std::vector<fp> L;											// common with basis and sqmr
	std::vector<fp> R;			 int ldR;				// common with basis and sqmr

	std::vector<fp> &Qlocked; int ldQlocked; // common with basis and sqmr
	std::vector<fp> &Llocked; 								// common with basis and sqmr
	std::vector<fp> &Rlocked; int ldRlocked; // locked eigen residual 


	/* CAUTION THIS SHOULD BE AFTER VARIABLES */
	LinearAlgebra la;
	Matrix<fp>    mat;
	Subspace<fp>  basis; // implement after basis implementation
	//SQMR<fp,sfp> sqmr;  // implement after s implementation
	bool Check_Convergence();
	
public:
										 
	void solve();									 
	JD() = delete;
	
	JD(std::vector<fp> vals_, std::vector<int> row_index_, std::vector<int> col_index_,
			int dim_, sparseDS_t DS_, std::vector<fp> &Q_, int &ldQ_, std::vector<fp> &L_, 
			std::vector<fp> &R_, int &ldR_, fp norm_, int nEvals, eigenTarget_t eigTarget_, fp tol_,
			int maxBasis_, int maxIters_); 

};

}


#include "jd.tcc"
#include "eigen_solver.tcc"
