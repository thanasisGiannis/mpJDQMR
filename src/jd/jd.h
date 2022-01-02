#pragma once

#include "../blasWrappers/blasWrappers.h"
#include "../matrix/matrix.h"
#include "../matrix/smatrix.h"
#include "../subspace/subspace.h"
#include "../solver/sqmr.h"

namespace mpjd{

enum class eigenTarget_t {
							/* Target of wanted eigenvalues */
							SM // smallest by absolute value
							/* future implementations */
							//BM // bigest by absolute value 
							//SR // smallest real
							//BR // bigest real
							
						};


template<class fp, class sfp>
class JD {
	
	eigenTarget_t  eigTarget;   // Smallest by absolute value 
	int 				   numEvals;    // number of wanted eigenvalues

	/* Shared variables between classes */
	std::vector<fp> w; int ldw; // common with basis and sqmr
															// next subspace direction
														  // reserved at subspace class
														  // will need to be used by inner solver
														  
	std::vector<fp> Q;			 int ldQ; 			// common with basis and sqmr
	std::vector<fp> L;											// common with basis and sqmr
	std::vector<fp> R;			 int ldR;				// common with basis and sqmr

	std::vector<fp> Qlocked; int ldQlocked; // common with basis and sqmr
	std::vector<fp> Llocked; 								// common with basis and sqmr
	std::vector<fp> Rlocked; int ldRlocked; // locked eigen residual 


	/* CAUTION THIS SHOULD BE AFTER VARIABLES */
	LinearAlgebra la;
	Matrix<fp>    mat;
	Subspace<fp>  basis; // implement after basis implementation
	//SQMR<fp,sfp> sqmr;  // implement after s implementation
	
	
public:
										 
	void solve();									 
	JD() = delete;
	
	JD(std::vector<fp> vals_, std::vector<int> row_index_, std::vector<int> col_index_,
			int dim_, sparseDS_t DS_,int nEvals, eigenTarget_t eigTarget_,int maxBasis_, target_t target_); 

};

}


#include "jd.tcc"
#include "eigen_solver.tcc"
