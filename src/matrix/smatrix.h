#pragma once
#include "matrix.h"
#include <memory>
namespace mpjd{

/*

	This class implements the 
	D = vecnorm(A);
	A_lp = fp_low(D\(A-eta I)/D)
	
	This matrix is needed from inner solver of JD
	
*/

/* 
	fpMat  : floating point precision of original matrix
	fpSMat : floating point precision of the scaled matrix [lower or equal of fpMat]
*/

template <class fp, class sfp>
class SMatrix {
	private:
	
		/* Original Matrix A Information */
		friend class Matrix<fp>; // Need private information about original matrix
		std::shared_ptr<Matrix<fp>&> mat;   // The A  matrix
		
		/* Scaled Matrix A_lp Information */
		std::vector<sfp>	m_vec_sVALS; // A_lp non zero values - same non zero pattern as A
		std::vector<fp>   m_vec_Norms; // vector representation of D matrix
		
		fp   m_eta;
		void update_eta(fp eta); // updates shift eta and m_vec_sVals 
		void create_vec_Norms();    // initializes m_vec_Norms
		void sMatrix_matVec_csr();//sfp *x, int ldx, sfp *y, int ldy, int blockSize ); // csr matvec
	public:
		SMatrix() = delete;
		SMatrix(Matrix<fp> &mat_);
		
		void sMatrix_matVec();//sfp *x, int ldx, sfp *y, int ldy, int blockSize );
		void vector_diag_scal();//std::vector<fp> *x, int ldx, std::vector<fp> *y, int ldy);
	};
}


// smatrix member functions implementations (templates)
// here to be included the template file with member definitions
//#include "smatrix.tcc"

