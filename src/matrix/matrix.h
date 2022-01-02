#pragma once
#include "../blasWrappers/blasWrappers.h"
#include <vector>
// matrix interface class
// needed by main program and JD main iteration
namespace mpjd{

enum class sparseDS_t
		 {
			  //  following future sparse representations
			  // CSR,
			  // CSC,
				COO
			};
			
template <class fp>
class Matrix{
	// sparse matrix information of a general matrix A
	// functionality of matvec for different sparse representations
	private:
		LinearAlgebra &m_la;
		void matVec_COO(std::vector<fp> &x, int ldx,std::vector<fp> &y, int ldy, int dimBlock);
		void matVec_COO(fp *x, int ldx,fp *y, int ldy, int dimBlock);
	protected:
		std::vector<fp>  m_vec_VALS;      // matrix values
		std::vector<int> m_vec_COL_INDEX; // column index vector	
		std::vector<int> m_vec_ROW_INDEX; // row index vector
		
		int const		       m_dim{}; 	    // dimension of matrix
		sparseDS_t 				 m_DS{};        // type of sparse matrix representation
		target_t 					 m_target{};		// target machinery for computations
	public:
		
		Matrix() = delete;
		Matrix(std::vector<fp> &vals_, std::vector<int> &cols_,
					 std::vector<int> &rows_, int dim_, sparseDS_t DS_ , LinearAlgebra &la_);	
		
		void matVec(fp *x, int ldx, fp *y, int ldy, int dimBlock); // y = Ax	
		void matVec(std::vector<fp> &x, int ldx, std::vector<fp> &y, int ldy, int dimBlock); // y = Ax	
		int Dim();
	};
}



// matrix member functions implementations (templates)
// here to be included the template file with member definitions
#include "matrix.tcc"
#include "matrix_matvecs.tcc"












