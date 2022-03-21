#pragma once
#include "../blasWrappers/blasWrappers.h"
#include <vector>
#include <iterator>
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
		
		fp 								 m_norm{};      // matrix norm
		int const		       m_dim{}; 	    // dimension of matrix
		sparseDS_t 				 m_DS{};        // type of sparse matrix representation
	  std::vector<fp>    tmp; // mv accumulator
	public:
		
		Matrix() = delete;
		Matrix(std::vector<fp> &vals_, std::vector<int> &cols_,
					 std::vector<int> &rows_, int dim_, sparseDS_t DS_ , fp norm_, LinearAlgebra &la_);	
		
		void matVec(fp *x, int ldx, fp *y, int ldy, int dimBlock); // y = Ax	
		void matVec(std::vector<fp> &x, int ldx, std::vector<fp> &y, int ldy, int dimBlock); // y = Ax	
		int Dim();
		fp Norm();
		
	};


template <class fp,class sfp>
class ScaledMatrix : public Matrix<fp>{
	private:

		void matVec_COO(std::vector<sfp> &x, int ldx,std::vector<sfp> &y, int ldy, int dimBlock);
		void matVec_COO(sfp *x, int ldx, sfp *y, int ldy, int dimBlock);
    
	  std::vector<fp>    m_cols_norms;
	  std::vector<sfp>    tmp; // mv accumulator
	  
	  std::vector<sfp>   m_vec_sVALS; // values to be updated at every step with ita
  
  
	  void init_vec_norms_COO();
	  void init_vec_norms();  
	  
	public:
		
		ScaledMatrix() = delete;
		ScaledMatrix(Matrix<fp> &mat_);	
		void update_matrix_shift(const fp ita);

		// TODO: 
    // matVec with scaled matrix		
		void matVec(sfp *x, int ldx, sfp *y, int ldy, int dimBlock); // y = Ax	
		void matVec(std::vector<sfp> &x, int ldx, std::vector<sfp> &y, int ldy, int dimBlock); // y = Ax	
		void applyScalMat(fp *x_, int ldx, int rows, int cols); // applying to a vector x the DiagVector ; x = Dx
    void applyScalInvMat(fp *x_, int ldx, int rows, int cols);  // applying to a vector x the inv DiagVector ; x = inv(D)x

		
		
	};
}


// matrix member functions implementations (templates)
// here to be included the template file with member definitions
#include "matrix.tcc"
#include "matrix_matvecs.tcc"
#include "ScaledMatrixMatvecs.tcc"











