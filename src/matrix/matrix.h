#pragma once

#include <vector>
#include <iterator>

#include "../include/mpjd_stats.h"
#include "../blasWrappers/blasWrappers.h"

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
class Matrix : public matrixStatistics {
  // sparse matrix information of a general matrix A
  // functionality of matvec for different sparse representations
  public:
      Matrix() = delete;
      Matrix(std::vector<fp> &vals_, std::vector<int> &cols_,
          std::vector<int> &rows_, int dim_, sparseDS_t DS_ ,
          fp norm_, LinearAlgebra &la_);	
          
      void matVec(fp *x, int ldx, fp *y, int ldy, int dimBlock); 	
      void matVec(std::vector<fp> &x, int ldx, 
          std::vector<fp> &y, int ldy, int dimBlock); 
      int Dim();
      fp Norm();

  protected:
      int const		       m_dim{}; 	      // dimension of matrix
      std::vector<fp>    m_vec_VALS;      // matrix values
      std::vector<int>   m_vec_COL_INDEX; // column index vector	
      std::vector<int>   m_vec_ROW_INDEX; // row index vector
      std::vector<fp>    tmp;             // mv accumulator
      sparseDS_t 				 m_DS{};          // type of sparse matrix representation
      fp 								 m_norm{};        // matrix norm
      
  private:
	    void matVec_COO(std::vector<fp> &x, int ldx,
	        std::vector<fp> &y, int ldy, int dimBlock);
	    void matVec_COO(fp *x, int ldx, fp *y, int ldy, int dimBlock);
	    
	    LinearAlgebra &m_la;
};


template <class fp,class sfp>
class ScaledMatrix : public Matrix<fp> {
	public:
		ScaledMatrix() = delete;
		ScaledMatrix(Matrix<fp> &mat_);	
		
		// matVec with scaled matrix		
		// y = Ax	
		void matVec(std::vector<sfp> &x, int ldx, 
		    std::vector<sfp> &y, int ldy, int dimBlock); 
		// applying to a vector x the DiagVector ; x = Dx
		void applyScalMat(fp *x_, int ldx, int rows, int cols);
		// applying to a vector x the inv DiagVector ; x = inv(D)x
    void applyScalInvMat(fp *x_, int ldx, int rows, int cols);  
    void update_matrix_shift(const fp ita);

	private:
		void matVec_COO(std::vector<sfp> &x, int ldx,
		    std::vector<sfp> &y, int ldy, int dimBlock);
		void matVec_COO(sfp *x, int ldx, sfp *y, int ldy, int dimBlock);
    void init_vec_norms_COO();
	  void init_vec_norms();  
	  
	  std::vector<fp>    m_cols_norms;
	  std::vector<sfp>   m_vec_sVALS;  // values to be updated at every step with ita
	  std::vector<sfp>   tmp;          // mv accumulator
};
}

// matrix member functions implementations (templates)
// here to be included the template file with member definitions
#include "matrix.tcc"
#include "matrix_matvecs.tcc"
#include "ScaledMatrixMatvecs.tcc"











