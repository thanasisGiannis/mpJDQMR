#pragma once
#include <vector>

// matrix interface class
namespace mpJDQMR{

template <class fp>
class Matrix{
	// matrix information can change between 
	std::vector<fp>   *m_vec_VALS{};      // matrix values
	std::vector<int>  *m_vec_COL_INDEX{}; // column index vector	
	std::vector<int>  *m_vec_ROW_INDEX{}; // row index vector
	int const		       m_dim{};    	      // dimension of matrix
	public:
		Matrix() = delete;
		Matrix(std::vector<fp> *Vals, std::vector<int> *Cols, std::vector<int> *Rows);	
		
		virtual void matVec(fp *x, int ldx, fp *y, int ldy, int dimBlock); // y = Ax	
		int Dim();
	};
}


// matrix member functions implementations (templates)
#include "../src/matrix/matrix.tcc"

