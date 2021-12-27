#pragma once
#include <vector>
#include <array>
#include <iostream>


// Matrix Constructor
// we are using a csr representation. 
// so dim = col_index(end)
template <class fp> 
mpJDQMR::Matrix<fp>::Matrix(std::vector<fp> *Vals, std::vector<int> *Rows, std::vector<int> *Cols )
:	m_dim(Cols->back()),
	m_vec_VALS(Vals),
	m_vec_COL_INDEX(Cols),
	m_vec_ROW_INDEX(Rows){}
	


// csr implementation of y = Ax
template <class fp>
void mpJDQMR::Matrix<fp>::matVec(std::vector<fp> *x, int ldx, std::vector<fp> *y, int ldy, int dimBlock){

/*
	for i = 0:n+1 
       y(i)  = 0 
       for j = row_ptr(i):row_ptr(i+1) 
           y(i) = y(i) + val(j) * x(col_ind(j))
       end;
   end;
*/

	for(auto i=0; i < m_dim+1; i++){
		y->at(i) = 0;
		for(auto j = m_vec_ROW_INDEX->at(i); j < m_vec_ROW_INDEX->at(i+1); j++){
				y->at(i) = y->at(i) + m_vec_VALS->at(j)*(x->at(m_vec_COL_INDEX->at(j)));
		}
	}


}


