#pragma once

#include <iostream>
// Matrix Constructor
// we are using a csr representation. 
// so dim = col_index(end)
template <class fp> 
mpjd::Matrix<fp>::Matrix(std::vector<fp> &vals_, std::vector<int> &cols_,
					 std::vector<int> &rows_, int dim_,  sparseDS_t DS_ , LinearAlgebra &la_ )
:	
	
	m_dim(dim_),
	m_vec_VALS(vals_),
	m_vec_COL_INDEX(cols_),
	m_vec_ROW_INDEX(rows_),
	m_DS(DS_),
	m_la(la_)
//	m_target(target_)
{
}
	
template <class fp>
int mpjd::Matrix<fp>::Dim(){return m_dim;}





