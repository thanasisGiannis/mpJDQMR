#pragma once

#include <iostream>
#include <cmath>
// Matrix Constructor
// we are using a csr representation. 
// so dim = col_index(end)
template <class fp> 
mpjd::Matrix<fp>::Matrix(std::vector<fp> &vals_, std::vector<int> &cols_,
					 std::vector<int> &rows_, int dim_,  sparseDS_t DS_ , fp norm_, LinearAlgebra &la_ )
:	
	
	m_dim(dim_),
	m_vec_VALS(vals_),
	m_vec_COL_INDEX(cols_),
	m_vec_ROW_INDEX(rows_),
	m_DS(DS_),
	m_norm(norm_),
	m_la(la_)
{
  std::cout << "Matrix" << std::endl;
}
	
	
		
template <class fp>
int mpjd::Matrix<fp>::Dim(){return m_dim;}

template <class fp>
fp mpjd::Matrix<fp>::Norm(){return m_norm;}



template <class fp, class sfp> 
mpjd::ScaledMatrix<fp,sfp>::ScaledMatrix(Matrix<fp> &mat_)
: Matrix<fp>(mat_)
{
  m_cols_norms.reserve(this->m_dim);
  m_vec_Scaled_VALS.reserve(this->m_vec_VALS.size());
  
  init_vec_norms(); // Create Diagonal Matrix
}


template<class fp> 
void printMat(std::vector<fp> A,int ldA, int rows,int cols,std::string name){
	fp *A_ = A.data();
//	std::cout <<"%"<< name << std::endl;
	
//	std::cout << name << "[]" << std::endl;
	for(auto i=0;i<rows;i++){	
		for(auto j=0;j<cols; j++){
			std::cout << name << "("<< i+1 << "," << j+1 << ")=" << A_[i+j*ldA] << ";" << std::endl;
		}
	}
}


/* DA = vecnorms(A) */
template<class fp, class sfp>
void mpjd::ScaledMatrix<fp,sfp>::init_vec_norms_COO(){
  
  /* DA = zeros(dim,1) */
  m_cols_norms.resize(this->m_dim,static_cast<fp>(0));
  
  fp *DA = m_cols_norms.data();
  fp *A  = this->m_vec_VALS.data();

  for(auto k=0; k < this->m_vec_COL_INDEX.size(); k++){
    auto j = this->m_vec_COL_INDEX.at(k);
    /* d = norm(A(:,j))*/
    /* DA(j) += d*d */
    DA[j] = DA[j]*DA[j] + A[k]*A[k];
    /* d = sqrt(d) */
    DA[j] = std::sqrt(DA[j]);
  } 
}


template<class fp, class sfp>
void mpjd::ScaledMatrix<fp,sfp>::init_vec_norms(){

  switch(this->m_DS){
    case mpjd::sparseDS_t::COO:
      init_vec_norms_COO();    
      break;
      
    default:
      std::cout << "Error" << std::endl;
      exit(-1);
  };
}


































