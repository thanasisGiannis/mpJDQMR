#pragma once

#include <iostream>
#include <cmath>
#include "../blasWrappers/blasWrappers.h"


template<class fp> 
void printMat(std::vector<fp> A,int ldA, int rows,int cols,std::string name){
	fp *A_ = A.data();
	for(auto i=0;i<rows;i++){	
		for(auto j=0;j<cols; j++){
			std::cout << name << "("<< i+1 << "," << j+1 << ")=" << A_[i+j*ldA] << ";" << std::endl;
		}
	}
}


// Matrix Constructor
// we are using a csr representation. 
// so dim = col_index(end)
template <class fp> 
mpjd::Matrix<fp>::Matrix(std::vector<fp> &vals_, std::vector<int> &cols_,
					 std::vector<int> &rows_, int dim_,  sparseDS_t DS_ , fp norm_, LinearAlgebra &la_)	
:	
	
	m_dim(dim_),
	m_vec_VALS(vals_),
	m_vec_COL_INDEX(cols_),
	m_vec_ROW_INDEX(rows_),
	m_DS(DS_),
	m_norm(norm_),
	m_la(la_)
{
  tmp.reserve(m_dim); // mv accumulator
  tmp.insert(tmp.begin(),tmp.capacity(),static_cast<fp>(0.0));
//  std::cout << "Matrix" << std::endl;
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
  m_vec_sVALS.reserve(this->m_vec_VALS.size());
  tmp.reserve(this->m_dim); // mv accumulator
  tmp.insert(tmp.begin(),tmp.capacity(),static_cast<sfp>(0.0));

  init_vec_norms(); // Create Diagonal Matrix
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
  
  for(auto j=0;j<this->m_dim;j++){
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



template<class fp, class sfp>
void mpjd::ScaledMatrix<fp,sfp>::update_matrix_shift(const fp ita){
  /* 
    for each row,col element of A
    row == col: dA(row,col) = (A(row,col)-ita)/(D(row)*D(col))
    row != col: dA(row,col) = (A(row,col))/(D(row)*D(col))
  */

  const int* const Rows = this->m_vec_ROW_INDEX.data();
  const int* const Cols = this->m_vec_COL_INDEX.data();
  const fp * const A    = this->m_vec_VALS.data();
  
  const int numVals = this->m_vec_VALS.size();
  
  m_vec_sVALS.clear();
  
  if(m_cols_norms.size() <= 0){
    exit(-1);
  }
  
  const fp* const cNrms = m_cols_norms.data();

  for(auto i{0}; i<numVals; i++){
    int row  = Rows[i];
    int col  = Cols[i];
    fp  val  = A[i];
    
    fp  cNrm_row = cNrms[row];
    fp  cNrm_col = cNrms[col];
    
    if(row == col){
      m_vec_sVALS.push_back(static_cast<sfp>((val-ita)/(cNrm_row*cNrm_col)));
    }else{
      m_vec_sVALS.push_back(static_cast<sfp>((val)/(cNrm_row*cNrm_col)));
    }
  }

}





template<class fp, class sfp>
void mpjd::ScaledMatrix<fp,sfp>::applyScalMat(fp *x_, int ldx, int rows, int cols){
 // applying to a vector x the DiagVector ; x = Dx
 auto D = m_cols_norms.data();
 auto x = x_;//.data();	
 
 for(auto r=0; r<rows; r++){
  for(auto c=0; c<cols; c++){
      x[r+c*ldx] *= D[r];
  }    
 }
	
}
template<class fp, class sfp>
void mpjd::ScaledMatrix<fp,sfp>::applyScalInvMat(fp *x_, int ldx, int rows, int cols){
  // applying to a vector x the inv DiagVector ; x = inv(D)x
 auto D = m_cols_norms.data();
 auto x = x_;//.data();	
 
 for(auto r=0; r<rows; r++){
  for(auto c=0; c<cols; c++){
      x[r+c*ldx] = x[r+c*ldx]/static_cast<fp>(D[r]);
  }    
 } 

}






















