#pragma once


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
void mpJDQMR::Matrix<fp>::matVec(fp *x, int ldx, fp *y, int ldy, int dimBlock){


	fp *xx = x;//->data();
	fp *yy = y;//->data();

	fp *vals = m_vec_VALS->data();
	int *rows = m_vec_ROW_INDEX->data();
	int *cols = m_vec_COL_INDEX->data();

/* Algorithm implemented 
	
	for i = 0:n+1 
       y(i)  = 0 
       for j = row_ptr(i):row_ptr(i+1) 
           y(i) = y(i) + val(j) * x(col_ind(j))
       end;
   end;
*/
	for(auto k=0; k<dimBlock; k++){
		for(auto i=0; i < m_dim+1; i++){
			yy[i+k*ldy] = 0;
			for(auto j = rows[i]; j < rows[i+1]; j++){
					yy[i+k*ldy] = yy[i+k*ldy] + vals[j]*(xx[cols[j]+k*ldx]);
			}
		}

	}

}
template<class fp>
int mpJDQMR::Matrix<fp>::Dim(){return m_dim;}
