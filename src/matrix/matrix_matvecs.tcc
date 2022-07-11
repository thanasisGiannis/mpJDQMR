/* 
  CHOOSE SPARSE MATVEC ALGORITHM TO USE 
*/

template <class fp>
void mpjd::Matrix<fp>::matVec(std::vector<fp> &x, int ldx, 
    std::vector<fp> &y, int ldy, int dimBlock) {
  
  updateMatVecs(dimBlock);
	switch( m_DS ) {
		case sparseDS_t::COO:
			matVec_COO(x, ldx, y, ldy, dimBlock);
			break;
		default:
			std::cout << "This should not have been reached" << std::endl;
			break;
		};
}

// COO implementation of y = Ax
template <class fp>
void mpjd::Matrix<fp>::matVec_COO(std::vector<fp> &x, int ldx, 
    std::vector<fp> &y, int ldy, int dimBlock) {
	
	auto xx = x.data();
	auto yy = tmp.data();
	fp  *vals = m_vec_VALS.data();
	int *rows = m_vec_ROW_INDEX.data();
	int *cols = m_vec_COL_INDEX.data();
	
	tmp.resize(m_dim*dimBlock);
	std::fill(tmp.begin(), tmp.end(), 0);
	
	for( auto k = 0; k < dimBlock; k++ ) {
		for( auto i = 0; i < m_vec_VALS.size(); i++ ) {
			tmp[rows[i] + k*ldy] += vals[i]*xx[cols[i]+k*ldx];
		}
	}
	y = tmp;
}
