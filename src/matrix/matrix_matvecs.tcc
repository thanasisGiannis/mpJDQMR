/* 
  CHOOSE SPARSE MATVEC ALGORITHM TO USE 
*/

template <class fp>
void mpjd::Matrix<fp>::matVec(const std::vector<fp> &x, const int ldx, 
    std::vector<fp> &y, const int ldy, const int dimBlock) {
  
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
void mpjd::Matrix<fp>::matVec_COO(const std::vector<fp> &x, const int ldx, 
    std::vector<fp> &y, const int ldy, const int dimBlock) {
	
	auto xx = x.data();
	//auto yy = tmp.data();
	fp  *vals = m_vec_VALS.data();
	int *rows = m_vec_ROW_INDEX.data();
	int *cols = m_vec_COL_INDEX.data();
	
	tmp.resize(m_dim*dimBlock);
	std::fill(tmp.begin(), tmp.end(), 0);
	
	int vec_size = static_cast<int>(m_vec_VALS.size());
	for( auto k = 0; k < dimBlock; k++ ) {
		for( auto i = 0; i < vec_size; i++ ) {
			tmp[rows[i] + k*ldy] += vals[i]*xx[cols[i]+k*ldx];
		}
	}
	y = tmp;
}
