/* 
  CHOOSE SPARSE MATVEC ALGORITHM TO USE 
*/

template <class fp, class sfp>
void mpjd::ScaledMatrix<fp, sfp>::matVec(
  const std::vector<sfp> &x, const int ldx, 
  std::vector<sfp> &y, const int ldy, const int dimBlock) {

  mpjd::matrixStatistics::updateMatVecs(dimBlock);

  switch ( this->m_DS ){
    case sparseDS_t::COO:
      matVec_COO(x, ldx, y, ldy, dimBlock);
      break;
    default:
      std::cout << "This should not have been reached" << std::endl;
      break;
    };
}

// COO implementation of y = Ax
template <class fp, class sfp>
void mpjd::ScaledMatrix<fp, sfp>::matVec_COO(
  const std::vector<sfp> &x, const int ldx,
  std::vector<sfp> &y, int ldy, int dimBlock) {
	
  auto xx = x.data();
  //auto yy = tmp.data();

  sfp *vals = m_vec_sVALS.data();
  int *rows = this->m_vec_ROW_INDEX.data();
  int *cols = this->m_vec_COL_INDEX.data();

	tmp.resize(this->m_dim*dimBlock);
	std::fill(tmp.begin(), tmp.end(), 0);
	
	int vec_size = static_cast<int>(m_vec_sVALS.size());
	for( auto k = 0; k < dimBlock; k++ ) {
		for( auto i = 0; i < vec_size; i++ ) {
			tmp[rows[i] + k*ldy] += vals[i]*xx[cols[i]+k*ldx];
		}
	}
	y = tmp;

}
