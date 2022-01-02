/* CHOOSE SPARSE MATVEC ALGORITHM TO USE */
template <class fp>
void mpjd::Matrix<fp>::matVec(std::vector<fp> &x, int ldx,std::vector<fp> &y, int ldy, int dimBlock){
	switch (m_DS){
		case sparseDS_t::COO:
			matVec_COO(x,ldx,y,ldy,dimBlock);
			break;
		default:
			std::cout << "This should not have been reached" << std::endl;
			break;
		};
}

template <class fp>
void mpjd::Matrix<fp>::matVec(fp *x, int ldx,fp *y, int ldy, int dimBlock){
	switch (m_DS){
		case sparseDS_t::COO:
			matVec_COO(x,ldx,y,ldy,dimBlock);
			break;
		default:
			std::cout << "This should not have been reached" << std::endl;
			break;
		};
}


// COO implementation of y = Ax
template <class fp>
void mpjd::Matrix<fp>::matVec_COO(std::vector<fp> &x, int ldx,std::vector<fp> &y, int ldy, int dimBlock){
	
	auto xx = x.data();
	auto yy = y.data();

	fp  *vals = m_vec_VALS.data();
	int *rows = m_vec_ROW_INDEX.data();
	int *cols = m_vec_COL_INDEX.data();

	for(auto k=0;k<dimBlock;k++){
		for(auto i=0;i<m_vec_VALS.size();i++){
			yy[cols[i]] += vals[i]*xx[rows[i]];
		}
	}
}

template <class fp>
void mpjd::Matrix<fp>::matVec_COO(fp *x, int ldx,fp *y, int ldy, int dimBlock){
	
	auto xx = x;
	auto yy = y;

	fp  *vals = m_vec_VALS.data();
	int *rows = m_vec_ROW_INDEX.data();
	int *cols = m_vec_COL_INDEX.data();

	for(auto k=0;k<dimBlock;k++){
		for(auto i=0;i<m_vec_VALS.size();i++){
			yy[cols[i]] += vals[i]*xx[rows[i]];
		}
	}
}

