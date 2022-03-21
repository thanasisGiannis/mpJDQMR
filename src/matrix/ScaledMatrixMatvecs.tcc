/* CHOOSE SPARSE MATVEC ALGORITHM TO USE */
template <class fp, class sfp>
void mpjd::ScaledMatrix<fp, sfp>
::matVec(std::vector<sfp> &x, int ldx,std::vector<sfp> &y, int ldy, int dimBlock){
	switch (this->m_DS){
		case sparseDS_t::COO:
			matVec_COO(x,ldx,y,ldy, dimBlock);
			break;
		default:
			std::cout << "This should not have been reached" << std::endl;
			break;
		};
}

// COO implementation of y = Ax
template <class fp, class sfp>
void mpjd::ScaledMatrix<fp,sfp>
::matVec_COO(std::vector<sfp> &x, int ldx,std::vector<sfp> &y, int ldy, int dimBlock){
	
	auto xx = x.data();
	auto yy = tmp.data();

	sfp *vals = m_vec_sVALS.data();
	int *rows = this->m_vec_ROW_INDEX.data();
	int *cols = this->m_vec_COL_INDEX.data();
	
	std::fill(tmp.begin(), tmp.end(), static_cast<sfp>(0.0));
	
	for(auto k=0;k<dimBlock;k++){
		for(auto i=0;i<m_vec_sVALS.size();i++){
			yy[rows[i] + k*ldy] += vals[i]*xx[cols[i]+k*ldx];
		}
	}
	
	y = tmp;
}









#if 0

template <class fp, class sfp>
void mpjd::ScaledMatrix<fp,sfp>
::matVec(sfp *x, int ldx,sfp *y, int ldy, int dimBlock){
	switch (this->m_DS){
		case sparseDS_t::COO:
			matVec_COO(x,ldx,y,ldy,dimBlock);
			break;
		default:
			std::cout << "This should not have been reached" << std::endl;
			break;
		};
}



template <class fp, class sfp>
void mpjd::ScaledMatrix<fp,sfp>
::matVec_COO(sfp *x, int ldx,sfp *y, int ldy, int dimBlock){
	
	auto xx = x;
	auto yy = y;

	sfp *vals = m_vec_sVALS.data();
	int *rows = this->m_vec_ROW_INDEX.data();
	int *cols = this->m_vec_COL_INDEX.data();

//	std::cout << "[Matrix] " << dimBlock << std::endl;
	for(auto i=0;i<m_vec_sVALS.size();i++){
		for(auto k=0;k<dimBlock;k++){
			yy[rows[i] + k*ldy] += vals[i]*xx[cols[i]+k*ldx];
		}
	}
}
#endif

