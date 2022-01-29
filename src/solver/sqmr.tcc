template<class fp>
mpjd::SQMR<fp>::SQMR(Matrix<fp> &mat_, std::vector<fp> &Q_, int &ldQ_,
	      std::vector<fp> &L_, std::vector<fp> &R_, int &ldR_,
	      std::vector<fp> &Qlocked_, int &ldQlocked_)
	      : mat(mat_),
	        Q(Q_), ldQ(ldQ_),
	        L(L_),
	        R(R_), ldR(ldR_),
          Qlocked(Qlocked_), ldQlocked(ldQlocked)       
{
}
	

template<class fp>
std::vector<fp> mpjd::SQMR<fp>::solve(){
  return R;
}



template<class fp,class sfp>
mpjd::ScaledSQMR<fp,sfp>::ScaledSQMR(Matrix<fp> &mat_, std::vector<fp> &Q_, int &ldQ_,
      std::vector<fp> &L_, std::vector<fp> &R_, int &ldR_,
      std::vector<fp> &Qlocked_, int &ldQlocked_) 
      : SQMR<fp>(mat_,Q_,ldQ_,L_,R_,ldR_,Qlocked_,ldQlocked_),
       mat(new ScaledMatrix<fp,sfp>(mat_))

{
  
}
      

template<class fp, class sfp>
std::vector<fp> mpjd::ScaledSQMR<fp,sfp>::solve(){
  /* TODO: Implement sQMR Algorithm Steps in Scaled Matrix */
  
  /* For loop to iterate through unlocked ritz pairs in order to solve correction equation */
  for(auto i=0; i< this->L.size(); i++){
    mat->update_matrix_shift(0);
  }
  return this->R;
}


