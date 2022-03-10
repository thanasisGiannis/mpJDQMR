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
      std::vector<fp> &Qlocked_, int &ldQlocked_, LinearAlgebra &la_) 
      : SQMR<fp>(mat_,Q_,ldQ_,L_,R_,ldR_,Qlocked_,ldQlocked_)
      ,mat(new ScaledMatrix<fp,sfp>(mat_))
      ,la(la_)

{
    
    sQ.reserve(this->Q.capacity()); ldsQ = this->ldQ; // Ritz vectors
    sQ.insert(sQ.begin(),sQ.capacity(),static_cast<fp>(0.0));
    
    sQlocked.reserve(this->Qlocked.capacity()); ldsQlocked; // Locked Ritz vectors
    sQlocked.insert(sQlocked.begin(),sQlocked.capacity(),static_cast<fp>(0.0));
    
	  sR.reserve(mat->Dim()); ldsR = this->ldR;
    sR.insert(sR.begin(),sR.capacity(),static_cast<fp>(0.0));
    

    XX.reserve(this->R.capacity()); 
    
    
    x.reserve(mat->Dim());                    
    x.insert(x.begin(),x.capacity(),static_cast<sfp>(0.0));

    /* Inner vectors */
    v1.reserve(mat->Dim()); // Lanczos vector
    v1.insert(v1.begin(),v1.capacity(),static_cast<sfp>(0.0));
    
    v2.reserve(mat->Dim()); // Lanczos vector
    v2.insert(v2.begin(),v2.capacity(),static_cast<sfp>(0.0));
 
    p0.reserve(mat->Dim()); // sQMR vector
    p0.insert(p0.begin(),p0.capacity(),static_cast<sfp>(0.0));

    p1.reserve(mat->Dim()); // sQMR vector
    p1.insert(p1.begin(),p1.capacity(),static_cast<sfp>(0.0));

    w.reserve(mat->Dim());  // new direction to sqmr iter 
    w.insert(w.begin(),w.capacity(),static_cast<sfp>(0.0));

}
      

template<class fp, class sfp>
std::vector<fp> mpjd::ScaledSQMR<fp,sfp>::solve(){
  /* TODO: Implement sQMR Algorithm Steps in Scaled Matrix */


  /* Casting input matrices into the inner loop precision */ 
  auto to_sfp = [](const fp d){return static_cast<sfp>(d);};
  auto to_fp  = [](const sfp d){return static_cast<fp>(d);};
  
  sQ.clear(); sQ.resize(this->Q.size());
  std::transform(this->Q.begin(),this->Q.end(),sQ.begin(),to_sfp);
  sQlocked.clear(); sQlocked.resize(this->Qlocked.size());
  std::transform(this->Qlocked.begin(),this->Qlocked.end(),sQlocked.begin(),to_sfp);
  
  
//  std::transform(this->L.begin(),this->L.end(),sL.begin(),to_sfp);
//  sL.resize(this->L.size());


  /* clear previous solution */
  XX.clear();
  
  auto iDR  = this->R;
  int ldiDR = this->ldR;

  /* Call the actual solver */
  for(auto i=0; i < this->L.size(); i++){
    /* Change Matrix shift*/
    mat->update_matrix_shift(*(this->L.data()+i));
   
    /* Casting input vectors into the inner loop precision */ 
    /* cast to solver precision ONLY the vectors to be used*/  
    
    mat->applyScalInvMat(iDR.data()+ldiDR*i,ldiDR,mat->Dim(),1); // DR = D\R
    
    
    auto rb   = iDR.begin();
    auto ldrb = ldiDR;
    sR.clear(); sR.resize(mat->Dim());
    std::transform(rb + (0+i*ldrb),rb + (0+(i+1)*ldrb), sR.begin(),to_sfp);


    // at this point right hand side vectors are up to date with the diagonal sca
    
    x = sR;    
    /* Casting output vectors into the outer loop precision */  
    XX.resize(XX.size()+mat->Dim());
    std::transform(x.begin(),x.end(), XX.end()-mat->Dim(),to_fp);
    //mat->applyScalInvMat(XX,mat->Dim(),mat->Dim(),this->L.size()); // DR = D\R
    
  }
    
  
  
  std::cout << std::endl;
  std::cout << sR.size() << std::endl;
  std::cout << x.size() << std::endl;  
  std::cout << XX.size() << std::endl;
/*
  std::cout << "D: " << std::setprecision(16) << *(this->L.data())  << std::endl;
  std::cout << "F: " << std::setprecision(16) << *sL.data() << std::endl;

  std::cout << std::endl;
*/
//  w = this->R;  
  return XX;
}




















