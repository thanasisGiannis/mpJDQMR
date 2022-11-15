
template<class fp,class sfp>
mpjd::BlockScaledSQMR<fp,sfp>::BlockScaledSQMR(Matrix<fp> &mat_, std::shared_ptr<std::vector<fp>> Q_, int &ldQ_,
	        std::shared_ptr<std::vector<fp>> L_, 
	        std::shared_ptr<std::vector<fp>> R_, int &ldR_,
	        std::shared_ptr<std::vector<fp>> Qlocked_, int &ldQlocked_, 
	        LinearAlgebra &la_) : ScaledSQMR<fp,sfp>(mat_, Q_, ldQ_,
	        L_, R_, ldR_, Qlocked_, ldQlocked_, la_){};


template<class fp,class sfp>
int mpjd::BlockScaledSQMR<fp,sfp>::solve_eq(){
  /*
    This will be the function that implements the 
    Block sQMR method.

    should inherit data vectors from parent class
    which are already initialized at the solve() function
  */
  auto& x  = this->x;
  auto& sR = this->sR;

  x = sR;
  return 0;
}


template<class fp, class sfp>
std::vector<fp> mpjd::BlockScaledSQMR<fp,sfp>::solve(int &iters){
  
 
  auto& mat      = this->mat;
  auto& la       = this->la;
  auto& sR       = this->sR;
  auto& x        = this->x;
  auto& sQ       = this->sQ; auto ldsQ = this->ldsQ;
  auto& sQlocked = this->sQlocked;  
  auto& R        = this->R; auto ldR = this->ldR;
  auto& XX       = this->XX;
  

  /* Casting input matrices into the inner loop precision */ 
  auto to_sfp = [](const fp d){return static_cast<sfp>(d);};
  auto to_fp  = [](const sfp d){return static_cast<fp>(d);};
  
  // cast to solver precision 
  sQ.clear(); sQ.resize(this->Q->size()); ldsQ = this->ldQ;
  std::transform(this->Q->begin(),this->Q->end(),sQ.begin(),to_sfp);
  
  //ldsQlocked = this->ldQlocked;
  sQlocked.clear(); sQlocked.resize(this->Qlocked->size());  
  std::transform(this->Qlocked->begin(),this->Qlocked->end(),
      sQlocked.begin(),to_sfp);

  /* clear previous solution */
  auto iDR  = this->R;
  int ldiDR = this->ldR;
  
  XX.clear();
  sR.clear();   
  /* Call the actual solver */
  for(auto i=0; i < this->L->size(); i++){
  
    /* Change Matrix shift*/
    mat->update_matrix_shift(*(this->L->data()+i));
   
    /* Casting input vectors into the inner loop precision */ 
    /* cast to solver precision ONLY the vectors to be used*/  
    mat->applyScalInvMat(iDR->data()+ldiDR*i,ldiDR,mat->Dim(),1); // DR = D\R
    
    auto rb   = iDR->begin();
    auto ldrb = ldiDR;

    // scale to right hand side vector
    auto nrmR = la.nrm2(mat->Dim(), iDR->data()+(0+i*ldrb),1);
    nrmR = static_cast<sfp>(1)/nrmR;
    la.scal(mat->Dim(), nrmR, iDR->data()+(0+i*ldrb), 1);

    // cast to solver precision 
    sR.resize(sR.size() + mat->Dim());
    std::transform(rb + (0+i*ldrb),rb + (0+(i+1)*ldrb), sR.data() + ldR*i,to_sfp);
  }
  // at this point right hand side vectors are 
  // up to date with the diagonal scal
  int innerIters = solve_eq();
  iters += innerIters;
  
  /* Casting output vectors into the outer loop precision */  
  XX.resize(x.size());
  std::transform(x.begin(),x.end(), XX.begin(),to_fp);
  // DR = D\R
  mat->applyScalInvMat(XX.data(),mat->Dim(),mat->Dim(),this->L->size()); 

  return XX;
}

