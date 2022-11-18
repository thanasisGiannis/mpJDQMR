
template<class fp,class sfp>
mpjd::BlockScaledSQMR<fp,sfp>::BlockScaledSQMR(Matrix<fp> &mat_, 
          std::shared_ptr<std::vector<fp>> Q_, int &ldQ_,
	        std::shared_ptr<std::vector<fp>> L_, 
	        std::shared_ptr<std::vector<fp>> R_, int &ldR_,
	        std::shared_ptr<std::vector<fp>> Qlocked_, int &ldQlocked_, 
	        LinearAlgebra &la_) 
	        : ScaledSQMR<fp,sfp>(mat_, Q_, ldQ_, L_, R_, ldR_, 
	                             Qlocked_, ldQlocked_, la_)
	        ,la(la_){
	        
	        
    auto nrhs = this->L->size();
    auto dim  = this->mat->Dim();
    
    // Ritz vectors
    this->sQ.reserve(this->Q->capacity()); this->ldsQ = this->ldQ; 
    this->sQ.insert(this->sQ.begin(),
                    this->sQ.capacity(),static_cast<sfp>(0.0));
    
    // Locked Ritz vectors
    this->sQlocked.reserve(this->Qlocked->capacity()); 
    this->ldsQlocked = ldQlocked_; 
    
    this->sQlocked.insert(this->sQlocked.begin(),
                          this->sQlocked.capacity(),static_cast<sfp>(0.0));
    
	 
    
    /*
      solving A*XX = -R;
    */
    this->XX.reserve(this->R->capacity()); // output of full precision
    
    
    /* lower precision data area 
    
      solving Ax = -sR;
    */
    this->x.reserve(dim*nrhs); ldx = dim;                                             
    this->x.insert(this->x.begin(),this->x.capacity(),static_cast<sfp>(0.0)); 
    this->sR.reserve(dim*nrhs); this->ldsR = this->ldR;          
    this->sR.insert(this->sR.begin(),this->sR.capacity(),static_cast<sfp>(0.0));  


                
          
	  // VECTORS TO BE USED IN THE BLOCK SOLVER
    /* sQMR step */ 
    p0.reserve(dim*nrhs); // dim x rhs
    p0.insert(p0.begin(), p0.capacity(),static_cast<sfp>(0.0));
    ldp0 = dim; 
    
    
    p1.reserve(dim*nrhs); // dim x rhs
    p1.insert(p1.begin(), p1.capacity(),static_cast<sfp>(0.0));
    ldp1 = dim; 
    
    p2.reserve(dim*nrhs); // dim x rhs
    p2.insert(p2.begin(), p2.capacity(),static_cast<sfp>(0.0));
    ldp2 = dim; 


    /* lanczos step */
    v1.reserve(dim*nrhs); // dim x rhs
    v1.insert(v1.begin(), v1.capacity(),static_cast<sfp>(0.0));
    ldv1 = dim; 


    v2.reserve(dim*nrhs); // dim x rhs
    v2.insert(v2.begin(), v2.capacity(),static_cast<sfp>(0.0));
    ldv2 = dim; 

    v3.reserve(dim*nrhs); // dim x rhs
    v3.insert(v3.begin(), v3.capacity(),static_cast<sfp>(0.0));
    ldv3 = dim; 

/*
    w.reserve(dim*nrhs); // dim x rhs
    w.insert(w.begin(), w.capacity(),static_cast<sfp>(0.0));
    ldw = dim; 
*/
    alpha.reserve(nrhs*nrhs); // rhs x rhs
    alpha.insert(alpha.begin(), alpha.capacity(),static_cast<sfp>(0.0));
    ldalpha = nrhs; 

    vita.reserve(nrhs*nrhs); // rhs x rhs
    vita.insert(vita.begin(), vita.capacity(),static_cast<sfp>(0.0));
    ldvita = nrhs; 

    vita2.reserve(nrhs*nrhs); // rhs x rhs
    vita2.insert(vita2.begin(), vita2.capacity(),static_cast<sfp>(0.0));
    ldvita2 = nrhs; 



    /* sQMR matrix */
    b0.reserve(nrhs*nrhs); // rhs x rhs
    b0.insert(b0.begin(), b0.capacity(),static_cast<sfp>(0.0));
    ldb0 = nrhs; 

    d0.reserve(nrhs*nrhs); // rhs x rhs
    d0.insert(d0.begin(), d0.capacity(),static_cast<sfp>(0.0));
    ldd0 = nrhs; 

    a1.reserve(nrhs*nrhs); // rhs x rhs
    a1.insert(a1.begin(), a1.capacity(),static_cast<sfp>(0.0));
    lda1 = nrhs; 

    b1.reserve(nrhs*nrhs); // rhs x rhs
    b1.insert(b1.begin(), b1.capacity(),static_cast<sfp>(0.0));
    ldb1 = nrhs; 

    c1.reserve(nrhs*nrhs); // rhs x rhs
    c1.insert(c1.begin(), c1.capacity(),static_cast<sfp>(0.0));
    ldc1 = nrhs; 

    d1.reserve(nrhs*nrhs); // rhs x rhs
    d1.insert(d1.begin(), d1.capacity(),static_cast<sfp>(0.0));
    ldd1 = nrhs; 


    /* lanczos matrix */
    ita.reserve(nrhs*nrhs); // rhs x rhs
    ita.insert(ita.begin(), ita.capacity(),static_cast<sfp>(0.0));
    ldita = nrhs; 

    thita.reserve(nrhs*nrhs); // rhs x rhs
    thita.insert(thita.begin(), thita.capacity(),static_cast<sfp>(0.0));
    ldthita = nrhs; 

    zita.reserve(nrhs*nrhs); // rhs x rhs
    zita.insert(zita.begin(), zita.capacity(),static_cast<sfp>(0.0));
    ldzita = nrhs; 

    zita_.reserve(nrhs*nrhs); // rhs x rhs
    zita_.insert(zita_.begin(), zita_.capacity(),static_cast<sfp>(0.0));
    ldzita_ = nrhs; 

    /* updating solution sQMR*/
    tau_.reserve(nrhs*nrhs); // rhs x rhs
    tau_.insert(tau_.begin(), tau_.capacity(),static_cast<sfp>(0.0));
    ldtau_ = nrhs; 

    tau.reserve(nrhs*nrhs); // rhs x rhs
    tau.insert(tau.begin(), tau.capacity(),static_cast<sfp>(0.0));
    ldtau = nrhs; 


    /* helping vector for Givens Rotations */
    z.reserve(nrhs*nrhs); // rhs x rhs
    z.insert(z.begin(), z.capacity(),static_cast<sfp>(0.0));
    ldz = nrhs; 

    
    // THIS MAYBE WILL NOT BE USED
    // we zero their capacity
    // for memory optimization

    this->delta.clear();
    this->delta.reserve(0);
    //delta.insert(delta.begin(),delta.capacity(),static_cast<sfp>(0.0));
    
    this->r.clear();
    this->r.reserve(0);
    
    this->d.clear();
    this->d.reserve(0);
    
    this->w.clear();
    this->w.reserve(0);

    this->QTd.clear();
    this->QTd.reserve(0);// L.size() = numEvals;        
       
};


template<class fp,class sfp>
int mpjd::BlockScaledSQMR<fp,sfp>::solve_eq(){
  /*
    This will be the function that implements the 
    Block sQMR method.

    should inherit data vectors from parent class
    which are already initialized at the solve() function
  */
#if 1  
  
  

  auto nrhs = this->L->size();
  auto dim  = this->mat->Dim();
  
  /*
    initiliazing all needed vectors to proper values
  */
  auto& sR = this->sR;
  auto& x  = this->x;
  memset(x.data(), 0, dim*nrhs*sizeof(sfp));

  /* sQMR step */ 
  memset(p0.data(), 0, dim*nrhs*sizeof(sfp));
  memset(p1.data(), 0, dim*nrhs*sizeof(sfp));
  memset(p2.data(), 0, dim*nrhs*sizeof(sfp));
  
  /* lanczos step */
  memset(v1.data(), 0, dim*nrhs*sizeof(sfp));
  memset(v2.data(), 0, dim*nrhs*sizeof(sfp));
  memset(v3.data(), 0, dim*nrhs*sizeof(sfp));

  
  memset(alpha.data(), 0, alpha.capacity()*sizeof(sfp));
  memset(vita.data(),  0, vita.capacity()*sizeof(sfp));
  memset(vita2.data(), 0, vita2.capacity()*sizeof(sfp));


  /* sQMR matrix */
  memset(b0.data(), 0, b0.capacity()*sizeof(sfp));
  memset(d0.data(), 0, d0.capacity()*sizeof(sfp));

  memset(a1.data(), 0, a1.capacity()*sizeof(sfp));
  memset(b1.data(), 0, b1.capacity()*sizeof(sfp));
  memset(c1.data(), 0, c1.capacity()*sizeof(sfp));
  memset(d1.data(), 0, d1.capacity()*sizeof(sfp));


  /* lanczos matrix */
  memset(ita.data(),   0, ita.capacity()*sizeof(sfp));
  memset(thita.data(), 0, thita.capacity()*sizeof(sfp));
  memset(zita.data(),  0, zita.capacity()*sizeof(sfp));
  memset(zita_.data(), 0, zita_.capacity()*sizeof(sfp));


  /* updating solution sQMR*/
  memset(tau_.data(), 0, tau_.capacity()*sizeof(sfp));
  memset(tau.data(),  0, tau.capacity()*sizeof(sfp));


  /* helping vector for Givens Rotations */
  memset(z.data(),  0, z.capacity()*sizeof(sfp));

  /* matrices set to be equal to identity */
  // d0 = I;
  for(auto i=0; i<nrhs; i++){
    d0[i + ldd0*i] = 1;
  }

  // a1 = I;
  for(auto i=0; i<nrhs; i++){
    a1[i + lda1*i] = 1;
  }
  
  // d1 = I;
  for(auto i=0; i<nrhs; i++){
    d1[i + ldd1*i] = 1;
  }
  




  //v3 = b;%-A*x;
  v3 = sR;
  //[v3,vita] = qr(v3,0);
  orth_v3_update_vita();

  //tau2_ = vita;
  



  
  for(auto i=0; i<nrhs; i++){
    for(auto j=0; j<nrhs; j++){
      std::cout << "vita(" << i+1 << "," << j+1 << ") = " << vita[i+j*lda1] << std::endl;
    }
  }

  
   
#endif


  //auto& x  = this->x;
  //auto& sR = this->sR;

  x = sR;
  return 0;

}

template<class fp, class sfp>
void mpjd::BlockScaledSQMR<fp,sfp>::orth_v3_update_vita(){

  auto VV = static_cast<sfp*>(v3.data());
  int ldV = ldv3;
  
  int rows = this->mat->Dim();
  int cols = this->L->size();

  /* V = orth(V) */
  for(int j=0; j < cols; j++){
	  for(int i=0; i < j ; i++){
	      // alpha = V(i)'v
			  vita[i+j*ldvita] = -la.dot(rows,&VV[0+i*ldV],1,&VV[0+j*ldV],1);
			  // v = v - V(i)*alpha
			  la.axpy(rows,vita[j+i*ldvita],&VV[0+i*ldV],1,&VV[0+j*ldV],1); 
	  }	
	  // v = v/norm(v)
	  vita[j+j*ldvita] = la.nrm2(rows,&VV[0+j*ldV],1);
  }

  
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

