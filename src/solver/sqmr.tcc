template<class fp>
mpjd::SQMR<fp>::SQMR(Matrix<fp> &mat_, 
    std::shared_ptr<std::vector<fp>> Q_, int &ldQ_,
    std::shared_ptr<std::vector<fp>> L_, 
    std::vector<fp> &R_, int &ldR_,
    std::vector<fp> &Qlocked_, int &ldQlocked_)
    : mat(mat_),
      Q{Q_}, ldQ(ldQ_),
      L{L_},
      R(R_), ldR(ldR_),
      Qlocked(Qlocked_), ldQlocked(ldQlocked){}
	
template<class fp>
std::vector<fp> mpjd::SQMR<fp>::solve(int &iters){

  iters = 0;
  return R;
}

template<class fp,class sfp>
mpjd::ScaledSQMR<fp,sfp>::ScaledSQMR(Matrix<fp> &mat_, 
    std::shared_ptr<std::vector<fp>> Q_, int &ldQ_,
    std::shared_ptr<std::vector<fp>> L_, 
    std::vector<fp> &R_, int &ldR_,
    std::vector<fp> &Qlocked_, int &ldQlocked_, LinearAlgebra &la_) 
    : SQMR<fp>(mat_,Q_,ldQ_,L_,R_,ldR_,Qlocked_,ldQlocked_),
      mat(new ScaledMatrix<fp,sfp>(mat_)),
      la(la_){
    
    // Ritz vectors
    sQ.reserve(this->Q->capacity()); ldsQ = this->ldQ; 
    sQ.insert(sQ.begin(),sQ.capacity(),static_cast<sfp>(0.0));
    
    // Locked Ritz vectors
    sQlocked.reserve(this->Qlocked.capacity()); ldsQlocked = ldQlocked_; 
    sQlocked.insert(sQlocked.begin(),sQlocked.capacity(),static_cast<sfp>(0.0));
    
	  sR.reserve(mat->Dim()); ldsR = this->ldR;
    sR.insert(sR.begin(),sR.capacity(),static_cast<sfp>(0.0));
    

    XX.reserve(this->R.capacity()); 
    
    /* lower precision data area */
    x.reserve(mat->Dim());                    
    x.insert(x.begin(),x.capacity(),static_cast<sfp>(0.0));

    delta.reserve(mat->Dim());
    delta.insert(delta.begin(),delta.capacity(),static_cast<sfp>(0.0));
    
    r.reserve(mat->Dim());
    r.insert(r.begin(), r.capacity(),static_cast<sfp>(0.0));
    
    d.reserve(mat->Dim());
    d.insert(d.begin(),d.capacity(),static_cast<sfp>(0.0));
    
    w.reserve(mat->Dim());
    w.insert(w.begin(),w.capacity(),static_cast<sfp>(0.0));

    QTd.reserve(this->L->capacity());// L.size() = numEvals;                    
    QTd.insert(QTd.begin(),QTd.capacity(),static_cast<sfp>(0.0));

    
}
      

template<class fp, class sfp>
std::vector<fp> mpjd::ScaledSQMR<fp,sfp>::solve(int &iters){
  /* TODO: Implement sQMR Algorithm Steps in Scaled Matrix */


  /* Casting input matrices into the inner loop precision */ 
  auto to_sfp = [](const fp d){return static_cast<sfp>(d);};
  auto to_fp  = [](const sfp d){return static_cast<fp>(d);};
  
  // cast to solver precision 
  sQ.clear(); sQ.resize(this->Q->size()); ldsQ = this->ldQ;
  std::transform(this->Q->begin(),this->Q->end(),sQ.begin(),to_sfp);
  
  //ldsQlocked = this->ldQlocked;
  sQlocked.clear(); sQlocked.resize(this->Qlocked.size());  
  std::transform(this->Qlocked.begin(),this->Qlocked.end(),
      sQlocked.begin(),to_sfp);

  /* clear previous solution */
  XX.clear();
  
  auto iDR  = this->R;
  int ldiDR = this->ldR;
  /* Call the actual solver */
  for(auto i=0; i < this->L->size(); i++){
  
    /* Change Matrix shift*/
    mat->update_matrix_shift(*(this->L->data()+i));
   
    /* Casting input vectors into the inner loop precision */ 
    /* cast to solver precision ONLY the vectors to be used*/  
    mat->applyScalInvMat(iDR.data()+ldiDR*i,ldiDR,mat->Dim(),1); // DR = D\R
    
    auto rb   = iDR.begin();
    auto ldrb = ldiDR;

    // scale to right hand side vector
    auto nrmR = la.nrm2(mat->Dim(), iDR.data()+(0+i*ldrb),1);
    nrmR = static_cast<sfp>(1)/nrmR;
    la.scal(mat->Dim(), nrmR, iDR.data()+(0+i*ldrb), 1);

    // cast to solver precision 
    sR.clear(); sR.resize(mat->Dim());
    std::transform(rb + (0+i*ldrb),rb + (0+(i+1)*ldrb), sR.begin(),to_sfp);

    // at this point right hand side vectors are 
    // up to date with the diagonal scal
    int innerIters = solve_eq();
    iters += innerIters;

    /* Casting output vectors into the outer loop precision */  
    XX.resize(XX.size()+mat->Dim());
    std::transform(x.begin(),x.end(), XX.end()-mat->Dim(),to_fp);
  }
  // DR = D\R
  mat->applyScalInvMat(XX.data(),mat->Dim(),mat->Dim(),this->L->size()); 
  return XX;
}

template<class fp, class sfp>
int mpjd::ScaledSQMR<fp,sfp>::solve_eq(){

    auto dim = mat->Dim();
    auto numEvals = this->L->size();
    
    int numLocked = sQlocked.size()/dim;

    // this should be input in this function
    sfp ita    = static_cast<sfp>(0.0);
    sfp thita_ = static_cast<sfp>(0.0);
    int qmrMaxIt  = 1000;//dim;//1000;
    double tol    = 1e-03;

    sfp Thita_  = static_cast<sfp>(0.0);
    sfp rho_    = static_cast<sfp>(0.0);
    sfp sigma   = static_cast<sfp>(0.0);
    sfp alpha   = static_cast<sfp>(0.0);  
    sfp normr   = static_cast<sfp>(0.0);
    sfp Thita   = static_cast<sfp>(0.0);
    sfp c       = static_cast<sfp>(0.0);
    sfp g_      = static_cast<sfp>(0.0);
    sfp r00     = static_cast<sfp>(0.0);
    sfp rho     = static_cast<sfp>(0.0);
    sfp vita    = static_cast<sfp>(0.0);
    sfp g       = static_cast<sfp>(0.0);
    sfp gama    = static_cast<sfp>(0.0);
    sfp xi      = static_cast<sfp>(0.0);
    sfp normt   = static_cast<sfp>(0.0); 
    sfp f       = static_cast<sfp>(0.0);
    sfp p       = static_cast<sfp>(0.0);
    sfp thita   = static_cast<sfp>(0.0);
    sfp pk      = static_cast<sfp>(0.0);
    sfp rkm     = static_cast<sfp>(0.0);
    sfp scalw   = static_cast<sfp>(0.0);
      

    sfp minus_alpha = static_cast<sfp>(0.0);
    sfp deltaScal1  = static_cast<sfp>(0.0);
    sfp deltaScal2  = static_cast<sfp>(0.0);
      
    sfp BITA   = static_cast<sfp>(0.0);
    sfp DELTA  = static_cast<sfp>(0.0);
    sfp GAMA   = static_cast<sfp>(0.0);
    sfp FI     = static_cast<sfp>(0.0);
    sfp PSI    = static_cast<sfp>(0.0);


    sfp minus_one  = static_cast<sfp>(-1.0);
    sfp one        = static_cast<sfp>( 1.0);
    sfp zero       = static_cast<sfp>( 0.0);

    
    std::vector<sfp>& b = sR;
    std::vector<sfp>& t = x; 
    
    t.clear();     t.insert(t.begin(),t.capacity(),static_cast<sfp>(0.0));
    
    delta.clear(); 
    delta.insert(delta.begin(), delta.capacity(),static_cast<sfp>(0.0));
    
    r.clear();   r.insert(r.begin(), r.capacity(),static_cast<sfp>(0.0));
    d.clear();   d.insert(d.begin(),d.capacity(),static_cast<sfp>(0.0));
    w.clear();   w.insert(w.begin(),w.capacity(),static_cast<sfp>(0.0));
    QTd.clear(); QTd.insert(QTd.begin(),QTd.capacity(),static_cast<sfp>(0.0));
    int ldQTd = QTd.capacity(); 
    
    /* r = -b */
    r = b;
    la.scal(dim,minus_one,r.data(),1);
    
    d = r;

    /* g = norm(r)*/
    g = la.nrm2(dim,b.data(),1);

    
    rho_ = la.dot(dim,r.data(),1,d.data(),1);
    
    auto iter = 0;
    for(iter; iter < qmrMaxIt; iter++){
        
        /* d = d - QQTd */
        la.gemm('T','N',numEvals,1,dim,one,
            sQ.data(),ldsQ,d.data(),dim,
            zero,QTd.data(),ldQTd);

        la.gemm('N','N',dim,1,numEvals,minus_one,
            sQ.data(),ldsQ,QTd.data(),ldQTd,
            one,d.data(),dim);         

        /* w = A*d*/                              
        mat->matVec(d,dim,w,dim,1);                                                     
        
        /* w = w-VVTw */
        la.gemm('T','N',numEvals,1,dim,one,
            sQ.data(), ldsQ,w.data(),dim,
            zero,QTd.data(),numEvals);

        la.gemm('N','N',dim,1,numEvals,minus_one,
            sQ.data(),ldsQ,QTd.data(),numEvals,
            one,w.data(),dim);

        /* sigma = d'*w */
        sigma = la.dot(dim,d.data(),1,w.data(),1);

        /* alpha = rho_/sigma */
        alpha = rho_/sigma;                       

        /* r = r -alpha*w */
        minus_alpha = minus_one*alpha;
        la.axpy(dim,minus_alpha,w.data(),1,r.data(),1);
        
        /* Thita = norm(r)/g */
        normr = la.nrm2(dim,r.data(),1);
        Thita = normr/g;
        
        /* c = 1./sqrt(1+Thita*Thita) */
        c = std::sqrt(one/(one+Thita*Thita));
        
        g =g*Thita*c;

        if(iter == 0){
           g_ = g;
        }
        
        /* delta = (c*c*alpha)*d + (c*c*Thita_*Thita_)*delta  */
        deltaScal1 = c*c*Thita_*Thita_;   
        deltaScal2 = c*c*alpha;
        la.scal(dim,deltaScal1,delta.data(),1);
        la.axpy(dim,deltaScal2,d.data(),1,delta.data(),1);
        
        /*  t  = t + delta */
        la.axpy(dim,one,delta.data(),1,t.data(),1);

        if(abs(g) < tol){
           //break; // TODO: it stops too early
        }

        gama = c*c*Thita_; 
        xi = c*c*alpha;    
        
        normt = la.nrm2(dim,r.data(),1); 
        
        f = 1 + normt*normt; 
        PSI = gama*(PSI + FI);
        FI = gama*gama*FI + xi*xi*sigma;
        GAMA = GAMA  + 2*PSI + FI;
        
        DELTA = gama*DELTA - xi*rho_;
        BITA = BITA + DELTA;
        p = ((thita_-ita+2*BITA+GAMA))/f;
        thita = ita+p;
        
        pk = (((thita_)-ita+BITA)*((thita_)-ita+BITA))/f - p*p;
        rkm = std::sqrt(((g)*(g))/f + pk);
        
        if(iter==0){
         r00 = rkm;
        }

        if(rho_ < tol){
        //   break; // TODO: it stops too early
        }

        rkm = sqrt(g*g/f);
        
        if( (g < rkm*std::max(static_cast<fp>(0.99 * std::sqrt(f)),static_cast<fp>(std::sqrt(g/g_))))
                 || (thita > thita_) || rkm<0.1*r00  || g < tol || rkm < tol){
          break; 
       }

       /*  w = r./norm(r); */
       w = r;
       scalw = la.nrm2(dim,w.data(),1); 
       scalw = one/scalw;
       la.scal(dim,scalw,w.data(),1); 
      
       /* rho = r'*w; */
       rho = la.dot(dim,r.data(),1,w.data(),1);
       vita = rho/rho_;
      
       /* d = w + vita*d; */
       la.scal(dim,vita,d.data(),1);
       la.axpy(dim,one,w.data(),1,d.data(),1);
      
        thita_ = thita;
        Thita_ = Thita;
        rho_ = rho;
        g_ = g;

    }
    
    return iter;
}



















