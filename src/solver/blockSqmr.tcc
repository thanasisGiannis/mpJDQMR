/*
 * TODO: use optimal stopping criteria
 * TODO: faster initialization of the vectors 
 * TODO: orth_v3_update_vita(); // this should be an object
 */  
#include <cmath>


#define UNUSED(expr) static_cast<void>(expr)

template<class fp,class sfp>
mpjd::BlockScaledSQMR<fp,sfp>::BlockScaledSQMR(Matrix<fp> &mat_, 
          std::shared_ptr<std::vector<fp>> Q_, int &ldQ_,
	        std::shared_ptr<std::vector<fp>> L_, 
	        std::shared_ptr<std::vector<fp>> R_, int &ldR_,
	        std::shared_ptr<std::vector<fp>> Qlocked_, int &ldQlocked_, 
	        LinearAlgebra &la_) 
	        : ScaledSQMR<fp,sfp>(mat_, Q_, ldQ_, L_, R_, ldR_, 
	                             Qlocked_, ldQlocked_, la_)
	        ,la(la_)
	        ,hh(Householder<sfp>( 2*L_->size(), L_->size(), la_ ))
	        ,mgs(MGS<sfp>(mat_.Dim(), L_->size(), la_))  {
	        
	        
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

    /* matrix vector accumulator */
    y.reserve(dim*nrhs); // rhs x rhs
    y.insert(y.begin(), y.capacity(),static_cast<sfp>(0.0));
    ldy = dim; 

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

    /* updating residual sR*/
    Ap0.reserve(dim*nrhs); // dim x rhs
    Ap0.insert(Ap0.begin(), Ap0.capacity(),static_cast<sfp>(0.0));
    ldAp0 = dim; 
    
    Ap1.reserve(dim*nrhs); // dim x rhs
    Ap1.insert(Ap1.begin(), Ap1.capacity(),static_cast<sfp>(0.0));
    ldAp1 = dim; 
    
    
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
    tau_2.reserve(nrhs*nrhs); // rhs x rhs
    tau_.insert(tau_.begin(), tau_.capacity(),static_cast<sfp>(0.0));
    tau_2.insert(tau_2.begin(), tau_2.capacity(),static_cast<sfp>(0.0));
    ldtau_ = nrhs; 

    tau.reserve(nrhs*nrhs); // rhs x rhs
    tau.insert(tau.begin(), tau.capacity(),static_cast<sfp>(0.0));
    ldtau = nrhs; 

    /* helping vectors for Givens Rotations */
    hhR.reserve(2*nrhs*nrhs); // 2nrhs x rhs
    hhR.insert(hhR.begin(), hhR.capacity(),static_cast<sfp>(0.0));
    ldhhR = 2*nrhs; 

    hhQ.reserve(2*nrhs*2*nrhs); // 2nrhs x 2nrhs
    hhQ.insert(hhQ.begin(), hhQ.capacity(),static_cast<sfp>(0.0));
    ldhhQ = 2*nrhs;


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
    This is the function that implements the 
    Block sQMR method.

    should inherit data vectors from parent class
    which are already initialized at the solve() function
    
  */  
  
  
  sfp one       = static_cast<sfp>(1.0);
  sfp minus_one = static_cast<sfp>(-1.0);
  sfp zero      = static_cast<sfp>(0.0);

  auto mat  = this->mat;
  auto nrhs = static_cast<int>(this->L->size());
  auto dim  = static_cast<int>(this->mat->Dim());

  auto sR_original = this->sR;
  auto sQlocked    = this->sQlocked; auto ldsQlocked = this->ldsQlocked;
  auto numLocked   = sQlocked.size()/dim;
  
  
  /*
    initiliazing all needed vectors to proper values
  */
  
  // TODO: MAYBE THOSE SHOULD BE PASSED AS FUNCTION ATTRIBUTES
  auto& sR = this->sR; auto ldsR = this->ldsR; //sR.resize(dim*nrhs);
  auto& x  = this->x; //x.resize(dim*nrhs); 
  
  
  
  std::fill(x.begin(), x.end(),zero);
  /* matvec accumulator */ 
  std::fill(y.begin(), y.end(),zero);
  
  /* sQMR step */
  std::fill(p0.begin(), p0.end(),zero);
  std::fill(p1.begin(), p1.end(),zero);
  std::fill(p2.begin(), p2.end(),zero);


  /* updating residual sR*/
  std::fill(Ap0.begin(), Ap0.end(),zero);
  std::fill(Ap1.begin(), Ap1.end(),zero);
  
  
  /* lanczos step */
  std::fill(v1.begin(), v1.end(),zero);
  std::fill(v2.begin(), v2.end(),zero);
  std::fill(v3.begin(), v3.end(),zero);
  
  
  std::fill(alpha.begin(), alpha.end(),zero);
  std::fill(vita.begin(),  vita.end(),zero);
  std::fill(vita2.begin(), vita2.end(),zero);
  

  /* sQMR matrix */
  std::fill(b0.begin(), b0.end(),zero);
  std::fill(d0.begin(), d0.end(),zero);


  std::fill(a1.begin(), a1.end(),zero);
  std::fill(b1.begin(), b1.end(),zero);
  std::fill(c1.begin(), c1.end(),zero);
  std::fill(d1.begin(), d1.end(),zero);


  /* lanczos matrix */
  std::fill(ita.begin(), ita.end(),zero);
  std::fill(thita.begin(), thita.end(),zero);
  std::fill(zita.begin(), zita.end(),zero);
  std::fill(zita_.begin(), zita_.end(),zero);
  


  /* updating solution sQMR*/
  std::fill(tau_.begin(), tau_.end(),zero);
  std::fill(tau.begin(), tau.end(),zero);
  
  
  /*
    helping vector for Householder QR 
    WILL BE INITIATED IN THE householderQR()
  */
  std::fill(hhR.begin(), hhR.end(),zero);
  std::fill(hhQ.begin(), hhQ.end(),zero);
  
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

  // v3 = v3-Qlocked*Qlocked'*v3;
  // alpha is used as tmp
  // alpha = Qlocked'*v3
  la.gemm('T', 'N', numLocked, nrhs, dim, 
    one, sQlocked.data(), ldsQlocked, v3.data(), ldv3,
    zero, alpha.data(), ldalpha);

  // v3 = v3 - Qlocked*alpha
  la.gemm('N', 'N', dim, nrhs, numLocked, 
    minus_one, sQlocked.data(), ldsQlocked, alpha.data(), ldalpha,
    one, v3.data(), ldv3);

  //[v3,vita2] = qr(v3,0);
  if(false == mgs.QR(dim, nrhs, v3,ldv3,vita2,ldvita2)) {
    swap(x,sR);
    return 0;
  }
  
  std::swap(v2, v3);
  std::swap(vita, vita2);
  std::swap(tau_, vita);
  
  
  // main loop of the block algorithm
  int loopNum;
  sfp nrm2      = static_cast<sfp>(1.0);
  sfp nrm2_prev = static_cast<sfp>(1.0);
  int maxloopNum = 3*dim;
  for(loopNum=0; loopNum<maxloopNum; loopNum++) {
    // v3 = A*v2-v1*vita;
    
    // v3 = -v1*vita;
    la.gemm('N', 'N', dim, nrhs, nrhs, 
      minus_one, v1.data(), ldv1, vita.data(), ldvita,
      zero, v3.data(), ldv3);

        
    // y = A*v2;    
    mat->matVec(v2, ldv2, y, ldy, nrhs);  

    // v3 = y+v3;
    la.geam('N', 'N',dim, nrhs,
              one, y.data(), ldy, one, v3.data(),ldv3,
              v3.data(), ldv3);

    
    // v3 = v3-Qlocked*Qlocked'*v3;
    // alpha is used as tmp
    // alpha = Qlocked'*v3
    la.gemm('T', 'N', numLocked, nrhs, dim, 
      one, sQlocked.data(), ldsQlocked, v3.data(), ldv3,
      zero, alpha.data(), ldalpha);
 
    // v3 = v3 - Qlocked*alpha
    la.gemm('N', 'N', dim, nrhs, numLocked, 
      minus_one, sQlocked.data(), ldsQlocked, alpha.data(), ldalpha,
      one, v3.data(), ldv3);
 
   
    
    //alpha = v2'*v3;
    la.gemm('T', 'N', nrhs, nrhs, dim, 
      one, v2.data(), ldv2, v3.data(), ldv3,
      zero, alpha.data(), ldalpha);


    //v3 = v3 - v2*alpha;
    la.gemm('N', 'N', dim, nrhs, nrhs, 
      minus_one, v2.data(), ldv2, alpha.data(), ldalpha,
      one, v3.data(), ldv3);



    //[v3,vita2] = qr(v3,0);
    if(false == mgs.QR(dim, nrhs, v3, ldv3, vita2, ldvita2)) {
      if(loopNum == 0) {
        std::swap(x,sR);
      }
      break;
    }
    //thita = b0*vita;
    la.gemm('N', 'N', nrhs, nrhs, nrhs, 
      one, b0.data(), ldb0, vita.data(), ldvita,
      zero, thita.data(), ldthita);


        
    //ita = a1*d0*vita+b1*alpha;
    //tau_2 = a1*d0 // use of tau_2 is for tmp    
    la.gemm('N', 'N', nrhs, nrhs, nrhs, 
      one, a1.data(), lda1, d0.data(), ldd0,
      zero, tau_2.data(), ldita);

    //ita = tau_2*vita
    la.gemm('N', 'N', nrhs, nrhs, nrhs, 
      one, tau_2.data(), ldita, vita.data(), ldvita,
      zero, ita.data(), ldita);

    //ita = ita+b1*alpha
    la.gemm('N', 'N', nrhs, nrhs, nrhs, 
      one, b1.data(), ldb1, alpha.data(), ldalpha,
      one, ita.data(), ldita);      

    //zita_ = c1*d0*vita+d1*alpha;
    //tau_2 = c1*d0
    la.gemm('N', 'N', nrhs, nrhs, nrhs, 
      one, c1.data(), ldc1, d0.data(), ldd0,
      zero, tau_2.data(), ldzita_);


    //zita_ = tau_2*vita
    la.gemm('N', 'N', nrhs, nrhs, nrhs, 
      one, tau_2.data(), ldzita_, vita.data(), ldvita,
      zero, zita_.data(), ldzita_);
    
    //zita_ = zita_ + d1*alpha
    la.gemm('N', 'N', nrhs, nrhs, nrhs, 
      one, d1.data(), ldd1, alpha.data(), ldalpha,
      one, zita_.data(), ldzita_);


    // hhR = [zita_; vita2]
    la.geam('N', 'N', nrhs, nrhs, zero, zita_.data(), ldzita_,
              one, zita_.data(), ldzita_, hhR.data(), ldhhR);
    la.geam('N', 'N', nrhs, nrhs, zero, vita2.data(), ldvita2,
              one, vita2.data(), ldvita2, hhR.data() + nrhs, ldhhR);
    
    // [hhQ,hhR] = qr([zita_; vita2];  
    hh.QR(2*nrhs, nrhs, hhQ, ldhhQ, hhR, ldhhR);  


    // update 
    std::swap(b0,b1);
    std::swap(d0,d1);
    
    // a1 = qq(1:rhs,1:rhs)';
    la.geam('N', 'T', nrhs, nrhs, zero, hhQ.data(), ldhhQ,
              one, hhQ.data(), ldhhQ, a1.data(), lda1);
              
    // c1 = qq(1:rhs,rhs+1:2*rhs)';
    la.geam('N', 'T', nrhs, nrhs, zero, hhQ.data() + nrhs*ldhhQ , ldhhQ,
              one, hhQ.data()+ nrhs*ldhhQ, ldhhQ, c1.data(), ldc1);
              

    // b1 = qq(rhs+1:2*rhs,1:rhs)';
    la.geam('N', 'T', nrhs, nrhs, zero, hhQ.data() + nrhs , ldhhQ,
              one, hhQ.data()+ nrhs, ldhhQ, b1.data(), ldb1);

    // d1 = qq(rhs+1:2*rhs,rhs+1:2*rhs)';
    la.geam('N', 'T', nrhs, nrhs, zero, hhQ.data() + nrhs +nrhs*ldhhQ , ldhhQ,
              one, hhQ.data() + nrhs + nrhs*ldhhQ, ldhhQ, d1.data(), ldd1);


    // zita = rr(1:rhs,1:rhs);
    la.geam('N', 'N', nrhs, nrhs, zero, hhR.data(), ldhhR,
              one, hhR.data(), ldhhR, zita.data(), ldzita);
  
    //p2 = (v2-p1*ita-p0*thita)/zita;
    // p2 = v2
    p2 = v2;
    // p2 = -p1*ita + p2
    la.gemm('N', 'N', dim, nrhs, nrhs, 
        minus_one, p1.data(), ldp1, ita.data(), ldita,
        one, p2.data(), ldp2);
    
    // p2 = -p0*thita + p2
    la.gemm('N', 'N', dim, nrhs, nrhs, 
        minus_one, p0.data(), ldp0, thita.data(), ldthita,
        one, p2.data(), ldp2);


    // p2 = p2/zita  
    la.trsm('C', 'R', 'U', 'N', 'N', dim, nrhs, one, zita.data(), ldzita,
            p2.data(), ldp2);
   

    // tau = a1*tau_;
    la.gemm('N', 'N', nrhs, nrhs, nrhs, 
      one, a1.data(), lda1, tau_.data(), ldtau_,
      zero, tau.data(), ldtau);
    

    // x = x+p2*tau;
    la.gemm('N', 'N', dim, nrhs, nrhs, 
      one, p2.data(), ldp2, tau.data(), ldtau,
      one, x.data(), ldx);

    // y = -Ap1*ita + y;
    la.gemm('N', 'N', dim, nrhs, nrhs, 
      minus_one, Ap1.data(), ldAp1, ita.data(), ldita,
      one, y.data(), ldy);
    
    // y = -Ap0*thita + y;
    la.gemm('N', 'N', dim, nrhs, nrhs, 
      minus_one, Ap0.data(), ldAp0, thita.data(), ldthita,
      one, y.data(), ldy);


    // y = y/zita;
    la.trsm('C', 'R', 'U', 'N', 'N', dim, nrhs, one, zita.data(), ldzita,
            y.data(), ldy);


    // sR = sR - y*tau;
    la.gemm('N', 'N', dim, nrhs, nrhs, 
      minus_one, y.data(), ldy, tau.data(), ldtau,
      one, sR.data(), ldsR);


    /*
      if(norm(sR)<tol)
        break;
      end

    */ 
     
    // MAYBE THIS LOOP SHOULD BE IN A FUNCTION
    int nConv = 0;
    for(int i=0; i<nrhs; i++) {
      nrm2_prev = nrm2;
      nrm2 = la.nrm2(dim, sR.data() + 0+i*ldsR, 1); 
      //std::cout << i+1 << " "<<  nrm2 << std::endl;
      // TODO:  at this point we need to implement stopping criteria
      //        based on optimal convergence of jacobi davidson 
      if (nrm2 < 1e-01 || std::abs(nrm2-nrm2_prev)/std::abs(nrm2) < 1e-01 ) {
        nConv++;
      }
    }
    
    if(nConv == nrhs) {
      break;
    }
    
    
    // tau_ = c1*tau_;
    std::swap(tau_2, tau_);
    la.gemm('N', 'N', nrhs, nrhs, nrhs, 
      one, c1.data(), ldc1, tau_2.data(), ldtau_,
      zero, tau_.data(), ldtau_);

        
    // update for new loop
    std::swap(v1,v2);
    std::swap(v2,v3);

    std::swap(p0,p1);
    std::swap(p1,p2);

    std::swap(Ap0,Ap1);
    std::swap(Ap1,y);

    std::swap(vita,vita2);
        
  } // end of main loop
  
  return loopNum;
}

#if 0
template<class fp,class sfp>
mpjd::BlockScaledSQMR<fp,sfp>::
Householder::Householder(int dim, int nrhs, LinearAlgebra &la_)
  : la(la_)
{
    
    hhQz.reserve(dim*dim); // dim x dim
    hhQz.insert(hhQz.begin(), hhQz.capacity(),static_cast<sfp>(0.0));
    ldhhQz = dim;

    hhx.reserve(dim*1); // dim x 1
    hhx.insert(hhx.begin(), hhx.capacity(),static_cast<sfp>(0.0));

    hhv.reserve(dim*1); // dim x 1
    hhv.insert(hhv.begin(), hhv.capacity(),static_cast<sfp>(0.0));

    hhu.reserve(nrhs*1);   // dim x 1
    hhu.insert(hhu.begin(), hhu.capacity(),static_cast<sfp>(0.0));

    hhvhhvt.reserve(dim*dim);   // dim x dim
    hhvhhvt.insert(hhvhhvt.begin(), hhvhhvt.capacity(),static_cast<sfp>(0.0));
    ldhhvhhvt = dim;

}


template<class fp,class sfp>
void mpjd::BlockScaledSQMR<fp,sfp>::
Householder::QR(int m, int n, std::vector<sfp> &R, int ldR, 
              std::vector<sfp> &Q, int ldQ) {

  
  if(m<n) exit(-1); // tall and thin only
  
  // Q = I
  std::fill(Q.begin(), Q.end(), static_cast<sfp>(0.0));
  
  for(int i=0; i<m; i++){
    Q[i+i*ldQ] = static_cast<sfp>(1.0);
  }
  //std::vector<sfp> Qprev;
  
  for(int k=0; k<n; k++) {
    // x = zeros(m,1); 
    std::fill(hhx.begin(), hhx.end(), static_cast<sfp>(0.0));
    
    // x(k:m,1)=R(k:m,k);
    memcpy(hhx.data()+k, R.data() + k+k*ldR, sizeof(sfp)*(m-k)); 

    //g=norm(x);
    sfp g = la.nrm2(m, hhx.data(), 1);
    
    // v=x; v(k)=x(k)+g;
    hhv = hhx;
    hhv[k] = hhx[k] + g;
  
    // s=norm(v);
    sfp s = la.nrm2(m,hhv.data(),1);  
    
    // if s!=0
    // v = v/s
    la.scal(m, static_cast<sfp>(1.0)/s,hhv.data(),1);
    
    // u=R'*v;
    la.gemm('T', 'N',
      n, 1, m, static_cast<sfp>(1.0), R.data(), ldR,
      hhv.data(), m, static_cast<sfp>(0.0), hhu.data(), n);


    // R=R-2*v*u'; 
    la.gemm('N', 'T',
      m, n, 1, static_cast<sfp>(-2.0), hhv.data(), m,
      hhu.data(), n, static_cast<sfp>(1.0), R.data(), ldR);


    // Q=Q-2*Q*(v*v');
    hhQz = Q;
    //hhvhhvt = v*v';
    la.gemm('N', 'T',
      m, m, 1, static_cast<sfp>(1.0), hhv.data(), m,
      hhv.data(), m, static_cast<sfp>(0.0), hhvhhvt.data(), ldhhvhhvt);      

 
    // Q = Q-2*Q*hhvhhvt
    la.gemm('N', 'N', m, m, m, 
    static_cast<sfp>(-2.0), hhQz.data(), ldhhQz, hhvhhvt.data(), ldhhvhhvt,
    static_cast<sfp>(1.0), Q.data(), ldQ);
    
  }
  
}
#endif


#if 0
template<class fp, class sfp>
mpjd::BlockScaledSQMR<fp,sfp>::
Cholesky::Cholesky(const int dim, const int nrhs, LinearAlgebra &la_)
: la(la_) {

  B.reserve(nrhs*nrhs);
  B.resize(nrhs*nrhs);
  std::fill(B.begin(), B.end(), static_cast<sfp>(0.0));
  ldB = nrhs;
  UNUSED(dim);

}


template<class fp, class sfp>
bool mpjd::BlockScaledSQMR<fp,sfp>::
Cholesky::chol(const int n, std::vector<sfp> &L, int ldL) {
  // L = chol(B);
  
  // L = zeros(n);
  std::fill(L.begin(), L.end(), static_cast<sfp>(0.0));

/*
  [n,~] = size(A);
  for j=1:n
      sum = 0;
      for k=1:j
          sum = sum + L(k,j)*L(k,j);
      end
      L(j,j) = sqrt(A(j,j)-sum);

      for i=j+1:n
          sum = 0;
          for k=1:j
             sum = sum + L(k,i)*L(k,j); 
          end
          L(j,i) = (1/L(j,j))*(A(j,i)-sum);
      end
  end
*/
  
  sfp sum;
  for(int j=0; j<n; j++){
    sum = static_cast<sfp>(0.0);
    for(int k=0; k<j; k++){
      sum = sum+ L[k+j*ldL]*L[k+j*ldL];
    }
    L[j+j*ldL] = std::sqrt(B[j+j*ldB]-sum);
    if(std::isnan(L[j+j*ldL])) return false;
    
    for(int i=j+1; i<n; i++){
      sum = static_cast<sfp>(0.0);
      for(int k=0; k<j; k++){
        sum = sum + L[k+i*ldL]*L[k+j*ldL];
      }
      L[j+i*ldL] = (static_cast<sfp>(1.0)/L[j+j*ldL])*(B[j+i*ldB]-sum);
      if(std::isnan(L[j+i*ldL])) return false;
    }
  }
  return true;
}

template<class fp, class sfp>
bool mpjd::BlockScaledSQMR<fp,sfp>::
Cholesky::QR(const int m, const int n,
             std::vector<sfp> &Q, const int ldQ,
             std::vector<sfp> &R, const int ldR) {
 
  // R : n x n
  // Q : m x n 
  std::fill(B.begin(), B.end(), static_cast<sfp>(0.0));
  // B = Q'*Q; 
  la.gemm('T', 'N',
    n, n, m, static_cast<sfp>(1.0), Q.data(), ldQ,
    Q.data(), ldQ, static_cast<sfp>(0.0), B.data(), ldB);

  if(false == chol(n, R, ldR)) return false;  
    
  // Q = Q/R;
  la.trsm('C', 'R', 'U', 'N', 'N', m, n, static_cast<sfp>(1.0), R.data(), ldR,
          Q.data(), ldQ);

  return true;
}
#endif
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
  
  int dim      = mat->Dim();
  int numEvals = static_cast<int>(this->L->size());
  
  
  UNUSED(R);
  UNUSED(ldsQ);
  
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
  
  sR.resize(dim*numEvals);
  XX.resize(dim*numEvals);
  x.resize(dim*numEvals);
  
  /* Call the actual solver */
  /* Change Matrix shift*/
  mat->update_matrix_shift(static_cast<fp>(0.0));
  for(auto i=0; i < numEvals; i++){
   
    /* Casting input vectors into the inner loop precision */ 
    /* cast to solver precision ONLY the vectors to be used*/  
    mat->applyScalInvMat(iDR->data()+ldiDR*i,ldiDR,mat->Dim(),1); // DR = D\R
    
    auto rb   = iDR->begin();
    auto ldrb = ldiDR;

    // scale to right hand side vector
    auto nrmR = la.nrm2(dim, iDR->data()+(0+i*ldrb),1);
    nrmR = static_cast<sfp>(1)/nrmR;
    la.scal(dim, nrmR, iDR->data()+(0+i*ldrb), 1);

    // cast to solver precision 
   
    std::transform(rb + (0+i*ldrb), rb + (0+(i+1)*ldrb), sR.data() + ldR*i,to_sfp);
  }
  // at this point right hand side vectors are 
  // up to date with the diagonal scal
  int innerIters = solve_eq();
  iters += innerIters;
  
  /* Casting output vectors into the outer loop precision */  
  std::transform(x.begin(),x.end(), XX.begin(),to_fp);

  // DR = D\R
  mat->applyScalInvMat(XX.data(),mat->Dim(),mat->Dim(),this->L->size()); 


  return XX;
}


