#include <cmath>
#include <omp.h>

#define UNUSED(expr) static_cast<void>(expr)

template<class fp>
mpjd::Cholesky<fp>::Cholesky(const int dim, const int nrhs, LinearAlgebra &la_)
: la(la_) {

  B.reserve(nrhs*nrhs);
  B.resize(nrhs*nrhs);
  std::fill(B.begin(), B.end(), static_cast<fp>(0.0));
  ldB = nrhs;
  UNUSED(dim);
}


template<class fp>
bool mpjd::Cholesky<fp>::chol(const int n, std::vector<fp> &L, int ldL) {
  // L = chol(B);
  
  // L = zeros(n);
  std::fill(L.begin(), L.end(), static_cast<fp>(0.0));
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
  
  fp sum;
  for(int j=0; j<n; j++){
    sum = static_cast<fp>(0.0);
    for(int k=0; k<j; k++){
      sum = sum+ L[k+j*ldL]*L[k+j*ldL];
    }
    L[j+j*ldL] = std::sqrt(B[j+j*ldB]-sum);
    if(std::isnan(L[j+j*ldL])) return false;
    
    for(int i=j+1; i<n; i++){
      sum = static_cast<fp>(0.0);
      for(int k=0; k<j; k++){
        sum = sum + L[k+i*ldL]*L[k+j*ldL];
      }
      L[j+i*ldL] = (static_cast<fp>(1.0)/L[j+j*ldL])*(B[j+i*ldB]-sum);
      if(std::isnan(L[j+i*ldL])) return false;
    }
  }
  return true;
}

template<class fp>
bool mpjd::Cholesky<fp>::QR(const int m, const int n,
             std::vector<fp> &Q, const int ldQ,
             std::vector<fp> &R, const int ldR) {
 
  // R : n x n
  // Q : m x n 
  std::fill(B.begin(), B.end(), static_cast<fp>(0.0));
  // B = Q'*Q; 
  la.gemm('T', 'N',
    n, n, m, static_cast<fp>(1.0), Q.data(), ldQ,
    Q.data(), ldQ, static_cast<fp>(0.0), B.data(), ldB);

  if(false == chol(n, R, ldR)) return false;  
    
  // Q = Q/R;
  la.trsm('C', 'R', 'U', 'N', 'N', m, n, static_cast<fp>(1.0), R.data(), ldR,
          Q.data(), ldQ);

  return true;
}




template<class fp>
mpjd::Householder<fp>::Householder(int dim, int nrhs, LinearAlgebra &la_)
  : la(la_)
{
    
    hhQz.reserve(dim*dim); // dim x dim
    hhQz.insert(hhQz.begin(), hhQz.capacity(),static_cast<fp>(0.0));
    ldhhQz = dim;

    hhx.reserve(dim*1); // dim x 1
    hhx.insert(hhx.begin(), hhx.capacity(),static_cast<fp>(0.0));

    hhv.reserve(dim*1); // dim x 1
    hhv.insert(hhv.begin(), hhv.capacity(),static_cast<fp>(0.0));

    hhu.reserve(nrhs*1);   // dim x 1
    hhu.insert(hhu.begin(), hhu.capacity(),static_cast<fp>(0.0));

    hhvhhvt.reserve(dim*dim);   // dim x dim
    hhvhhvt.insert(hhvhhvt.begin(), hhvhhvt.capacity(),static_cast<fp>(0.0));
    ldhhvhhvt = dim;

}


template<class fp>
void mpjd::Householder<fp>::QR(int m, int n, std::vector<fp> &Q, int ldQ, 
              std::vector<fp> &R, int ldR) {

  
  if(m<n) exit(-1); // tall and thin only
  
  // Q = I
  std::fill(Q.begin(), Q.end(), static_cast<fp>(0.0));
  
  for(int i=0; i<m; i++){
    Q[i+i*ldQ] = static_cast<fp>(1.0);
  }
  //std::vector<fp> Qprev;
  
  for(int k=0; k<n; k++) {
    // x = zeros(m,1); 
    std::fill(hhx.begin(), hhx.end(), static_cast<fp>(0.0));
    
    // x(k:m,1)=R(k:m,k);
    memcpy(hhx.data()+k, R.data() + k+k*ldR, sizeof(fp)*(m-k)); 

    //g=norm(x);
    fp g = la.nrm2(m, hhx.data(), 1);
    
    // v=x; v(k)=x(k)+g;
    hhv = hhx;
    hhv[k] = hhx[k] + g;
  
    // s=norm(v);
    fp s = la.nrm2(m,hhv.data(),1);  
    
    // if s!=0
    // v = v/s
    la.scal(m, static_cast<fp>(1.0)/s,hhv.data(),1);
    
    // u=R'*v;
    la.gemm('T', 'N',
      n, 1, m, static_cast<fp>(1.0), R.data(), ldR,
      hhv.data(), m, static_cast<fp>(0.0), hhu.data(), n);


    // R=R-2*v*u'; 
    la.gemm('N', 'T',
      m, n, 1, static_cast<fp>(-2.0), hhv.data(), m,
      hhu.data(), n, static_cast<fp>(1.0), R.data(), ldR);


    // Q=Q-2*Q*(v*v');
    hhQz = Q;
    //hhvhhvt = v*v';
    la.gemm('N', 'T',
      m, m, 1, static_cast<fp>(1.0), hhv.data(), m,
      hhv.data(), m, static_cast<fp>(0.0), hhvhhvt.data(), ldhhvhhvt);      

 
    // Q = Q-2*Q*hhvhhvt
    la.gemm('N', 'N', m, m, m, 
    static_cast<fp>(-2.0), hhQz.data(), ldhhQz, hhvhhvt.data(), ldhhvhhvt,
    static_cast<fp>(1.0), Q.data(), ldQ);
    
  }
  
}



template<class fp>
mpjd::MGS<fp>::MGS(const int dim, const int nrhs, LinearAlgebra &la_)
: la(la_) {
  UNUSED(nrhs);
  UNUSED(dim);
}

template<class fp>
bool mpjd::MGS<fp>::QR(int m, int n, std::vector<fp> &Q, int ldQ, 
              std::vector<fp> &R, int ldR) {
  /*
  [~,n] = size(Q);
  R = zeros(n);
  for j=1:n
      r = norm(Q(:,j));
      R(j,j) = r;
      Q(:,j) = Q(:,j)/r;
      for k=j+1:n
         r = Q(:,j)'*Q(:,k);
         R(j,k) = r;
         Q(:,k) = Q(:,k) - r*Q(:,j); 
      end
  end

  */
  std::fill(R.begin(), R.end(), static_cast<fp>(0.0));              

  for(int j=0; j<n; j++) {
    fp r = la.nrm2(m,Q.data() + 0 +j*ldQ,1);
    R[j+j*ldR] = r;
    la.scal(m,static_cast<fp>(1.0)/r,Q.data() + 0 +j*ldQ,1);
    for(auto k=j+1; k<n; k++) { 
      r = la.dot(m,Q.data() + 0 +j*ldQ,1,Q.data() + 0 + k*ldQ,1);
      R[j+k*ldR] = r;
      la.axpy(m, -r, Q.data() + 0 +j*ldQ, 1, Q.data() + 0 + k*ldQ, 1);
    }
  } 
  return true;
}



template<class fp>
bool mpjd::MGS<fp>::orth(int m, int n, std::vector<fp> &Q, int ldQ) {
  /*
  [~,n] = size(Q);
  R = zeros(n);
  for j=1:n
      r = norm(Q(:,j));
      Q(:,j) = Q(:,j)/r;
      for k=j+1:n
         r = Q(:,j)'*Q(:,k);
         Q(:,k) = Q(:,k) - r*Q(:,j); 
      end
  end

  */
  for(int j=0; j<n; j++) {
    fp r = la.nrm2(m,Q.data() + 0 +j*ldQ,1);
    la.scal(m,static_cast<fp>(1.0)/r,Q.data() + 0 +j*ldQ,1);
    for(auto k=j+1; k<n; k++) { 
      r = la.dot(m,Q.data() + 0 +j*ldQ,1,Q.data() + 0 + k*ldQ,1);
      la.axpy(m, -r, Q.data() + 0 +j*ldQ, 1, Q.data() + 0 + k*ldQ, 1);
    }
  } 
  return true;
}



template<class fp>
bool mpjd::MGS<fp>::orthAgainst(const int m, 
                     const int nQ, std::vector<fp> &Q, const int ldQ,
                     const int nW, std::vector<fp> &W, const int ldW) {
                     
  /*
    w = orth(V,w);
  */
  
  #pragma omp parallel for 
  for(int j=0; j<nW; j++) {
    for(int i=0; i<nQ; i++) {
      // alpha = w(:,j)'*Q(:,i);
      fp alpha = la.dot(m, W.data()+0+j*ldW, 1, Q.data()+0+i*ldQ, 1);
      // w(:,j) = w(:,j) - Q(:,i)*alpha;
      la.axpy(m, -alpha, Q.data()+0+i*ldQ, 1, W.data()+0+j*ldW, 1);
    }
  }
  return true;
}

















