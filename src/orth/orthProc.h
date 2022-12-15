#pragma once 

namespace mpjd{
template<class fp>
class Householder {
  public:
    Householder(int dim, int nrhs, LinearAlgebra &la_);
    void QR(int m, int n, std::vector<fp> &Q, int ldQ, 
          std::vector<fp> &R, int ldR);
  
  private:
    std::vector<fp> hhQz; int ldhhQz;
    std::vector<fp> hhx;                    // dim x 1
    std::vector<fp> hhv;                    // dim x 1     
    std::vector<fp> hhu;                    //  nrhs x 1     
    std::vector<fp> hhvhhvt; int ldhhvhhvt; // dim x dim     
    LinearAlgebra &la;
};


template<class fp>
class Cholesky {
  public: 
    Cholesky(const int dim, const int nrhs, LinearAlgebra &la_);
    bool QR(const int m, const int n,
                  std::vector<fp> &R, const int ldR,
                  std::vector<fp> &Q, const int ldQ);
    
    private:
      std::vector<fp> B; int ldB; // this vector will be used
                                   // to create upper triangular 
                                   // cholesky matrix
      
      LinearAlgebra &la;
      bool chol(const int dim, std::vector<fp> &R, const int ldR);
};


template<class fp>
class MGS {
  public: 
    MGS(const int dim, const int nrhs, LinearAlgebra &la_);
    bool QR(const int m, const int n,
                  std::vector<fp> &Q, const int ldQ,
                  std::vector<fp> &R, const int ldR);
    
    private:      
      LinearAlgebra &la;
};



}


#include "orthProc.tcc"
