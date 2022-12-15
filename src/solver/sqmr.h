#pragma once

#include <memory>
#include <iomanip>

#include "../blasWrappers/blasWrappers.h"
#include "../matrix/matrix.h"
#include "../orth/orthProc.h"

namespace mpjd{

template<class fp>
class SQMR {
  public:
	  SQMR() = delete;
	  SQMR(Matrix<fp> &mat_, std::shared_ptr<std::vector<fp>> Q_, int &ldQ_,
	        std::shared_ptr<std::vector<fp>> L_,
	        std::shared_ptr<std::vector<fp>> R_, int &ldR_,
	        std::shared_ptr<std::vector<fp>> Qlocked_, int &ldQlocked_);
	  virtual std::vector<fp> solve(int &iters);

  protected:
    LinearAlgebra   la;
	  std::shared_ptr<std::vector<fp>> Q; int &ldQ; // Ritz vectors
	  std::shared_ptr<std::vector<fp>> L;           // Ritz values
	  std::shared_ptr<std::vector<fp>> R;       int &ldR;       // Residual vector
	  std::shared_ptr<std::vector<fp>> Qlocked; int &ldQlocked; // Locked Ritz vectors
	  Matrix<fp>      &mat;                     // Coefficient Matrix
};


template<class fp, class sfp>
class ScaledSQMR : public SQMR<fp> {
  public:
	  ScaledSQMR() = delete;
	  ScaledSQMR(Matrix<fp> &mat_, std::shared_ptr<std::vector<fp>> Q_, int &ldQ_,
	        std::shared_ptr<std::vector<fp>> L_, 
	        std::shared_ptr<std::vector<fp>> R_, int &ldR_,
	        std::shared_ptr<std::vector<fp>> Qlocked_, int &ldQlocked_, 
	        LinearAlgebra &la_);
	  virtual std::vector<fp> solve(int &iters);

  protected:
    int solve_eq();

	  std::vector<sfp> sQ;       int ldsQ;       // Ritz vectors
	  std::vector<sfp> sR;       int ldsR;       // Residual vector
	  std::vector<sfp> sQlocked; int ldsQlocked; // Locked Ritz vectors
	  
	  ScaledMatrix<fp,sfp>   *mat;
    LinearAlgebra          &la;
	  
	  std::vector<fp>  XX;// solution to sqmr of dimension (dim x numEvals)
    std::vector<sfp> x;                              

    /* lower precision data area */
    std::vector<sfp> delta;
    std::vector<sfp> r;     
    std::vector<sfp> d;     
    std::vector<sfp> w;     
    std::vector<sfp> QTd; // of dimension (1 x numEvals)
};

template<class fp, class sfp>
class BlockScaledSQMR : public ScaledSQMR<fp,sfp> {
  public:
    BlockScaledSQMR() = delete;
    BlockScaledSQMR(Matrix<fp> &mat_, std::shared_ptr<std::vector<fp>> Q_, int &ldQ_,
	        std::shared_ptr<std::vector<fp>> L_, 
	        std::shared_ptr<std::vector<fp>> R_, int &ldR_,
	        std::shared_ptr<std::vector<fp>> Qlocked_, int &ldQlocked_, 
	        LinearAlgebra &la_);
    virtual std::vector<fp> solve(int &iters) override;
  private:
/*  
    class Householder {
      public:
        Householder(int dim, int nrhs, LinearAlgebra &la_);
        void QR(int m, int n, std::vector<sfp> &R, int ldR, 
              std::vector<sfp> &Q, int ldQ);
      
      private:
        std::vector<sfp> hhQz; int ldhhQz;
        std::vector<sfp> hhx;                    // dim x 1
        std::vector<sfp> hhv;                    // dim x 1     
        std::vector<sfp> hhu;                    //  nrhs x 1     
        std::vector<sfp> hhvhhvt; int ldhhvhhvt; // dim x dim     
        LinearAlgebra &la;
    };
    
    class Cholesky {
      public: 
        Cholesky(const int dim, const int nrhs, LinearAlgebra &la_);
        bool QR(const int m, const int n,
                      std::vector<sfp> &R, const int ldR,
                      std::vector<sfp> &Q, const int ldQ);
        
        private:
          std::vector<sfp> B; int ldB; // this vector will be used
                                       // to create upper triangular 
                                       // cholesky matrix
          
          LinearAlgebra &la;
          bool chol(const int dim, std::vector<sfp> &R, const int ldR);
    };
*/    
    int solve_eq(); 
        
    /* sparse matrix vector accumulator */
    std::vector<sfp> y; int ldy; // dim x nrhs
    
    /* sQMR step */ 
    std::vector<sfp> p0; int ldp0; // dim x nrhs
    std::vector<sfp> p1; int ldp1; // dim x nrhs
    std::vector<sfp> p2; int ldp2; // dim x nrhs

    /* lanczos step */
    std::vector<sfp> v1; int ldv1; // dim x nrhs
    std::vector<sfp> v2; int ldv2; // dim x nrhs
    std::vector<sfp> v3; int ldv3; // dim x nrhs

    std::vector<sfp> alpha; int ldalpha; // nrhs x nrhs
    std::vector<sfp> vita;  int ldvita;  // nrhs x nrhs 
    std::vector<sfp> vita2; int ldvita2; // nrhs x nrhs 

    /* sQMR matrix */
    std::vector<sfp> b0; int ldb0; // nrhs x nrhs
    std::vector<sfp> d0; int ldd0; // nrhs x nrhs

    std::vector<sfp> a1; int lda1; // nrhs x nrhs
    std::vector<sfp> b1; int ldb1; // nrhs x nrhs
    std::vector<sfp> c1; int ldc1; // nrhs x nrhs
    std::vector<sfp> d1; int ldd1; // nrhs x nrhs

    /* lanczos matrix */
    std::vector<sfp> ita;    int ldita;   // nrhs x nrhs
    std::vector<sfp> thita;  int ldthita; // nrhs x nrhs 
    std::vector<sfp> zita;   int ldzita;  // nrhs x nrhs 
    std::vector<sfp> zita_;  int ldzita_; // nrhs x nrhs 

    /* updating solution sQMR*/
    std::vector<sfp> tau_;  int ldtau_; // nrhs x nrhs 
    std::vector<sfp> tau_2;             // nrhs x nrhs
    std::vector<sfp> tau;   int ldtau;  // nrhs x nrhs 

    /* updating residual sR*/
    std::vector<sfp> Ap0;  int ldAp0;   // dim x nrhs 
    std::vector<sfp> Ap1;  int ldAp1;   // dim x nrhs 

    /* supportive data */
    int ldx;
    LinearAlgebra &la;
    
    // householder orthogonalization provider
    Householder<sfp> hh;
    /* helping vector for Householder QR */
    std::vector<sfp> hhR;  int ldhhR;        // 2nrhs x nrhs 
    std::vector<sfp> hhQ;  int ldhhQ;        // 2nrhs x nrhs 
       
    // Cholesky orthogonalization provider
    MGS<sfp> mgs;
};
}




#include "sqmr.tcc"
#include "blockSqmr.tcc"


