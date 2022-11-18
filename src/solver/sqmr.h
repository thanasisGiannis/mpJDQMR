#pragma once

#include <memory>
#include <iomanip>

#include "../blasWrappers/blasWrappers.h"
#include "../matrix/matrix.h"

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
    int solve_eq(); 
    void orth_v3_update_vita(); 
    
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
    std::vector<sfp> tau;   int ldtau;  // nrhs x nrhs 

    /* helping vector for Givens Rotations */
    std::vector<sfp> z;  int ldz; // 2nrhs x nrhs 
    
    /* supportive data */
    int ldx;
    LinearAlgebra &la;
};
}




#include "sqmr.tcc"
#include "blockSqmr.tcc"

