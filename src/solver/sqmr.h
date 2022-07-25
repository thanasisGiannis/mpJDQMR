//TODO: replace references to smart pointers
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

  private:
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

}

#include "sqmr.tcc"

