#pragma once
#include "../blasWrappers/blasWrappers.h"
#include "../matrix/matrix.h"
#include <memory>
#include <iomanip>
namespace mpjd{

template<class fp>
class SQMR {

protected:

  LinearAlgebra      la;
	
	std::vector<fp> &Q;       int &ldQ;       // Ritz vectors
	std::vector<fp> &L;                       // Ritz values
	std::vector<fp> &R;       int &ldR;       // Residual vector
	std::vector<fp> &Qlocked; int &ldQlocked; // Locked Ritz vectors
	
	Matrix<fp>      &mat; // Coefficient Matrix

public:
	SQMR() = delete;
	SQMR(Matrix<fp> &mat_, std::vector<fp> &Q_, int &ldQ_,
	      std::vector<fp> &L_, std::vector<fp> &R_, int &ldR_,
	      std::vector<fp> &Qlocked_, int &ldQlocked_);
	
	virtual std::vector<fp> solve();
};


template<class fp, class sfp>
class ScaledSQMR : public SQMR<fp> {

  /* Input low precision matrices */
	std::vector<sfp> sQ;       int ldsQ;       // Ritz vectors
	std::vector<sfp> sR;       int ldsR;       // Residual vector
	std::vector<sfp> sQlocked; int ldsQlocked; // Locked Ritz vectors
	
	ScaledMatrix<fp,sfp>   *mat;//(Matrix<fp> &mat_);	
  LinearAlgebra          &la;
	
	std::vector<fp>  XX;        // solution to sqmr- dimension : dim x numEvals
  std::vector<sfp> x;                              
  /* lower precision data area */
  std::vector<sfp> delta;
  std::vector<sfp> r;     
  std::vector<sfp> d;     
  std::vector<sfp> w;     
  std::vector<sfp> QTd; // dimension : numEvals

  
  void solve_eq();
public:
   
	/* ========= */
	ScaledSQMR() = delete;
	ScaledSQMR(Matrix<fp> &mat_, std::vector<fp> &Q_, int &ldQ_,
	      std::vector<fp> &L_, std::vector<fp> &R_, int &ldR_,
	      std::vector<fp> &Qlocked_, int &ldQlocked_, LinearAlgebra &la_);
	
	virtual std::vector<fp> solve();
};

}

#include "sqmr.tcc"

