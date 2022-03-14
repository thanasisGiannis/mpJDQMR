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

  /* Input matrices */
	std::vector<sfp> sQ;       int ldsQ;       // Ritz vectors
	//std::vector<sfp> sL;                       // Ritz values
	std::vector<sfp> sR;       int ldsR;       // Residual vector
	std::vector<sfp> sQlocked; int ldsQlocked; // Locked Ritz vectors
	
	
	std::vector<fp>  XX;        // solution to sqmr
  std::vector<sfp> x;                    

  /* Inner vectors */
  std::vector<sfp> v1; // Lanczos vector
  std::vector<sfp> v2; // Lanczos vector
   
  std::vector<sfp> p0; // sQMR vector
  std::vector<sfp> p1; // sQMR vector
  
  std::vector<sfp> w;  // new direction to sqmr iter 
 	
	ScaledMatrix<fp,sfp>   *mat;//(Matrix<fp> &mat_);	
  LinearAlgebra          &la;
  
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

