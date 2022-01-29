#pragma once
#include "../matrix/matrix.h"
#include <memory>
namespace mpjd{

template<class fp>
class SQMR {

protected:
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

	std::vector<sfp> sQ;       int sldQ;       // Ritz vectors
	std::vector<sfp> sL;                       // Ritz values
	std::vector<sfp> sR;       int sldR;       // Residual vector
	std::vector<sfp> sQlocked; int sldQlocked; // Locked Ritz vectors
	
	//std::shared_ptr<Matrix<fp>>   mat; // Coefficient Matrix
  ScaledMatrix<fp,sfp>   *mat;//(Matrix<fp> &mat_);	

public:
	/* ========= */
	ScaledSQMR() = delete;
	ScaledSQMR(Matrix<fp> &mat_, std::vector<fp> &Q_, int &ldQ_,
	      std::vector<fp> &L_, std::vector<fp> &R_, int &ldR_,
	      std::vector<fp> &Qlocked_, int &ldQlocked_);
	
	virtual std::vector<fp> solve();
};

}

#include "sqmr.tcc"

