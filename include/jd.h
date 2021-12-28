#pragma once

#include <vector>
#include "matrix.h"


template <class fp>
void printVec	(std::vector<fp> vec, std::string s){
	std::cout << s << std::endl;
	for(auto v = vec.begin(); v!=vec.end(); v++){
		std::cout << *v << std::endl;
	}
}

namespace mpJDQMR{

template<class fp,class sfp>
class JD{
  // Uppercasted letters represent matrices
  // Lowercasted letters represent vectors
  
	std::vector<fp> *T{}; int ldT;// projected matrix    : maxBasis x maxBasis
	std::vector<fp> *V{}; int ldV;// rayleigh ritz basis : dim x maxBasis
	std::vector<fp> *Q{}; int ldQ;// ritz vectors        : dim x numEvals
	std::vector<fp> *L{};  				// ritz values         : dim x numEvals
	std::vector<fp> *R{}; int ldR;// residual vectors    : dim x numEvals

//	std::vector<fp> *AV{}; int ldAV;// 

	Matrix<fp> *Mat{};      // matrix object
	int  			  numEvals{}; // number of eigenpairs required
	const int   dim{};      // matrix dimension
	const int   maxBasis{}; // maximum subspace dimension
	const int   resSize{};  // basis restart size
	
	
	void orth(std::vector<fp> *V,int ldV, int rows, int cols, std::vector<fp> v);
	void orth(std::vector<fp> *V,int ldV, int rows, int cols) const ;
	void matRand(std::vector<fp> *V,int rows, int cols);
	
	void intitT(); // update projected matrix using projection matrix V
public:
	JD() = delete;
	JD(Matrix<fp> *_Mat,int const _dim, int const _numEvals, int const _maxBasis,int const resSize);
	~JD();
	void eigenSolve();
	
};
}

// include member functions
#include "../src/jd/jd.tcc"

