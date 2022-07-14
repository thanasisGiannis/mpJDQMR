#pragma once

#include <memory>

#include "../blasWrappers/blasWrappers.h"
#include "../matrix/matrix.h"
#include "../subspace/subspace.h"
#include "../solver/sqmr.h"
#include "../include/mpjd_stats.h"

namespace mpjd{

struct mpjdParam {
  double tol = 1e-08;

  eigenTarget_t eigTarget = eigenTarget_t::SM;

  int numEvals = 1;
  int maxBasis = 15;
  int maxIters = 1000;
  int dim      = -1;

  bool printStats     = false;
  bool printIterStats = false;
};

template<class fp, class sfp>
class JD : public basisStatistics, public jdStatistics {

  public:
	  JD() = delete;
	  JD(std::vector<fp> vals_, std::vector<int> row_index_, std::vector<int> col_index_,
			  int dim_, sparseDS_t DS_, std::vector<fp> &Q_, int &ldQ_, std::vector<fp> &L_, 
			  std::vector<fp> &R_, int &ldR_, fp norm_, struct mpjdParam &params); 

	  void solve();									 

  private:	
	  bool Check_Convergence();
	  void printIterationStats(const int loopIter);

    eigenTarget_t  eigTarget; // Smallest by absolute value 
	  struct mpjdParam &parameters;
	  
	  int 					 maxIters;
	  int						 dim;
	  fp						 tol;
	  
	  /* Shared variables between classes */
	  std::shared_ptr<std::vector<fp>> w; int ldw;  // common with basis and sqmr
															                    // next subspace direction
														                      // reserved at subspace class
														                      // will need to be used 
														                      // by inner solver
														    
	  std::vector<fp> Q;			 int ldQ; 			// common with basis and sqmr
	  std::vector<fp> L;											// common with basis and sqmr
	  std::vector<fp> R;			 int ldR;				// common with basis and sqmr

	  std::vector<fp> &Qlocked; int ldQlocked; // common with basis and sqmr
	  std::vector<fp> &Llocked; 								// common with basis and sqmr
	  std::vector<fp> &Rlocked; int ldRlocked; // locked eigen residual 

	  /* CAUTION THIS SHOULD BE AFTER VARIABLES */
	  LinearAlgebra      la;
	  Matrix<fp>         mat;
	  std::unique_ptr<Subspace<fp>>   basis; // implement after basis implementation
	  std::unique_ptr<SQMR<fp>>       sqmr;  // implement after s implementation
};
}

#include "jd.tcc"
#include "eigen_solver.tcc"
