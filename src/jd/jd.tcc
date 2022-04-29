#include <iostream>
#include <string>
#include <vector>


template<class fp, class sfp>
mpjd::JD<fp,sfp>::JD(std::vector<fp> vals_, std::vector<int> row_index_, std::vector<int> col_index_,
			int dim_, sparseDS_t DS_, std::vector<fp> &Q_, int &ldQ_, std::vector<fp> &L_, 
			std::vector<fp> &R_, int &ldR_, fp norm_, struct mpjdParam &params):
	parameters(params),
	mat(vals_, row_index_, col_index_, dim_, DS_, norm_, la),
	dim(params.dim),
	tol(params.tol),
	maxIters(params.maxIters),
	Qlocked(Q_), ldQlocked(ldQ_),
	Llocked(L_),
	Rlocked(R_), ldRlocked(ldR_),
	basis(std::move(std::unique_ptr<Subspace<fp>>(new Subspace<fp>(mat, dim_, params.numEvals,
	                    params.eigTarget, params.maxBasis,la,w,ldw, 
											Q,ldQ,L,R,ldR,
											Qlocked,ldQlocked,Llocked,Rlocked,ldRlocked)))),
	sqmr(std::move(std::unique_ptr<ScaledSQMR<fp,sfp>>(new ScaledSQMR<fp,sfp>(mat, Q, 
	                                           ldQ,L, R, ldR,
	                                           Qlocked,ldQlocked,la)))),										
	eigTarget(params.eigTarget)
					
{


}

