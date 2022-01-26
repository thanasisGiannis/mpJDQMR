
#include <iostream>
#include <string>
#include <vector>

template<class fp, class sfp>
mpjd::JD<fp,sfp>::JD(std::vector<fp> vals_, std::vector<int> row_index_, std::vector<int> col_index_,
			int dim_, sparseDS_t DS_, std::vector<fp> &Q_, int &ldQ_, std::vector<fp> &L_,
			std::vector<fp> &R_, int &ldR_,
			fp norm_, int nEvals, eigenTarget_t eigTarget_, fp tol_,
			int maxBasis_, int maxIters_) 
			
:
//	la(),
	mat(vals_, row_index_, col_index_, dim_, DS_, norm_, la),
	dim(dim_),
	tol(tol_),
	maxIters(maxIters_),
	Qlocked(Q_), ldQlocked(ldQ_),
	Llocked(L_),
	Rlocked(R_), ldRlocked(ldR_),
	basis(std::move(std::unique_ptr<Subspace<fp>>(new Subspace<fp>(mat, dim_, nEvals,
	                    eigTarget_, maxBasis_,la,w,ldw, 
											Q,ldQ,L,R,ldR,
											Qlocked,ldQlocked,Llocked,Rlocked,ldRlocked)))),
	sqmr(std::move(std::unique_ptr<ScaledSQMR<sfp,fp>>(new ScaledSQMR<sfp,fp>(mat, Q, 
	                                           ldQ,L, R, ldR, Qlocked, ldQlocked)))),										
	eigTarget(eigTarget_)
					
{


}




