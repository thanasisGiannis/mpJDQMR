
#include <iostream>
#include <string>
#include <vector>

template<class fp, class sfp>
mpjd::JD<fp,sfp>::JD(std::vector<fp> vals_, std::vector<int> row_index_, std::vector<int> col_index_,
			int dim_, sparseDS_t DS_,int nEvals, eigenTarget_t eigTarget_,int maxBasis_, target_t target_) 
			
:
	la(target_),
	mat(vals_, row_index_, col_index_, dim_, DS_, la),
	numEvals(nEvals),
	basis(Subspace<fp>(mat, dim_, nEvals,eigTarget_, maxBasis_,la,w,ldw, 
											Q,ldQ,L,R,ldR,
											Qlocked,ldQlocked,Llocked,Rlocked,ldRlocked)),
	eigTarget(eigTarget_)
					
{

	std::cout<< "Q " <<  Q.capacity() << " "<< ldQ << std::endl;
	std::cout<< "L " <<  L.capacity() << std::endl;

	std::cout<< "R " <<  R.capacity() << " "<< ldR << std::endl;

	std::cout<< "Qlocked " <<  Qlocked.capacity() << " "<< ldQlocked << std::endl;

	std::cout<< "Llocked " <<  Llocked.capacity() << std::endl;
	std::cout<< "Rlocked " <<  Rlocked.capacity() << " "<< ldRlocked << std::endl;


}




