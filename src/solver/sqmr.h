#pragma once
#include "../matrix/matrix.h"
#include "../matrix/smatrix.h"
#include <vector>

namespace mpjd{

template<class fp, class sfp>
class SQMR {

	std::vector<sfp> Qlocked; int ldQlocked;
	std::vector<sfp> Q; int ldQ;
	
public:
	SQMR() = delete;
	SQMR(Matrix<fp> &mat_);
};

}
