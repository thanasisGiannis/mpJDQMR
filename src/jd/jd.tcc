#include <iostream>
#include <string>
#include <vector>


template<class fp, class sfp>
mpjd::JD<fp,sfp>::JD(std::vector<fp> vals_, 
  std::vector<int> row_index_, std::vector<int> col_index_,
  int dim_, sparseDS_t DS_, 
  fp norm_, struct mpjdParam &params)
  : eigTarget(params.eigTarget),
    parameters(params),
    maxIters(params.maxIters),
    dim(params.dim),
    tol(params.tol),
    Qlocked{std::make_shared<std::vector<fp>>()},
    Llocked{std::make_shared<std::vector<fp>>()},
    Rlocked{std::make_shared<std::vector<fp>>()}, //ldRlocked(ldR_),
    mat(vals_, row_index_, col_index_, dim_, DS_, norm_, la),
    basis(std::move(std::unique_ptr<Subspace<fp>>(new Subspace<fp>(mat, dim_, 
      params.numEvals, params.eigTarget, params.maxBasis,la,
      w = std::shared_ptr<std::vector<fp>>(new std::vector<fp>),ldw, 
      Q = std::shared_ptr<std::vector<fp>>(new std::vector<fp>),ldQ,
      L = std::shared_ptr<std::vector<fp>>(new std::vector<fp>),
      R = std::shared_ptr<std::vector<fp>>(new std::vector<fp>),ldR,
      Qlocked,ldQlocked,Llocked,Rlocked,ldRlocked)))),
    sqmr(std::move(std::unique_ptr<BlockScaledSQMR<fp,sfp>>(new BlockScaledSQMR<fp,sfp>(
      mat, Q , ldQ,L, /*R, ldR,*/ Qlocked, ldQlocked, la))))										
     {}

template<class fp, class sfp>
void mpjd::JD<fp,sfp>::printIterationStats(const int loopIter) {


  int numEvals = static_cast<int>(R->size()/dim);  
  for(auto j=0; j<numEvals; j++){
    fp *R_ = R->data();
    fp rho = la.nrm2(dim,&R_[0+j*ldR],1);
    std::cout << std::setprecision(3) 
      <<  std::scientific
      << "\%||R(" << loopIter << "," << j << ")||=" << rho/mat.Norm() 
      << std::endl;
  }
  std::cout << "\% ======================" << std::endl;
}


