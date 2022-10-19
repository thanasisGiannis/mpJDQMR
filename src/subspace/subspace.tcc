//TODO: Format file

#include <iostream>

#include <random>
#include <algorithm>
#include <vector>
#include <iterator>

#include "../blasWrappers/blasWrappers.h"

template<class fp>
void init_vec_zeros(std::vector<fp> &v, const int nVals){

	for (auto j=0;j<nVals; j++){
		v.push_back(static_cast<fp>(0));
	}
} 

template<class fp>
void init_vec_zeros(std::shared_ptr<std::vector<fp>> v, const int nVals){

	for (auto j=0;j<nVals; j++){
		v->push_back(static_cast<fp>(0));
	}
} 

template<class fp>
mpjd::Subspace<fp>::Subspace(Matrix<fp> &mat_, const int dim_, 
    const int numEvals_, const  eigenTarget_t eigTarget_, const int maxBasis_, 
    LinearAlgebra &la_,
    std::shared_ptr<std::vector<fp>> w_, int &ldw_,	
    std::shared_ptr<std::vector<fp>> Q_, int &ldQ_, 
    std::shared_ptr<std::vector<fp>> L_, 
    std::shared_ptr<std::vector<fp>> R_, int &ldR_, 
    std::shared_ptr<std::vector<fp>> Qlocked_, int &ldQlocked_, 
    std::shared_ptr<std::vector<fp>> Llocked_,
    std::shared_ptr<std::vector<fp>> Rlocked_, int &ldRlocked_)
    : la(la_),
	    mat(mat_),
	    dim(dim_),
	    numEvals(numEvals_),
	    eigTarget(eigTarget_),
	    maxBasis(maxBasis_),
	    blockSize(numEvals_),
	    basisSize(0),
	    lockedNumEvals(0),
	    w{w_}, ldw(ldw_),
	    Q{Q_}, ldQ(ldQ_),
	    L{L_},
	    R{R_}, ldR(ldR_),
	    Qlocked{Qlocked_}, ldQlocked(ldQlocked_),
	    Llocked{Llocked_},
	    Rlocked{Rlocked_}, ldRlocked(ldRlocked_) {

  // subspace basis
	V.reserve(dim*maxBasis*numEvals); ldV = dim; 

  // projected matrix
	T.reserve(maxBasis*numEvals*maxBasis*numEvals); ldT = maxBasis*numEvals;
	init_vec_zeros(T,maxBasis*numEvals*maxBasis*numEvals);
	
	// eigenvectors of T
	q.reserve(maxBasis*numEvals*numEvals); ldq = maxBasis*numEvals; 
	init_vec_zeros(q,maxBasis*numEvals*numEvals);
		
	// buffer for syev
	qq.reserve(maxBasis*numEvals*maxBasis*numEvals); ldqq = maxBasis*numEvals; 
	init_vec_zeros(qq,maxBasis*numEvals*maxBasis*numEvals);

  // buffer for syev
	LL.reserve(maxBasis*numEvals); 
	init_vec_zeros(LL,maxBasis*numEvals);

  // locked eigenvectors
	Q->reserve(dim*numEvals); ldQ = dim; 
	init_vec_zeros(Q,dim*numEvals);

  // locked eigenvectors
  Qprev.reserve(dim*numEvals); ldQprev = dim; 
	init_vec_zeros(Qprev,dim*numEvals);

  // locked eigenvalues
	L->reserve(numEvals); 				        
	init_vec_zeros(L,numEvals);

  // eigen residual 
  R->reserve(dim*numEvals); ldR = dim; 
	init_vec_zeros(R,dim*numEvals);

	// locked eigenvectors
	Qlocked->reserve(dim*numEvals); ldQlocked = dim; 
	//Qlocked->clear();
  init_vec_zeros(Qlocked,dim*numEvals);
  
  // locked eigenvalues
	Llocked->reserve(numEvals); 								      
	//Llocked->clear();
  init_vec_zeros(Llocked,numEvals);

  // locked eigen residual 
	Rlocked->reserve(dim*numEvals); ldRlocked = dim; 
	//Rlocked->clear();
	init_vec_zeros(Rlocked,dim*numEvals);
	
	// locked eigenvectors
  QTw.reserve(numEvals*numEvals); ldQTw = numEvals ; 
	init_vec_zeros(QTw,numEvals*numEvals);
	
	// tmp vector to used in Av
	Aw.reserve(dim*blockSize); ldAw = dim;  
	init_vec_zeros<fp>(Aw,dim*blockSize);

	AV.reserve(dim*blockSize*maxBasis); ldAV = dim;

  // new subspace direction
	w->reserve(dim*blockSize);  ldw  = dim;  
}

template<class fp>
void mpjd::Subspace<fp>::Subspace_init_direction(){

	/*
		Initialize basis w with random vals
	*/
	std::random_device rand_dev;
	std::mt19937 			 generator(rand_dev());
	std::uniform_int_distribution<int> distr(-100,100);

	for(auto i=0;i<dim*numEvals;i++){
		w->push_back(static_cast<fp>(distr(generator))/static_cast<fp>(100));
	}
}


template<class fp>
void mpjd::Subspace<fp>::Subspace_orth_direction(){

	auto VV = static_cast<fp*>(V.data());
	auto vv = static_cast<fp*>(w->data());

	int rows = dim;
	int cols = basisSize*blockSize;
  
  VV = static_cast<fp*>(V.data());
  /* w = orth(V,w) */
	for(int j=0; j < blockSize; j++){
		for(int i=0; i < cols ; i++){
		    // alpha = V(i)'v
				fp alpha {-la.dot(rows,&VV[0+i*ldV],1,&vv[0+j*ldV],1)};
				// v = v - V(i)*alpha
				la.axpy(rows,alpha,&VV[0+i*ldV],1,&vv[0+j*ldV],1); 
		}	
	}

  /* w = orth(w) */
  for(auto k=0;k<3;k++){
	  for(int j=0; j < blockSize; j++){
		  for(int i=0; i < j ; i++){
		      // alpha = V(i)'v
				  auto alpha {-la.dot(rows,&vv[0+i*ldV],1,&vv[0+j*ldV],1)};
				  // v = v - V(i)*alpha
				  la.axpy(rows,alpha,&vv[0+i*ldV],1,&vv[0+j*ldV],1);     	
		  }	
		  // v = v/norm(v)
		  auto alpha {la.nrm2(rows,&vv[0+j*ldV],1)}	;
		  la.scal(dim,static_cast<fp>(1.0/alpha),&vv[0+j*ldV],1);    
	  }
	}  
	updateOrthogonalizations(1);
}

template<class fp>
void mpjd::Subspace<fp>::Subspace_orth_basis(){

	auto VV = static_cast<fp*>(V.data());

	int rows = dim;
	int cols = basisSize*numEvals;

  /* V = orth(V) */
  for(int k=0; k<3; k++){
	  for(int j=0; j < cols; j++){
		  for(int i=0; i < j ; i++){
		      // alpha = V(i)'v
				  auto alpha {-la.dot(rows,&VV[0+i*ldV],1,&VV[0+j*ldV],1)};
				  // v = v - V(i)*alpha
				  la.axpy(rows,alpha,&VV[0+i*ldV],1,&VV[0+j*ldV],1); 
		  }	
		  // v = v/norm(v)
		  auto alpha {la.nrm2(rows,&VV[0+j*ldV],1)}	;
		  la.scal(dim,static_cast<fp>(1.0/alpha),&VV[0+j*ldV],1);    
	  }
	}
	updateOrthogonalizations(1);
}


template<class fp>
void mpjd::Subspace<fp>::Subspace_project_at_new_direction(){

  fp one       = static_cast<fp>( 1.0);
  fp minus_one = static_cast<fp>(-1.0);
  fp zero      = static_cast<fp>( 0.0);
  
	mat.matVec(*w,ldw, Aw, ldAw, blockSize);     // Aw  = A*w
	
	AV.insert(AV.end(), Aw.begin() , Aw.end()); // [AV = [AV Aw]

	fp *Aw_ = Aw.data();
	fp *w_  =  w->data();
	fp *T_  =  T.data();
	fp *V_  =  V.data();
		

	//	T = [	T 0; 0 w'Aw]
	la.gemm('T','N', blockSize, blockSize, dim, one, w_, ldw,
	    Aw_, ldAw, zero, &T_[basisSize*blockSize + basisSize*blockSize*ldT],ldT);	
						
	/* only the diagonal part is of interest */					
	if(basisSize < 1) return;

	//	T = [	T V'Aw; 0 w'Aw]
	la.gemm('T','N', basisSize*blockSize, blockSize, dim, one, V_, ldV,
	    Aw_, ldAw, zero, &T_[0 + basisSize*blockSize*ldT],ldT);	
	
	//	T = [	T V'Aw; w'AV w'Aw]
	la.gemm('T','N', blockSize, basisSize*blockSize, dim, one, Aw_, ldAw,
	    V_, ldV, zero, &T_[basisSize*blockSize + 0*ldT],ldT);	
}	

template<class fp> 
void mpjd::Subspace<fp>::Subspace_update_basis(){

	// V = [V w]
	V.insert(V.end(), w->begin() , w->end());
	basisSize = basisSize + 1;
}

template<class fp>
void mpjd::Subspace<fp>::Subspace_projected_mat_eig(){

	qq = T; // copy data values from T to qq 
					// eig replaces QQ with eigenvectors of T
	la.eig((basisSize)*blockSize, qq.data() , ldqq, LL.data(), 
	    numEvals, eigTarget);

	/* 
	  eig returns ALL eigenpairs of T
    extract the wanted ones
    LL contains eigenvalues in descending order
  */	   
	L->clear(); // Clear previous
	q.clear(); // 

	auto LLstart_ = LL.begin();
	auto LLend_   = LL.begin();
	auto qqstart_ = qq.begin();
	auto qqend_   = qq.begin();

	switch(eigTarget){
		case eigenTarget_t::SM:
			LLstart_ += 0;
			LLend_   += blockSize;
			
			qqstart_ += (0*ldq);
			qqend_   += (blockSize*ldq);
			break;
		case eigenTarget_t::LM:
			LLstart_ += ((basisSize-1)*blockSize);
			LLend_   += ((basisSize)*blockSize);
			
			qqstart_ += (((basisSize-1)*blockSize)*ldq);
			qqend_   += (((basisSize)*blockSize)*ldq);
			break;
		default:
				exit(-1);
	}


	L->insert(L->end(), LLstart_ , LLend_);
	q.insert(q.end(), qqstart_, qqend_);

	/* Q = V*qq */
	fp one  = static_cast<fp>(1.0);
	fp zero = static_cast<fp>(0.0);
	
	Qprev = *Q;

	la.gemm('N', 'N',dim, blockSize, basisSize*blockSize,
	    one, V.data() , ldV, q.data() , ldq, zero, Q->data() , ldQ);
}

template<class fp>
void mpjd::Subspace<fp>::Subspace_eig_residual(){

	// R = AV*q-Q*L
	fp *AV_ = AV.data();
	fp *q_  =  q.data();
	fp *Q_  =  Q->data();
	fp *L_  =  L->data();
	fp *R_  =  R->data();
	
	fp one        = static_cast<fp>(1.0);
	fp zero       = static_cast<fp>(0.0);
	fp minus_one  = static_cast<fp>(-1.0);
	
	*R = *Q;
	
	// R = R*L = Q*L 
	for(auto j=0; j<numEvals; j++){
		la.scal(dim, L_[j], &R_[0+j*ldR], 1);
	}
	
	// R = AV*q-R = AV*q-Q*L
	la.gemm('N', 'N',dim, numEvals, basisSize*blockSize,
	    one, AV_, ldAV, q_, ldq, minus_one, R_, ldR);
}

template<class fp>
bool mpjd::Subspace<fp>::Check_Convergence(const fp tol){

	fp *R_ = R->data();
	fp  rho{};
  conv_num.clear();
  
	for(auto j=0;j<numEvals;j++){
		rho = la.nrm2(dim,&R_[0+j*ldR],1);
		if(rho < tol*mat.Norm()){
			conv_num.push_back(j);
		}
	}

#if 1	// TODO: Locking procedure - the #else code block
	if(conv_num.size() >= numEvals){
      *Rlocked = *R;
      *Llocked = *L;
      *Qlocked = *Q;
      
      lockedNumEvals = numEvals; 
	    numEvals       = 0;	
	    blockSize      = 0;
      return true;
	}else{
      return false;
	}
#else


#endif
};


template<class fp>
void mpjd::Subspace<fp>::Subspace_Check_size_and_restart(){

  /* check if maxBasisSize is reached and if true restart  */
  if(basisSize == maxBasis-1){
    Subspace_restart();
  }
}


template<class fp>
void mpjd::Subspace<fp>::Subspace_restart(){

  /* 
    restart basis as:
    V = orth([Qprev Q R])
  */
  
  /* V = orth([Qprev Q R]) */
  V.clear(); 
  V.insert(V.end(), Qprev.begin(), Qprev.end());
  V.insert(V.end(), Q->begin(), Q->end());
  V.insert(V.end(), R->begin(), R->end());

  basisSize = 3;
  
  Subspace_orth_basis();
  
  
  /* T = zeros() */
  std::fill(T.begin(), T.end(), static_cast<fp>(0.0));  

  /* AV = A*V */
 	std::fill(AV.begin(), AV.end(), static_cast<fp>(0.0));
 	mat.matVec(V,ldV, AV, ldAV, basisSize*numEvals);     // AV  = A*V

  /* T = V'AV */
  fp one  = 1.0;
  fp zero = 0.0;
  la.gemm('T', 'N', basisSize*numEvals, basisSize*numEvals, dim,
						one, V.data(), ldV, AV.data(), ldAV, zero, T.data(), ldT);
						
  updateRestarts(1);
}

template<class fp>
int mpjd::Subspace<fp>::getBasisSize(){return basisSize;}

template<class fp>
int mpjd::Subspace<fp>::getMaxBasisSize(){return maxBasis;}













