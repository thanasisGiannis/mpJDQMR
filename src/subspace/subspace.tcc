#include <iostream>
#include <random>
#include <algorithm>
#include <vector>
#include <iterator>
#include "../blasWrappers/blasWrappers.h"

//#include "../include/helper.h"

template<class fp> 
void printMat(std::vector<fp> A,int ldA, int rows,int cols,std::string name){
	fp *A_ = A.data();
//	std::cout <<"%"<< name << std::endl;
	
//	std::cout << name << "[]" << std::endl;
	for(auto i=0;i<rows;i++){	
		for(auto j=0;j<cols; j++){
			std::cout << name << "("<< i+1 << "," << j+1 << ")=" << A_[i+j*ldA] << ";" << std::endl;
		}
	}
}



template<class fp>
void init_vec_zeros(std::vector<fp> &v, int nVals){

	for (auto j=0;j<nVals; j++){
		v.push_back(static_cast<fp>(0));
	}

} 



template<class fp>
mpjd::Subspace<fp>::Subspace(Matrix<fp> &mat_, int dim_,int numEvals_, eigenTarget_t eigTarget_, int maxBasis_, LinearAlgebra &la_,
				 std::vector<fp> &w_, int &ldw_,	
				 std::vector<fp> &Q_, int &ldQ_, std::vector<fp> &L_, 
				 std::vector<fp> &R_, int &ldR_, 
				 std::vector<fp> &Qlocked_, int &ldQlocked_,std::vector<fp> &Llocked_,
				 std::vector<fp> &Rlocked_, int &ldRlocked_)

:
	la(la_),
	mat(mat_),
	dim(dim_),
	numEvals(numEvals_),
	eigTarget(eigTarget_),
	maxBasis(maxBasis_),
	blockSize(numEvals_),
	basisSize(0),
	lockedNumEvals(0),
	w(w_),
	ldw(ldw_),
	Q(Q_),ldQ(ldQ_),
	L(L_),
	R(R_),ldR(ldR_),
	Qlocked(Qlocked_),ldQlocked(ldQlocked_),
	Llocked(Llocked_),
	Rlocked(Rlocked_),ldRlocked(ldRlocked_)
{

	V.reserve(dim*maxBasis*numEvals); ldV = dim; // subspace basis
	
	T.reserve(maxBasis*numEvals*maxBasis*numEvals); ldT = maxBasis*numEvals;// projected matrix
	init_vec_zeros(T,maxBasis*numEvals*maxBasis*numEvals);
	
	q.reserve(maxBasis*numEvals*numEvals); ldq = maxBasis*numEvals; // eigenvectors of T
	init_vec_zeros(q,maxBasis*numEvals*numEvals);
	
	qq.reserve(maxBasis*numEvals*maxBasis*numEvals); ldqq = maxBasis*numEvals; // buffer for syev
	init_vec_zeros(qq,maxBasis*numEvals*maxBasis*numEvals);

	LL.reserve(maxBasis*numEvals); // buffer for syev
	init_vec_zeros(LL,maxBasis*numEvals);

	
	Q.reserve(dim*numEvals); ldQ = dim; // locked eigenvectors
	init_vec_zeros(Q,dim*numEvals);

	L.reserve(numEvals); 				        // locked eigenvalues
	init_vec_zeros(L,numEvals);


	R.reserve(dim*numEvals); ldR = dim; // eigen residual 
	init_vec_zeros(R,dim*numEvals);

	Qlocked.reserve(dim*numEvals); ldQlocked = dim; // locked eigenvectors

	Llocked.reserve(numEvals); 								      // locked eigenvalues

	Rlocked.reserve(dim*numEvals); ldRlocked = dim; // locked eigen residual 
	
	Aw.reserve(dim*blockSize); ldAw = dim;  // tmp vector to used in Av
	init_vec_zeros<fp>(Aw,dim*blockSize);

	AV.reserve(dim*blockSize*maxBasis); ldAV = dim;

	w.reserve(dim*blockSize);  ldw  = dim;  // new subspace direction
	
}



template<class fp>
void mpjd::Subspace<fp>::Subspace_init_direction(){

	/*
		Initialize basis V with random vectors
		Then orthogonalize them if in blocking mode
	*/

	std::random_device rand_dev;
	std::mt19937 			 generator(rand_dev());
	std::uniform_int_distribution<int> distr(-100,100);

	for(auto i=0;i<dim*numEvals;i++){
		w.push_back(static_cast<fp>(distr(generator))/static_cast<fp>(100));
	}

}


template<class fp>
void mpjd::Subspace<fp>::Subspace_orth_direction(){

	auto VV = static_cast<fp*>(V.data());
	auto vv = static_cast<fp*>(w.data());

	int rows = dim;
	int cols = basisSize*blockSize;

	for(int j=0; j < blockSize; j++){
		for(int i=0; i < cols ; i++){
				fp alpha {-la.dot(rows,&VV[0+i*ldV],1,&vv[0+j*ldV],1)};// alpha = V(i)'v
				la.axpy(rows,alpha,&VV[0+i*ldV],1,&vv[0+j*ldV],1);     	// v = v - V(i)*alpha
		}	
	}

	for(int j=0; j < blockSize; j++){
		for(int i=0; i < j ; i++){
				auto alpha {-la.dot(rows,&vv[0+i*ldV],1,&vv[0+j*ldV],1)};// alpha = V(i)'v
				la.axpy(rows,alpha,&vv[0+i*ldV],1,&vv[0+j*ldV],1);     	// v = v - V(i)*alpha
		}	
		auto alpha {la.nrm2(rows,&vv[0+j*ldV],1)}	;
		la.scal(dim,static_cast<fp>(1.0/alpha),&vv[0+j*ldV],1);    // v = v/norm(v)
	}
	
}


template<class fp>
void mpjd::Subspace<fp>::Subspace_project_at_new_direction(){

	mat.matVec(w,ldw, Aw, ldAw, blockSize);     // Aw  = A*w
	AV.insert(AV.end(), Aw.begin() , Aw.end()); // [AV = [AV Aw]

	fp *Aw_ = Aw.data();
	fp *w_  =  w.data();
	fp *T_  =  T.data();
	fp *V_  =  V.data();
		
	fp one  = static_cast<fp>(1.0);
	fp zero = static_cast<fp>(0.0);

	/**
		T = [	T 0; 0 w'Aw]
	**/
	
//	static int k=-1;
//	k++;
//	std::cout <<"\%!" << Aw.data() << std::endl;
	la.gemm('T','N', blockSize, blockSize, dim, one, w_, ldw,
						Aw_, ldAw, zero, &T_[basisSize*blockSize + basisSize*blockSize*ldT],ldT);	
						
//	printMat(T,ldT,10,10,"T0"+std::to_string(k));
	/* only the diagonal part is of interest */					
	if(basisSize < 1) return;

	/**
		T = [	T V'Aw; 0 w'Aw]
	**/
	
	la.gemm('T','N', basisSize*blockSize, blockSize, dim, one, V_, ldV,
						Aw_, ldAw, zero, &T_[0 + basisSize*blockSize*ldT],ldT);	
	
//	printMat(T,ldT,10,10,"T1"+std::to_string(k));
	/**
		T = [	T V'Aw; w'AV w'Aw]
	**/
	la.gemm('T','N', blockSize, basisSize*blockSize, dim, one, Aw_, ldAw,
						V_, ldV, zero, &T_[basisSize*blockSize + 0*ldT],ldT);	

//	printMat(T,ldT,10,10,"T2"+std::to_string(k));
	
}	

template<class fp> 
void mpjd::Subspace<fp>::Subspace_update_basis(){
	// V = [V w]
	V.insert(V.end(), w.begin() , w.end());
	basisSize = basisSize + 1;
//	std::cout <<"!" << basisSize << std::endl;
}


template<class fp>
void mpjd::Subspace<fp>::Subspace_projected_mat_eig(){
	
	
	qq = T; // copy data values from T to qq 
					// eig replaces QQ with eigenvectors of T

	la.eig((basisSize)*blockSize, qq.data() , ldqq, LL.data(), numEvals, eigTarget);

	// eig returns ALL eigenpairs of T
	// extract the wanted ones
	// LL contains eigenvalues in descending order

	L.clear(); // Clear previous
	q.clear(); // 

	auto LLstart_ = LL.begin();
	auto LLend_   = LL.begin();
	auto qqstart_  = qq.begin();
	auto qqend_    = qq.begin();

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


	L.insert(L.end(), LLstart_ , LLend_);
	q.insert(q.end(), qqstart_, qqend_);

	/* Q = V*qq */
	fp one  = static_cast<fp>(1.0);
	fp zero = static_cast<fp>(0.0);
	
	la.gemm('N', 'N',dim, blockSize, basisSize*blockSize,
					one, V.data() , ldV, q.data() , ldq, zero, Q.data() , ldQ);
	
}



template<class fp>
void mpjd::Subspace<fp>::Subspace_eig_residual(){
	// R = AV*q-Q*L
	fp *AV_ = AV.data();
	fp *q_  =  q.data();
	fp *Q_  =  Q.data();
	fp *L_  =  L.data();
	fp *R_  =  R.data();
	
	fp one        = static_cast<fp>(1.0);
	fp zero       = static_cast<fp>(0.0);
	fp minus_one  = static_cast<fp>(-1.0);
	
	// R = Q
	R = Q;
	
	// R = R*L = Q*L 
	for(auto j=0; j<numEvals; j++){
		la.scal(dim, L_[j], &R_[0+j*ldR], 1);
	}
	
	// R = AV*q-R = AV*q-Q*L
	la.gemm('N', 'N',dim, numEvals, basisSize*blockSize,
						one, AV_, ldAV, q_, ldq, minus_one, R_, ldR);



}

template<class fp>
void mpjd::Subspace<fp>::Check_Convergence_n_Lock(fp tol){

	fp *R_ = R.data();
	fp  rho{};
	std::vector<int> conv_num{};
	for(auto j=0;j<numEvals;j++){
		rho = la.nrm2(dim,&R_[0+j*ldR],1);

		if(rho < tol*mat.Norm()){
			conv_num.push_back(j);
		}
		
	}
	
	while(conv_num.size() != 0){
			/* pop last element of conv as j*/
			int j = conv_num.back();
			conv_num.pop_back();		
			/** 
				 insert j-th eigenpair
			*/
			Rlocked.insert(Rlocked.end(),R.at(0+j*ldQ),R.at(0 + j*ldQ+dim-1));
			Qlocked.insert(Qlocked.end(),Q.at(0+j*ldQ),Q.at(0 + j*ldQ+dim-1));
			Llocked.insert(Llocked.end(),L.at(j),L.at(j));
					
			/**
				  remove j-th eigenpair
			*/
			R.erase(R.begin()+ (0+j*ldQ),R.begin() +(0 + j*ldQ+dim));
			Q.erase(Q.begin()+ (0+j*ldQ),Q.begin() +(0 + j*ldQ+dim));
			L.erase(L.begin() + j);
			
			lockedNumEvals++;
			numEvals--;	
	}	
};


template<class fp>
int mpjd::Subspace<fp>::getBasisSize(){return basisSize;}

template<class fp>
int mpjd::Subspace<fp>::getMaxBasisSize(){return maxBasis;}

template<class fp>
bool mpjd::Subspace<fp>::Converged(){
	
	if(numEvals <= 0){
		return true;
	}else{
		return false;
	}
}













