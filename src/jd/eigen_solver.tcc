#include <iomanip>
#include <math.h>
template<class fp, class sfp>
void mpjd::JD<fp,sfp>::solve(){

	basis->Subspace_init_direction();           // w = rand()
	basis->Subspace_orth_direction();           // w = orth(w)
	basis->Subspace_project_at_new_direction(); // T= w'*A*w // 
	basis->Subspace_update_basis();             // V = [ V w]
	basis->Subspace_projected_mat_eig();        // [q,L] = eig(T); Q = V*q;
	
	
	for(auto iter=0;iter<maxIters;iter++){
    
    basis->Subspace_eig_residual();             // R = AV*q-Q*L

	  // Check Convergence
	  if(basis->Check_Convergence(tol)){
		  return;
	  }

    
    basis->Subspace_Check_size_and_restart();   // V = Q; T = L
    
    w = sqmr->solve();                          //	w = (I-QlockedQlocked')(I-QQ')(A-L)(I-QlockedQlocked')(I-QQ')\R 
   	//w = R;
   	
		basis->Subspace_orth_direction();           // w = orth(V,w); w = orth(Qlocked,w)

		basis->Subspace_project_at_new_direction(); // T= w'*A*w // 

		basis->Subspace_update_basis();             // V = [ V w]

		basis->Subspace_projected_mat_eig();        // [q,L] = eig(T)

		
#if 1
		//std::cout << "\%============== " <<  iter << std::endl;
		for(auto j=0;j<R.size()/dim;j++){
			fp *R_ = R.data();
			fp rho = la.nrm2(dim,&R_[0+j*ldR],1);
			std::cout << std::setprecision(3) 
								<<  std::scientific
								<< "\%||R(" << iter << "," << j << ")||=" << rho/mat.Norm() << std::endl;
			//std::cout << "\%L(" << j << ")=" << *(L.data()+j) << std::endl;
			
		}
		//std::cout << "\%============== " <<  iter << std::endl;
#endif
	}
	
}


