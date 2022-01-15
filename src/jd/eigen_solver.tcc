
template<class fp, class sfp>
void mpjd::JD<fp,sfp>::solve(){

	basis.Subspace_init_direction();           // w = rand()
	basis.Subspace_orth_direction();           // w = orth(w)
	basis.Subspace_project_at_new_direction(); // T= w'*A*w // 
	basis.Subspace_update_basis();             // V = [ V w]
	basis.Subspace_projected_mat_eig();        // [q,L] = eig(T); Q = V*q;
	basis.Subspace_eig_residual();             // R = AV*q-Q*L
	basis.Check_Convergence_n_Lock(tol);			 // Check for early convergence

	if(basis.Converged()){
		//return;
	}
	
	for(auto iter=0;iter<maxIters;iter++){


		// While no convergence
		w=R; //		w = R // step to be changed
		basis.Subspace_orth_direction();           // w = orth(V,w)

		basis.Subspace_project_at_new_direction(); // T= w'*A*w // 

		basis.Subspace_update_basis();             // V = [ V w]

		basis.Subspace_projected_mat_eig();        // [q,L] = eig(T)

		basis.Subspace_eig_residual();             // R = AV*q-Q*L
		
		// Check Convergence
//		basis.Check_Convergence_n_Lock(tol);
	
#if 1
		std::cout << "\%============== " <<  iter << std::endl;
		for(auto j=0;j<R.size()/dim;j++){
			fp *R_ = R.data();
			fp rho = la.nrm2(dim,&R_[0+j*ldR],1);
			std::cout << "\%||R(:," << j << ")||=" << rho/mat.Norm() << std::endl;
			std::cout << "\%L(" << j << ")=" << *(L.data()+j) << std::endl;
			
		}
		std::cout << "\%============== " <<  iter << std::endl;
#endif

		if(basis.Converged()){
			//return;
		}
	
#if 0
		// restart basis
		// if basisSize is not less than maxBasis then restart
		if(~(basis.getBasisSize() < basis.getMaxBasisSize())){
		
			std::cout << "restart basis" << std::endl;
			std::cout << "basisSize: "<< basis.getBasisSize() << std::endl;
			std::cout << "maxBasisSize: "<< basis.getMaxBasisSize() << std::endl;
			return;
			
		}
#endif

	}
}


