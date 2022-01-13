
template<class fp, class sfp>
void mpjd::JD<fp,sfp>::solve(){

	basis.Subspace_init_direction();           // w = rand()
	basis.Subspace_orth_direction();           // w = orth(w)
	basis.Subspace_project_at_new_direction(); // T= w'*A*w // 
	basis.Subspace_update_basis();             // V = [ V w]
	basis.Subspace_projected_mat_eig();        // [q,L] = eig(T)
	basis.Subspace_eig_residual();             // R = AV*q-Q*L
	basis.Check_Convergence_n_Lock(tol);			 // Check for early convergence

	if(basis.Converged()){
		return;
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
		basis.Check_Convergence_n_Lock(tol);
	
		// if 	
		if(basis.Converged()){
			return;
		}
		// restart basis
		// if basisSize is not less than maxBasis then restart
		if(~(basis.getBasisSize() < basis.getMaxBasisSize()-1)){
		
			std::cout << "restart basis" << std::endl;
			std::cout << "basisSize: "<< basis.getBasisSize() << std::endl;
			std::cout << "maxBasisSize: "<< basis.getMaxBasisSize() << std::endl;
			
			return;
			
		}


	}
}


