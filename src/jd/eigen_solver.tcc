
template<class fp, class sfp>
void mpjd::JD<fp,sfp>::solve(){

	basis.Subspace_init_direction(); // w = rand()
	basis.Subspace_orth_direction(); // w = orth(w)
	basis.Subspace_project_at_new_direction(); // T= w'*A*w // 
	basis.Subspace_update_basis(); // V = [ V w]
	basis.Subspace_projected_mat_eig();   // [q,L] = eig(T)
	basis.Subspace_eig_residual(); // R = AV*q-Q*L
	
	// While no convergence
	//		w = R // step to be changed
	//    w = orth(V,w)
	//		T = w'*A*w
	//		V = [V w]
	//		[q,L] = eig(T)
	//		R = ΑV*q-Q*L)	
	
}

