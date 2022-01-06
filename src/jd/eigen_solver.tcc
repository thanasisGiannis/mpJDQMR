
template<class fp, class sfp>
void mpjd::JD<fp,sfp>::solve(){

	basis.Subspace_init_direction(); // w = rand()
	basis.Subspace_orth_direction(); // w = orth(w)
	basis.Subpsace_project_at_new_direction(); // T= w'*A*w // 
	
#if 0	
	basis.Subspace_projection_matrix_update(w,ldw);
	basis.Subspace_extract_ritz_pairs();
	basis.Subspace_rits_residual();	
#endif
}

