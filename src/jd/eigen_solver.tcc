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

	  if(basis->Check_Convergence(tol)){
		  break;
	  }

    
    basis->Subspace_Check_size_and_restart();   // V = Q; T = L
    
    w = sqmr->solve();                          //	w = (I-QlockedQlocked')(I-QQ')(A-L)(I-QlockedQlocked')(I-QQ')\R 
   	//w = R;
   	
		basis->Subspace_orth_direction();           // w = orth(V,w); w = orth(Qlocked,w)

		basis->Subspace_project_at_new_direction(); // T= w'*A*w // 

		basis->Subspace_update_basis();             // V = [ V w]

		basis->Subspace_projected_mat_eig();        // [q,L] = eig(T)

		
    if(parameters.printIterStats) printIterationStats(iter);

	}
	
	if(parameters.printStats){
  	printStats();
  }
}


