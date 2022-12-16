#include <iomanip>
#include <math.h>

template<class fp, class sfp>
std::tuple<std::vector<fp>,std::vector<fp>,std::vector<fp>> 
  mpjd::JD<fp, sfp>::solve() {

  basis->Subspace_init_direction();           // w = rand()
  basis->Subspace_orth_direction();           // w = orth(w)
  basis->Subspace_project_at_new_direction(); // T= w'*A*w // 
  basis->Subspace_update_basis();             // V = [ V w]
  basis->Subspace_projected_mat_eig();        // [q,L] = eig(T); Q = V*q;

  int innerIters=0; // for statistics usage
  auto iter = 0;
  for( iter = 0; iter < maxIters; iter++ ) {
    
    basis->Subspace_eig_residual();             // R = AV*q-Q*L

    if( parameters.printIterStats )             // this has to be right after
      printIterationStats(iter);                // the residual vector calculation
      
    if( basis->Check_Convergence_and_lock(tol, iter == maxIters-1 ) ){
      break;
    }
    
    basis->Subspace_Check_size_and_restart();   // V = Q; T = L
    
    innerIters=0;
    //	w = (I-QlockedQlocked')(I-QQ')(A-L)(I-QlockedQlocked')(I-QQ')\R
    sqmr->solve(*w, dim, innerIters);                
    //*w = *R;
    
   	updateNumInnerLoop(innerIters);             // statistics
   	
   	//w = R;
    basis->Subspace_orth_direction();           // w = orth(V,w); 
                                                // w = orth(Qlocked,w)
                                                
    basis->Subspace_project_at_new_direction(); // T= w'*A*w // 
    basis->Subspace_update_basis();             // V = [ V w]
    basis->Subspace_projected_mat_eig();        // [q,L] = eig(T)
   
  }

  if( parameters.printStats ){
	  printStats();
  }
  
  return std::tuple<std::vector<fp>,
                    std::vector<fp>,
                    std::vector<fp>>(*Qlocked,*Llocked,*Rlocked);
}


