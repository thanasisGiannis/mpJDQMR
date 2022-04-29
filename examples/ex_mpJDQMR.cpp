#include <iostream>
#include <string>
#include <iterator>
#include "mpjd.h"
#include "half.hpp"

#include <fstream>
#include <tgmath.h>

/* Read from mtx file a sparse matrix in COO representation */
template<class fp>
void readSymMtx(std::string matName,
				std::vector<int> &rows, std::vector<int> &cols, std::vector<fp> &vals, int &dim, fp &norm);
				

using namespace mpjd;
int main(){

//	Matrix<int> mat1{};
	
	
	std::vector<double>  vals;//{1.0,1.0,1.0,1.0};
	std::vector<int>     rows;//{0,0,1,2};
	std::vector<int>     cols;//{0,1,1,2}; 
	
	std::vector<double>     x;//{1,1,1};
	std::vector<double>     y;//{0,0,0};
	std::vector<double> 		Q{}; int ldQ;
	std::vector<double>     L{}; 
	std::vector<double>			R{}; int ldR;
	
	
	int dim{};
	double norm{};
	LinearAlgebra la;
	
//	readSymMtx<double>("../matrices/bfwb62.mtx",rows,cols,vals,dim,norm);
	readSymMtx<double>("../matrices/finan512.mtx",rows,cols,vals,dim,norm);

  mpjd::mpjdParam params;
  params.numEvals = 1;
  params.maxIters = 3*dim;
  params.dim = dim;
  params.tol = 1e-02;
  params.printStats = true;

	auto *jd = new JD<double,half>{vals, rows, cols, dim,
			sparseDS_t::COO, Q, ldQ, L, R, ldR, norm, params};
			
	jd->solve();	
	
	double *Q_ = Q.data();
	double  rho{};
	
	for(auto j=0;j<params.numEvals;j++){
	rho = *(L.data() + j);// a.nrm2(dim,&Q_[0+j*ldQ],1);
	//std::cout << "L(:,j) = " << *(L.data()+j) << std::endl;
	std::cout << "rho("<<j<<") = " << rho << std::endl;
		
	}

	delete jd;
}





/* Read from mtx file a sparse matrix in COO representation */
template<class fp>
void readSymMtx(std::string matName,
				std::vector<int> &rows, std::vector<int> &cols, std::vector<fp> &vals, int &dim, fp &norm){

		std::ifstream ifile{matName};
		int num_row, num_col, num_lines;

		norm = static_cast<fp>(0);
		// Ignore comments headers
		while (ifile.peek() == '%') ifile.ignore(2048, '\n');

		// Read number of rows and columns
		ifile >> num_col >> num_row >> num_lines;
		
		dim = num_row;

		// fill the matrix with data
		for (int l = 0; l < num_lines; l++)
		{
				fp val;
				int row, col;
				ifile >> col >> row >> val;
				col-=1;
				row-=1;
				//matrix[(row -1) + (col -1) * num_row] = data;
				rows.push_back(row);
				cols.push_back(col);
				vals.push_back(static_cast<fp>(val));

	
				if(row != col){
					// add the symmetric part
					rows.push_back(col);
					cols.push_back(row);
					vals.push_back(static_cast<fp>(val));
				}
				
				if(fabs(static_cast<fp>(val)	) > norm){ 
					norm = fabs(static_cast<fp>(val));
				}
		}
		ifile.close();
}

