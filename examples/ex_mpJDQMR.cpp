#include <iostream>
#include <string>
#include <iterator>
#include "mpjd.h"
#include <fstream>

/* Read from mtx file a sparse matrix in COO representation */
template<class fp>
void readSymMtx(std::string matName,
				std::vector<int> &rows, std::vector<int> &cols, std::vector<fp> &vals, int &dim);
				

using namespace mpjd;
int main(){

//	Matrix<int> mat1{};
	
	
	std::vector<double>  vals;//{1.0,1.0,1.0,1.0};
	std::vector<int>     rows;//{0,0,1,2};
	std::vector<int>     cols;//{0,1,1,2}; 
	
	std::vector<double>     x;//{1,1,1};
	std::vector<double>     y;//{0,0,0};
	int dim{};
	int numEvals{5};
	int maxBasis{15};
	readSymMtx<double>("../matrices/bfwb62.mtx",rows,cols,vals,dim);

	
	JD<double,double> *jd = new JD<double,double>{vals, rows, cols, dim,
			sparseDS_t::COO,numEvals, eigenTarget_t::SM, maxBasis,  target_t::HOST};
	jd->solve();	
	
	delete jd;
}





/* Read from mtx file a sparse matrix in COO representation */
template<class fp>
void readSymMtx(std::string matName,
				std::vector<int> &rows, std::vector<int> &cols, std::vector<fp> &vals, int &dim){

		std::ifstream ifile{matName};
		int num_row, num_col, num_lines;

		// Ignore comments headers
		while (ifile.peek() == '%') ifile.ignore(2048, '\n');

		// Read number of rows and columns
		ifile >> num_row>> num_col >> num_lines;
		dim =num_row;
		// Create 2D array and fill with zeros
		//double* matrix;              
		//matrix = new double[num_row * num_col];      
		//std::fill(matrix, matrix + num_row *num_col, 0.);.

		// fill the matrix with data
		for (int l = 0; l < num_lines; l++)
		{
				fp val;
				int row, col;
				ifile >> row >> col >> val;
				//matrix[(row -1) + (col -1) * num_row] = data;
				rows.push_back(row-1);
				cols.push_back(col-1);
				vals.push_back(val);
				
				if(rows != cols){
					// add the symmetric part
					rows.push_back(col-1);
					cols.push_back(row-1);
				}
		}

		ifile.close();
}

