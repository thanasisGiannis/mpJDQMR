#include "mpjd.h"
#include "half.hpp"

#include <iostream>
#include <string>
#include <iterator>
#include <fstream>
#include <tgmath.h>

/* Read from mtx file a sparse matrix in COO representation */
template<class fp>
void readSymMtx(std::string matName,
  std::vector<int> &rows, std::vector<int> &cols,
  std::vector<fp> &vals, int &dim, fp &norm) {

        std::ifstream ifile{matName};
        int num_row, num_col, num_lines;

        norm = static_cast<fp>(0);
        // Ignore comments headers
        while (ifile.peek() == '%') ifile.ignore(2048, '\n');

        // Read number of rows and columns
        ifile >> num_col >> num_row >> num_lines;

        dim = num_row;

        // fill the matrix with data
        for( int l = 0; l < num_lines; l++ ){
                fp val;
                int row, col;
                ifile >> col >> row >> val;
                col-=1;
                row-=1;
                //matrix[(row -1) + (col -1) * num_row] = data;
                rows.push_back(row);
                cols.push_back(col);
                vals.push_back(static_cast<fp>(val));


                if( row != col ) {
                    // add the symmetric part
                    rows.push_back(col);
                    cols.push_back(row);
                    vals.push_back(static_cast<fp>(val));
                }

                if( fabs(static_cast<fp>(val)) > norm ) {
                    norm = fabs(static_cast<fp>(val));
                }
        }
        ifile.close();
}



int main() {

    std::vector<double>  vals;//{1.0,1.0,1.0,1.0};
    std::vector<int>     rows;//{0,0,1,2};
    std::vector<int>     cols;//{0,1,1,2};

    std::vector<double>     x;//{1,1,1};
    std::vector<double>     y;//{0,0,0};

    int    dim{};
    double norm{};

    mpjd::LinearAlgebra la;

//	readSymMtx<double>("../matrices/bfwb62.mtx",rows,cols,vals,dim,norm);
    readSymMtx<double>("../matrices/finan512.mtx",rows,cols,vals,dim,norm);

  mpjd::mpjdParam params;
  params.numEvals = 5;
  params.maxIters = 30*dim;
  params.dim = dim;
  params.tol = 1e-8;
  params.printStats     = true;
  params.printIterStats = true;


    mpjd::JD<double,half> jd{vals, rows, cols, params.dim,
      mpjd::sparseDS_t::COO,
      //Q, ldQ, L, R, ldR,
      norm, params};

    auto t = jd.solve();

    std::vector<double> Q{};
    std::vector<double> L{};
    std::vector<double> R{};

    std::swap(Q,std::get<0>(t));
    std::swap(L,std::get<1>(t));
    std::swap(R,std::get<2>(t)); int ldR = dim;

    for(int j=0;j<params.numEvals;j++){
      double rho{la.nrm2(dim,R.data()+(0+j*ldR),1)};
      std::cout << "ritz("<<j<<") = "
                << *(L.data() + j)
                << " / rho("<<j<<") = "
                << rho/norm << std::endl;
    }

}






