#include <iostream>
#include <string>
#include <iterator>
#include "matrix.h"
#include "jd.h"


using namespace mpJDQMR;
int main(){

//	Matrix<int> mat1{};
	
	
	std::vector<double>  vals{1.0,1.0,1.0,1.0};
	std::vector<int>     rows{0,2,3,4};
	std::vector<int>     cols{0,1,1,2};
	
	std::vector<double>     x{1,1,1};
	std::vector<double>     y{0,0,0};
	
	
	
	Matrix<double> *mat = new Matrix<double>{&vals,&rows,&cols};
	
	JD<double,double> *jd = new JD<double,double>(mat, mat->Dim(), 2, 15,1);
	jd->eigenSolve();
	
	mat->matVec(x.data(),3,y.data(),3,1);
	
	
	printVec(x,"x");
	printVec(y,"y");	

	delete mat;
	delete jd;
}
