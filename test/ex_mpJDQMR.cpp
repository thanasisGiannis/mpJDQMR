#include <iostream>
#include <string>
#include "matrix.h"
#include <iterator>

template <class fp>
void printVec	(std::vector<fp> vec, std::string s){
	std::cout << s << std::endl;
	for(auto v = vec.begin(); v!=vec.end(); v++){
		std::cout << *v << std::endl;
	}
}

using namespace mpJDQMR;
int main(){

//	Matrix<int> mat1{};
	
	
	std::vector<double>  vals{1.0,1.0,1.0,1.0};
	std::vector<int>     rows{0,2,3,4};
	std::vector<int>     cols{0,1,1,2};
	
	std::vector<double>     x{1,1,1};
	std::vector<double>     y{0,0,0};
	
	
	
	Matrix<double> mat{&vals,&rows,&cols};
	mat.matVec(&x,3,&y,3,1);
	
	printVec(x,"x");
	printVec(y,"y");	
}
