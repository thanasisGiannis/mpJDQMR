#pragma once

template<class fp>
void printVec(std::vector<fp> &x,std::string S){

	std::cout << S << std::endl;
	fp *xx = x.data();

	for(auto j=0;j<x.size();j++){
			std::cout << j << ": " << xx[j] << std::endl;
	}
}


