#pragma once
#include <iostream>
#include <string>
namespace mpjd {

enum class target_t
		 {  
		 		// following future targets 
		 		// DEVICE, //[=GPU]
		 		HOST
		 	};


class LinearAlgebra {

	target_t target;	
public:
	
	LinearAlgebra(target_t target_);

	std::string getTarget();
	
	double dot(int dim, double *x, int incx, double *y, int incy);
	void axpy(int dim, double alpha, double *x, int incx, double *y, int incy);
	double nrm2(int dim, double *x,int incx);
	void scal(int dim, double alpha, double *x, int incx);


};
}
//void dot();

