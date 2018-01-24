#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <tiny_helper_cuda.h>
#include <cassert>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <iostream>


namespace bp = boost::python;
namespace np = boost::python::numpy;



class Vector
{
protected:
	float * dev_vector = nullptr;
	np::ndarray host_vector;
	int size = 0;

public:

    Vector(np::ndarray const & array);
	Vector(Vector const& rhs);
	Vector(int size);
    
	np::ndarray get_result();
    
    void multiply_by(float k);
    
	
	Vector& operator+= (Vector const& rhs);

	//assignment  operator
	Vector& operator=(Vector const& );

	friend Vector& operator+ (Vector const& lhs, Vector const& rhs);

	friend std::ostream & operator<<(std::ostream & lhs, Vector  rhs); //Str

	~Vector();

    
};








