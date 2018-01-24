#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <tiny_helper_cuda.h>

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

namespace bp = boost::python;
namespace np = boost::python::numpy;


class Vector
{
private:
	float * dev_vector;
	np::ndarray host_vector;
	int size;

public:

    Vector(np::ndarray const & array);
    
	np::ndarray get_result();
    
    void multiply_by(float k);
     
    
	~Vector();

    
};








