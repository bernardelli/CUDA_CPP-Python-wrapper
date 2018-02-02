#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <tiny_helper_cuda.h>
#include <cassert>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include<iostream>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

namespace py = pybind11;



class Vector
{
protected:
	float * dev_vector = nullptr;
	int size = 0;
	std::vector<float> host_vector;

public:

    Vector(std::vector<float> const & array);
	Vector(Vector const & rhs);
	Vector(int size, float *dev_array = nullptr);



	std::vector<float> get_result();
    
    void multiply_by(float k);
    
	
	Vector& operator+= (Vector const& rhs);

	//assignment  operator
	//Vector& operator=(Vector const& );

	friend Vector* operator+ (Vector const& lhs, Vector const& rhs);

	std::string  str(); //Str

	std::string repr();

	~Vector();

    
};


PYBIND11_MODULE(vectorlib, m)
{


	py::class_<Vector>(m, "Vector")
		.def(py::init<std::vector<float> const &>())
		.def(py::init<int>())
		.def(py::init<Vector const&>())
		.def("get_result", &Vector::get_result)
		.def("multiply_by", &Vector::multiply_by)
		.def(py::self += py::self)
		.def(py::self + py::self)
		.def("__str__", &Vector::str)
		.def("__repr__", &Vector::repr);

}




//return_value_policy<manage_new_object>()
//with_custodian_and_ward<...>()


