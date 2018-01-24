
#include <vector.h>
#include <string>
#include <vector.cuh>

namespace bp = boost::python;
namespace np = boost::python::numpy;



Vector::Vector(np::ndarray const & array)
	: host_vector(array.astype(np::dtype::get_builtin<float>()))
{

	size = host_vector.shape(0);

	std::cout << size << std::endl;

    checkCudaErrors(cudaMalloc((void **)&dev_vector, size *sizeof(float)));
	checkCudaErrors(cudaMemcpy(dev_vector, reinterpret_cast<float*>(host_vector.get_data()), size * sizeof(float), cudaMemcpyHostToDevice));
}

Vector::Vector(Vector const& rhs)
	: host_vector(rhs.host_vector), size(rhs.size)
{
	checkCudaErrors(cudaMalloc((void **)&dev_vector, size * sizeof(float)));
	checkCudaErrors(cudaMemcpy(dev_vector, reinterpret_cast<float*>(host_vector.get_data()), size * sizeof(float), cudaMemcpyHostToDevice));
}

Vector::Vector(int size_)
	: size(size_), host_vector(np::empty(1, reinterpret_cast<Py_intptr_t *>(&size_), np::dtype::get_builtin<float>()))
{
	checkCudaErrors(cudaMalloc((void **)&dev_vector, size * sizeof(float)));
}
    
np::ndarray Vector::get_result()
{
    checkCudaErrors(cudaMemcpy(reinterpret_cast<float*>(host_vector.get_data()), dev_vector, size * sizeof(float), cudaMemcpyDeviceToHost));
    return host_vector;
}
    
void Vector::multiply_by(float k)
{
	vectormult(dev_vector, k, size);
}
     
    
Vector::~Vector()
{
	checkCudaErrors(cudaFree(dev_vector));
}

    

Vector& operator+ (Vector const& lhs, Vector const& rhs) {

	assert(lhs.size == rhs.size);

	Vector tmp(lhs.size);

	vectoradd(rhs.dev_vector, lhs.dev_vector, tmp.dev_vector, lhs.size);

	return tmp;
}

Vector& Vector::operator+= (Vector const& rhs)
{
	assert(size == rhs.size);

	vectoradd(rhs.dev_vector, dev_vector, dev_vector, size);
	return *this;
}

Vector& Vector::operator=(Vector const& rhs)
{
	if (size <= rhs.size)
	{
		size = rhs.size;
		checkCudaErrors(cudaFree(dev_vector));
		checkCudaErrors(cudaMalloc((void **)&dev_vector, size * sizeof(float)));
		host_vector = np::empty(1, reinterpret_cast<Py_intptr_t *>(&size), np::dtype::get_builtin<float>());
	}

	checkCudaErrors(cudaMemcpy(dev_vector, reinterpret_cast<float*>(host_vector.get_data()), size * sizeof(float), cudaMemcpyHostToDevice));


	return *this;
}



std::ostream & operator<<(std::ostream & lhs, Vector  rhs)
{
	bp::str str = bp::str(rhs.get_result());
	char const* c_str = bp::extract<char const*>(str);
	lhs << c_str;
	return lhs;
}


BOOST_PYTHON_MODULE(vectorlib)
{

	np::initialize();

	bp::class_<Vector>("Vector", bp::init<np::ndarray const &>())
		.def(bp::init<int>())
		.def("get_result", &Vector::get_result)
		.def("multiply_by", &Vector::multiply_by)
		.def(bp::self + bp::self)
		.def(bp::self += bp::self)
		.def(str(bp::self))
		;
}