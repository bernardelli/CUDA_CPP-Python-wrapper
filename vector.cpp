
#include <vector.h>
#include <iostream>

extern "C"
void vectormult(float* dev_vector, float k, int size);

namespace bp = boost::python;
namespace np = boost::python::numpy;



Vector::Vector(np::ndarray const & array)
	: host_vector(array.astype(np::dtype::get_builtin<float>()))
{

	
	//Py_intptr_t shape[1] = { SIZE };
	//np::ndarray result = np::zeros(1, shape, np::dtype::get_builtin<float>());

	
	size = host_vector.shape(0);

	std::cout << size << std::endl;

    checkCudaErrors(cudaMalloc((void **)&dev_vector, size *sizeof(float)));
	checkCudaErrors(cudaMemcpy(dev_vector, reinterpret_cast<float*>(host_vector.get_data()), size * sizeof(float), cudaMemcpyHostToDevice));
}
    
np::ndarray Vector::get_result()
{
    checkCudaErrors(cudaMemcpy(reinterpret_cast<float*>(host_vector.get_data()), dev_vector, size * sizeof(float), cudaMemcpyDeviceToHost));
    return host_vector.copy();
}
    
void Vector::multiply_by(float k)
{
	vectormult(dev_vector, k, size);
}
     
    
Vector::~Vector()
{
	checkCudaErrors(cudaFree(dev_vector));
}

    






BOOST_PYTHON_MODULE(vectorlib)
{

	np::initialize();

	bp::class_<Vector>("Vector", bp::init<np::ndarray const &>())
		.def("get_result", &Vector::get_result)
		.def("multiply_by", &Vector::multiply_by)
		;
}


