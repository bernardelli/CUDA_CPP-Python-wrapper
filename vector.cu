#define SIZE 128


#include <cuda.h>
#include <vector>
#include <tiny_helper_cuda.h>


namespace bn = boost::numpy;

__global__ void
vectormult_kernel( float *A, float k, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        A[i] *=  k;
    }
};

void vectormult(float * A, float k)
{
	vectormult_kernel <<<1, SIZE >>>(A, k, SIZE);
};


class Vector
{
private:
	float * dev_vector;
	std::vector<float> host_vector;

public:
	//Vector(const Vector&) = delete;
	//Vector& operator= (const Vector&) = delete;
    Vector(int x)
    {
        host_vector.reserve(SIZE);
        checkCudaErrors(cudaMalloc((void **)&dev_vector, SIZE*sizeof(float)));
    };
    
    std::vector<float> get_result()
    {
        checkCudaErrors(cudaMemcpy(host_vector.data(), dev_vector, SIZE*sizeof(float), cudaMemcpyDeviceToHost));
        return host_vector;
    };
    
    void multiply_by(float k)
    {
		vectormult(dev_vector, k);
		//vectormult <<<1, SIZE>>>(dev_vector, x, SIZE);
	};
    
    
    
    void fill(std::vector<float> data)
    {
        host_vector = data;
		
        checkCudaErrors(cudaMemcpy(dev_vector, host_vector.data(), SIZE*sizeof(float), cudaMemcpyHostToDevice));
    };
    
    
    
	~Vector()
	{
		checkCudaErrors(cudaFree(dev_vector));
	};

    
};




#include <boost/python.hpp>

BOOST_PYTHON_MODULE(vector_wrapped)
{
	namespace python = boost::python;

	bn::initialize();

	python::class_<Vector>("Vector", python::init<int>())
		.def("fill", &Vector::fill)
		.def("get_result", &Vector::get_result)
		.def("multiply_by", &Vector::multiply_by)
		;
}
