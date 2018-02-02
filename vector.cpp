
#include <vector.h>
#include <string>
#include <vector.cuh>




Vector::Vector(std::vector<float> const & array_)
	: host_vector(array_)
{

	size = host_vector.size();


    checkCudaErrors(cudaMalloc((void **)&dev_vector, size *sizeof(float)));
	checkCudaErrors(cudaMemcpy(dev_vector, reinterpret_cast<float*>(host_vector.data()), size * sizeof(float), cudaMemcpyHostToDevice));
}

Vector::Vector(Vector const& rhs)
	: host_vector(rhs.host_vector), size(rhs.size)
{
	checkCudaErrors(cudaMalloc((void **)&dev_vector, size * sizeof(float)));
	checkCudaErrors(cudaMemcpy(dev_vector, reinterpret_cast<float*>(host_vector.data()), size * sizeof(float), cudaMemcpyHostToDevice));
}

Vector::Vector(const int size_, float *dev_array_)
	: size(size_), dev_vector(dev_array_)
{
	host_vector.resize(size);
	if(dev_vector == nullptr)
		checkCudaErrors(cudaMalloc((void **)&dev_vector, size * sizeof(float)));
}
    
std::vector<float> Vector::get_result()
{
    checkCudaErrors(cudaMemcpy(reinterpret_cast<float*>(host_vector.data()), dev_vector, size * sizeof(float), cudaMemcpyDeviceToHost));
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

    

Vector* operator+ (Vector const& lhs, Vector const& rhs) {

	assert(lhs.size == rhs.size);

	float* dev_result;
	checkCudaErrors(cudaMalloc((void **)&dev_result, lhs.size * sizeof(float)));

	vectoradd(rhs.dev_vector, lhs.dev_vector, dev_result, lhs.size);

	Vector* ref = new Vector(lhs.size, dev_result);

	return ref;
}


std::string Vector::repr()
{
	std::stringstream s;

	s << "CUDA vector:" << std::endl << str();

	return s.str();
}

Vector& Vector::operator+= (Vector const& rhs)
{
	assert(size == rhs.size);

	vectoradd(rhs.dev_vector, dev_vector, dev_vector, size);
	return *this;
}

/*
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
}*/



std::string  Vector::str()
{
	get_result();
	std::stringstream s;
	s << "[";
	
	for (auto const & n : host_vector)
		s << n << ", ";

	s << "]";
	return s.str();
}


