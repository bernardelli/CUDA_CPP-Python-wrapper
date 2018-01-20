cmake_minimum_required(VERSION 3.8)


project (TestWrapper VERSION 0.99 DESCRIPTION "C++ CUDA Wrapper Test")


find_package(CUDA REQUIRED)
find_package(Boost REQUIRED COMPONENTS python)
find_package(PythonLibs REQUIRED)
find_package(NumPy REQUIRED)




include_directories(vectorlib ${PYTHON_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/inc)


#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -g -G;)  

cuda_add_library(vectorlib vector.cu)

target_link_libraries(vectorlib ${CUDA_LIBRARIES}  ${PYTHON_LIBRARIES} ${Boost_LIBRARIES}) 