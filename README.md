<h2> Building a  CUDA/C++ Python Wrapper with CMake</h2>

This repository will give you a short example on how to build a C++ class with CUDA bindings into a  Python wrapper.

<h5>Why would I use this?</h5>
It is a good idea to test a larger CUDA/C++ project with all the swiftness and convenience of Python.

<h5>Why CMake?</h5>

CMake is a great tool when you want a project to run on different computers with different OS or library paths, but sometimes writing the CMake configuration file can be a pain in the ass. I ~~kinda~~ managed to make it work, so enjoy  :wink:

<h5>Requirements:</h5>

* CMake
* Boost
* CUDA
* Python

<h5>To Do list</h5>

- [x] make it work on Visual Studio
- [ ] clean out the code and write useful comments
- [ ] make it work on Linux/gcc
- [ ] write some cool examples here