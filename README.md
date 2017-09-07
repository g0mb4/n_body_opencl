# n_body_opencl
n body simulation in OpenCL

# project
choose one of these at compilation: 
* n_body_opencl.cpp - n body simulation in OpenCL
* n_body_opencl_glut.cpp - n body simulation in OpenCL with visulaisation (slower)

# compilation
the following stuffs are needed in order to compile:
* n_body_opencl - OpenCL headers/libs
* n_body_opencl_glut - OpenCL headers/libs, freeglut headers/libs

# usage
```
n_body_opencl -n <number_of_bodies> -i <number_of_iterations>
```
```
n_body_opencl_glut -n <number_of_bodies>
press Esc to exit
```
