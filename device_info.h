#ifndef __DEVICE_INFO_H__
#define __DEVICE_INFO_H__

#ifdef _MSC_VER
	#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>

#ifdef __APPLE__
	#include <OpenCL/opencl.h>
#else
	#include <CL/cl.h>
#endif

#include "opencl_error.h"

bool print_device_info(void);

#endif