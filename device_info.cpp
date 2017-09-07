#include "device_info.h"

bool print_device_info(void){

	cl_int ret;
	// Find the number of OpenCL platforms
	cl_uint num_platforms = 0;
	ret = clGetPlatformIDs(0, NULL, &num_platforms);
	if (num_platforms == 0)
	{
		printf("Found 0 platforms!\n");
		return false;
	}
	// Create a list of platform IDs
	cl_platform_id *platform = new cl_platform_id[num_platforms];
	ret = clGetPlatformIDs(num_platforms, platform, NULL);
	if (ret != CL_SUCCESS){
		printf("Unable to get platform list\n");
		return false;
	}

	printf("\nNumber of OpenCL platforms: %d\n", num_platforms);
	printf("\n-------------------------\n");

	// Investigate each platform
	for (unsigned int i = 0; i < num_platforms; i++)
	{
		cl_char string[10240] = { 0 };
		// Print out the platform name
		ret = clGetPlatformInfo(platform[i], CL_PLATFORM_NAME, sizeof(string), &string, NULL);
		if (ret != CL_SUCCESS){
			printf("Unable to get platform name\n");
			return false;
		}
		printf("Platform: %s\n", string);

		// Print out the platform vendor
		ret = clGetPlatformInfo(platform[i], CL_PLATFORM_VENDOR, sizeof(string), &string, NULL);
		if (ret != CL_SUCCESS){
			printf("Unable to get platform vendor\n");
			return false;
		}
		printf("Vendor: %s\n", string);

		// Print out the platform OpenCL version
		ret = clGetPlatformInfo(platform[i], CL_PLATFORM_VERSION, sizeof(string), &string, NULL);
		if (ret != CL_SUCCESS){
			printf("Unable to get OpenCL version\n");
			return false;
		}
		printf("Version: %s\n", string);

		// Count the number of devices in the platform
		cl_uint num_devices;
		ret = clGetDeviceIDs(platform[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
		if (ret != CL_SUCCESS){
			printf("Unable to find devices\n");
			return false;
		}

		// Get the device IDs
		cl_device_id *device = new cl_device_id[num_devices];
		ret = clGetDeviceIDs(platform[i], CL_DEVICE_TYPE_ALL, num_devices, device, NULL);
		if (ret != CL_SUCCESS){
			printf("Unable to get devices\n");
			return false;
		}
		printf("Number of devices: %d\n", num_devices);

		// Investigate each device
		for (unsigned int j = 0; j < num_devices; j++)
		{
			printf("-------------------------\n");

			// Get device name
			ret = clGetDeviceInfo(device[j], CL_DEVICE_NAME, sizeof(string), &string, NULL);
			if (ret != CL_SUCCESS){
				printf("Unable to get device name\n");
				continue;
			}
			printf("\tName: %s\n", string);

			// Get device OpenCL version
			ret = clGetDeviceInfo(device[j], CL_DEVICE_OPENCL_C_VERSION, sizeof(string), &string, NULL);
			if (ret != CL_SUCCESS){
				printf("Unable to get OpenCL version\n");
				continue;
			}
			printf("\tVersion: %s\n", string);

			// Get Max. Compute units
			cl_uint num;
			ret = clGetDeviceInfo(device[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &num, NULL);
			if (ret != CL_SUCCESS){
				printf("Unable to get device max compute units\n");
				continue;
			}
			printf("\tMax. Compute Units: %d\n", num);

			// Get local memory size
			cl_ulong mem_size;
			ret = clGetDeviceInfo(device[j], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &mem_size, NULL);
			if (ret != CL_SUCCESS){
				printf("Unable to get device local memory size\n");
				continue;
			}
			printf("\tLocal Memory Size: %llu KB\n", mem_size / 1024);

			// Get global memory size
			ret = clGetDeviceInfo(device[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &mem_size, NULL);
			if (ret != CL_SUCCESS){
				printf("Unable to get device global memory size\n");
				continue;
			}
			printf("\tGlobal Memory Size: %llu MB\n", mem_size / (1024 * 1024));

			// Get maximum buffer alloc. size
			ret = clGetDeviceInfo(device[j], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &mem_size, NULL);
			if (ret != CL_SUCCESS){
				printf("Unable to get device max allocation size\n");
				continue;
			}
			printf("\tMax Alloc Size: %llu MB\n", mem_size / (1024 * 1024));

			// Get work-group size information
			size_t size;
			ret = clGetDeviceInfo(device[j], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &size, NULL);
			if (ret != CL_SUCCESS){
				printf("Unable to get device max work-group size\n");
				continue;
			}
			printf("\tMax Work-group Total Size: %ld\n", size);

			// Find the maximum dimensions of the work-groups
			ret = clGetDeviceInfo(device[j], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &num, NULL);
			if (ret != CL_SUCCESS){
				printf("Unable to get device max wor item dimensons\n");
				continue;
			}
			// Get the max. dimensions of the work-groups
			size_t* dims = new size_t[num];
			ret = clGetDeviceInfo(device[j], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(dims), &dims, NULL);
			if (ret != CL_SUCCESS){
				printf("Unable to get device max work-item size\n");
				continue;
			}
			printf("\tMax Work-group Dims: ( ");

			for (size_t k = 0; k < num; k++)
			{
				printf("%ld ", dims[k]);
			}
			printf(")\n");
			
		}

		printf("\n-------------------------\n");
	}

	return true;
}