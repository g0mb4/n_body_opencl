#ifdef _MSC_VER
	#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef __APPLE__
	#include <OpenCL/opencl.h>
#else
	#include <CL/cl.h>
#endif

#include "opencl_error.h"
#include "device_info.h"
#include "n_body.h"

#define KERNEL_FILE	"./n_body.cl"
#define OUT_FILE	"./out.txt"

#define MAX_SOURCE_SIZE (0x100000)

#define N_BODIES		100
#define N_ITERATIONS	100

float frand(void){
	return (float)rand() / (float)RAND_MAX;
}

int main(int argc, char **argv)
{
	int a, i;

	cl_device_id device_id = NULL;
	cl_context context = NULL;
	cl_command_queue command_queue = NULL;
	cl_program program = NULL;
	cl_kernel kernel = NULL;
	cl_platform_id platform_id = NULL;
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;
	cl_int ret;

	body_mstate *host_bodies_mstate_old, *host_bodies_mstate_new;
	body_props *host_bodies_props;

	cl_mem dev_bodies_mstate_old, dev_bodies_mstate_new;
	cl_mem dev_bodies_props;

	int n_bodies = N_BODIES;
	float dt = 0.001;
	int interation, n_iterations = N_ITERATIONS;

	clock_t t_loop, t_total;
	t_total = clock();

	for (a = 0; a < argc; a++){
		if (!strcmp(argv[a], "--nbodies") || !strcmp(argv[a], "-n")){
			n_bodies = atoi(argv[++a]);
		} else if (!strcmp(argv[a], "--iters") || !strcmp(argv[a], "-i")){
			n_iterations = atoi(argv[++a]);
		}
	}

	FILE *fp;
	char *source_str;
	size_t source_size;

	/* Load the source code containing the kernel*/
	fp = fopen(KERNEL_FILE, "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}

	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);

	if (print_device_info() == false){
		fprintf(stderr, "Unable to get device information\n");
		exit(2);
	}

	host_bodies_mstate_old = (body_mstate*)malloc(n_bodies * sizeof(body_mstate));
	host_bodies_mstate_new = (body_mstate*)malloc(n_bodies * sizeof(body_mstate));
	host_bodies_props = (body_props*)malloc(n_bodies * sizeof(body_props));

	if (host_bodies_mstate_old == NULL || host_bodies_mstate_new == NULL || host_bodies_props == NULL){
		fprintf(stderr, "Failed to allocate the host memory\n");
		exit(3);
	}

	/* test */
	if (n_bodies == 3){
		cl_float3 foo;
		/* Planet 1 */
		memset(&foo, 0, sizeof(cl_float3));
		foo.x = -1.0f;
		host_bodies_mstate_old[0].pos = foo;
		host_bodies_mstate_new[0].pos = foo;

		memset(&foo, 0, sizeof(cl_float3));
		host_bodies_mstate_old[0].vel = foo;
		host_bodies_mstate_new[0].vel = foo;

		host_bodies_props[0].mass = 1.0f;
		host_bodies_props[0].radius = 0.1f;
		host_bodies_props[0].coll_coef = 0.4f;
		/* Planet 2 */
		memset(&foo, 0, sizeof(cl_float3));
		foo.x = 0.4f;
		foo.y = 0.4f;
		host_bodies_mstate_old[1].pos = foo;
		host_bodies_mstate_new[1].pos = foo;

		memset(&foo, 0, sizeof(cl_float3));
		foo.y = -1.0f;
		host_bodies_mstate_old[1].vel = foo;
		host_bodies_mstate_new[1].vel = foo;

		host_bodies_props[1].mass = 0.01f;
		host_bodies_props[1].radius = 0.1f;
		host_bodies_props[1].coll_coef = 0.5f;
		/* Planet 2 */
		memset(&foo, 0, sizeof(cl_float3));
		foo.x = 1.0f;
		host_bodies_mstate_old[2].pos = foo;
		host_bodies_mstate_new[2].pos = foo;

		memset(&foo, 0, sizeof(cl_float3));
		host_bodies_mstate_old[2].vel = foo;
		host_bodies_mstate_new[2].vel = foo;

		host_bodies_props[2].mass = 1.0f;
		host_bodies_props[2].radius = 0.1f;
		host_bodies_props[2].coll_coef = 0.3f;
	}
	else {
		srand(time(0));
		for (i = 0; i < n_bodies; i++){
			cl_float3 foo;

			foo.x = (frand() * 20.0f) - (10.0f);
			foo.y = (frand() * 20.0f) - (10.0f);
			foo.z = (frand() * 20.0f) - (10.0f);

			host_bodies_mstate_old[i].pos = foo;
			host_bodies_mstate_new[i].pos = foo;

			foo.x = frand() * 2.0f - 1.0f;
			foo.y = frand() * 2.0f - 1.0f;
			foo.z = frand() * 2.0f - 1.0f;

			host_bodies_mstate_old[i].vel = foo;
			host_bodies_mstate_new[i].vel = foo;

			host_bodies_props[i].mass = frand();
			host_bodies_props[i].radius = 0.1f;
			host_bodies_props[i].coll_coef = frand() * 0.5f;
		}
	}
	
	// Find number of platforms
	ret = clGetPlatformIDs(0, NULL, &ret_num_platforms);
	CHECK_ERROR(ret, "Finding platforms");
	if (ret_num_platforms == 0)
	{
		printf("Found 0 platforms!\n");
		return EXIT_FAILURE;
	}

	cl_platform_id *platforms = new cl_platform_id[ret_num_platforms];
	ret = clGetPlatformIDs(ret_num_platforms, platforms, NULL);
	CHECK_ERROR(ret, "Getting platforms");

	// Secure a GPU
	for (i = 0; i < ret_num_platforms; i++)
	{
		ret = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL);
		if (ret == CL_SUCCESS)
		{
			break;
		}
	}

	if (device_id == NULL)
		CHECK_ERROR(ret, "Finding a device");

	/* Create OpenCL context */
	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
	CHECK_ERROR(ret, "Create context");

	/* Create Command Queue */
	command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
	CHECK_ERROR(ret, "Creating command queue");

	/* Create Kernel Program from the source */
	program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
	CHECK_ERROR(ret, "Creating program");

	/* Build Kernel Program */
	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	if (ret != CL_SUCCESS)
	{
		size_t len;
		char buffer[2048];

		printf("Error: Failed to build program executable!\n%s\n", err_code(ret));
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		printf("%s\n", buffer);
		return EXIT_FAILURE;
	}

	/* Create OpenCL Kernel */
	kernel = clCreateKernel(program, "n_body", &ret);
	CHECK_ERROR(ret, "Creating kernel");

	// Create the input and output arrays in device memory
	dev_bodies_mstate_old = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(body_mstate) * n_bodies, NULL, &ret);
	CHECK_ERROR(ret, "Creating buffer dev_bodies_mstate_old");

	dev_bodies_mstate_new = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(body_mstate) * n_bodies, NULL, &ret);
	CHECK_ERROR(ret, "Creating buffer dev_bodies_mstate_new");

	dev_bodies_props = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(body_props) * n_bodies, NULL, &ret);
	CHECK_ERROR(ret, "Creating buffer dev_bodies_props");

	ret = clEnqueueWriteBuffer(command_queue, dev_bodies_props, CL_TRUE, 0, sizeof(body_props) * n_bodies, host_bodies_props, 0, NULL, NULL);
	CHECK_ERROR(ret, "Copying host_bodies_props to device at dev_bodies_props");

	// main loop
	for (interation = 0; interation < n_iterations; interation++){
		t_loop = clock();

		ret = clEnqueueWriteBuffer(command_queue, dev_bodies_mstate_old, CL_TRUE, 0, sizeof(body_mstate) * n_bodies, host_bodies_mstate_old, 0, NULL, NULL);
		CHECK_ERROR(ret, "Copying host_bodies_mstate_old to device at dev_bodies_mstate_old");

		/* Set OpenCL Kernel Parameters */
		ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &dev_bodies_mstate_old);
		ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &dev_bodies_props);
		ret |= clSetKernelArg(kernel, 2, sizeof(int), &n_bodies);
		ret |= clSetKernelArg(kernel, 3, sizeof(float), &dt);
		ret |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &dev_bodies_mstate_new);
		CHECK_ERROR(ret, "Setting kernel arguments");

		/* Define an index space (global work size) of work items for execution*/
		size_t global_work_size[1];
		global_work_size[0] = n_bodies;

		/* Execute OpenCL Kernel */
		ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);
		CHECK_ERROR(ret, "Execute kernel");

		ret = clFinish(command_queue);
		CHECK_ERROR(ret, "Waiting for kernel to finish");

		/* Copy results from the memory buffer */
		ret = clEnqueueReadBuffer(command_queue, dev_bodies_mstate_new, CL_TRUE, 0, sizeof(body_mstate) * n_bodies, host_bodies_mstate_new, 0, NULL, NULL);
		CHECK_ERROR(ret, "Copying dev_bodies_mstate_new to device at host_bodies_mstate_new");

		/* Iterate buffers */
		memcpy(host_bodies_mstate_old, host_bodies_mstate_new, sizeof(body_mstate) * n_bodies);

		t_loop = clock() - t_loop;
		double loop_time = ((double)t_loop) / CLOCKS_PER_SEC;

		printf("(%d / %d) done in %f s\n", (interation + 1), n_iterations, loop_time);
	}

	t_total = clock() - t_total;
	double total_time = ((double)t_total) / CLOCKS_PER_SEC;

	fp = fopen(OUT_FILE, "w");
	if (!fp) {
		fprintf(stderr, "Failed to save file.\n");
	} else {
		for (int i = 0; i < n_bodies; i++){
			cl_float3 pos = host_bodies_mstate_old[i].pos;
			cl_float3 vel = host_bodies_mstate_old[i].vel;
			cl_float mass = host_bodies_props[i].mass;
			fprintf(fp, "%f;%f;%f;%f;%f;%f;%f\n", pos.x, pos.y, pos.z, vel.x, vel.y, vel.z, mass);
		}

		fclose(fp);
	}

	printf("Simulation done! %d Planets. %d Iterations. %f s.\n", n_bodies, n_iterations, total_time);

	/* Finalization */
	ret = clFlush(command_queue);
	ret = clFinish(command_queue);
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(dev_bodies_mstate_old);
	ret = clReleaseMemObject(dev_bodies_mstate_new);
	ret = clReleaseMemObject(dev_bodies_props);
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);

	free(source_str);
	free(host_bodies_mstate_old);
	free(host_bodies_mstate_new);
	free(host_bodies_props);

	exit(EXIT_SUCCESS);
}