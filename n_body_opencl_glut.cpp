#ifdef _MSC_VER
	#define _CRT_SECURE_NO_WARNINGS

	#include <Windows.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#ifdef __APPLE__
	#include <OpenCL/opencl.h>
#else
	#include <CL/cl.h>
#endif

#include "opencl_error.h"
#include "device_info.h"
#include "n_body.h"

#include <GL/GL.h>
#include <GL/glut.h>
#include <GL/freeglut.h>

#define KERNEL_FILE	"./n_body.cl"
#define OUT_FILE	"./out.txt"

#define MAX_SOURCE_SIZE (0x100000)

#define N_BODIES	100

#define CIRCLE_POINTS	10

#define FPS_SAMPLE		1000	// ms
int fps_ctr = 0;
clock_t t_fps;
double fps = 0.0;

int width = 600, height = 600;
bool mouseIn = false;
unsigned char key = 0;
int mx, my;

unsigned long int frame_ctr = 0;;

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

clock_t t_loop;

void changeSize(int w, int h)
{
	if (h == 0)
		h = 1;

	float ratio = (float)w / (float)h;

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	glViewport(0, 0, w, h);

	gluPerspective(45, ratio, 1, 1000);

	glMatrixMode(GL_MODELVIEW);
}

/* mass = [0.0 ... 1.0] */
cl_float3 getGradient(float mass){

	cl_float3 colorA = { 0.0f, 0.0f, 1.0f };
	cl_float3 colorB = { 1.0f, 0.0f, 0.0f };

	cl_float3 color;
	
	color.x = colorA.x + mass * (colorB.x - colorA.x);
	color.y = colorA.y + mass * (colorB.y - colorA.y);
	color.z = colorA.z + mass * (colorB.z - colorA.z);

	return color;
}

void displayText(float x, float y, const char *string) {
	int j = strlen(string);

	glColor3f(1.0, 1.0, 1.0);
	glRasterPos2f(x, y);
	for (int i = 0; i < j; i++) {
		glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, string[i]);
	}
}

void renderScene(void)
{
	t_fps = clock();
	// update
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

	// draw
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glLoadIdentity();

	glMatrixMode(GL_MODELVIEW);

	glTranslatef(0.0, 0.0f, -20.0f);

	for (int i = 0; i< n_bodies; i++)
	{
		cl_float3 c = getGradient(host_bodies_props[i].mass);
		glColor4f(c.x, c.y, c.z, 1.0f);
		glBegin(GL_LINE_STRIP);
		for (int j = 0; j < CIRCLE_POINTS + 1; j++)
		{
			float Angle = j * (2.0 * 3.14159265359 / CIRCLE_POINTS);
			float X = cos(Angle) * 0.1 + host_bodies_mstate_old[i].pos.x;
			float Y = sin(Angle) * 0.1 + host_bodies_mstate_old[i].pos.y;
			glVertex2f(X, Y);
		}
		glEnd();
	}

	t_fps = clock() - t_fps;
	double loop_time = ((double)t_fps) / CLOCKS_PER_SEC;

	fps = 1.0 / loop_time;

	char msg[64];
	sprintf(msg, "n_body_opencl_glut | p = %d, t = %.3f s, FPS = %.1f", n_bodies, ((double)frame_ctr * dt), fps);

	//displayText(-7, 7, msg);

	glutSwapBuffers();


	glutSetWindowTitle(msg);
	frame_ctr++;
	key = 0;
}

void processNormalKeys(unsigned char pressedKey, int x, int y)
{
	if (pressedKey == 27){
		glutLeaveMainLoop();

		FILE *fp = fopen(OUT_FILE, "w");
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

	}

	if (pressedKey == 'w' || pressedKey == 'a' || pressedKey == 's' || pressedKey == 'd')
		key = pressedKey;
}

void mouseButton(int button, int state, int x, int y)
{
	if (button == GLUT_LEFT_BUTTON)
	{
	}
}

void mouseMove(int xx, int yy)
{
	mx = xx;
	my = yy;
}

float frand(void){
	return (float)rand() / (float)RAND_MAX;
}

int main(int argc, char **argv)
{
	int a, i;

	for (a = 0; a < argc; a++){
		if (!strcmp(argv[a], "--nbodies") || !strcmp(argv[a], "-n")){
			n_bodies = atoi(argv[++a]);
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

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(100, 100);
	glutInitWindowSize(width, height);
	glutCreateWindow("n_body_opencl_glut");

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

	glutDisplayFunc(renderScene);
	glutReshapeFunc(changeSize);
	glutIdleFunc(renderScene);

	glutKeyboardFunc(processNormalKeys);

	glutMouseFunc(mouseButton);
	glutPassiveMotionFunc(mouseMove);

	glEnable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);

	glutMainLoop();

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