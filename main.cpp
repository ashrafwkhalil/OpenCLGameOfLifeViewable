#define CL_TARGET_OPENCL_VERSION 200
#define __CL_ENABLE_EXCEPTIONS

// has to be first! 
#include "GL/glew.h"

// define includes
#include "definitions.h"

// system includes
#include <ctime>
#include <fstream>
#include <iostream>
#include <ostream>
#include <string>
#include <stdexcept>
#include <vector>
//#include <vector>
//#include <exception>

// OpenGL includes
#if defined (__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#ifndef glutCloseFunc
#define glutCloseFunc glutWMCloseFunc
#endif
#else
#include <GL/freeglut.h>
#endif

// OpenCL includes
#include <CL/opencl.h>

// my includes
#include "kernels.h"
int xxCompute(int i) {
	return i % WIDTH;
}
int yyCompute(int i) {
	return i / WIDTH;
}


// variables
cl_uint2 m_positions[WIDTH * HEIGHT];
cl_uint2* p_pos = m_positions;

cl_device_id* cdDevices;
cl_device_id* cdDevices2;
cl_uchar3 m_colors[WIDTH * HEIGHT];
cl_uchar3* p_col = m_colors;

cl_uchar3 m_colors_read[WIDTH * HEIGHT];
cl_uchar3* p_read = m_colors_read;

cl_ulong m_rand_ulong[10];
cl_ulong* p_ulong_rand = m_rand_ulong;

cl_ulong current_time = 0;
cl_ulong* p_current_time = &current_time;

cl_int err; // for returning errors

cl_uint num_platforms;
cl_platform_id platform_ids[1];

cl_uint num_devices;
cl_device_id device_ids[2]; // only two devices expected 0 is GPU 1 is CPU

cl_context m_context, m_contextcpu; // 1 platform, 2 devices on my laptop

cl_command_queue queue_gpu, queue_cpu; // for a laptop with integrated graphics

cl_program program, program_life, programCpu;

cl_kernel generate_random, generate_next_matrix;

cl_mem color_buffer, color_copy_buffer, rand_input_buffer, time_input_buffer, matrixBuffer, dbuffer_matrix, bufferRandom, dbuffer_matrix2;

size_t color_data_size = (sizeof(cl_uchar3) * (WIDTH * HEIGHT)), rand_data_size = (sizeof(cl_ulong) * 10), neighbors_data_size = (sizeof(cl_ulong) * 80), time_data_size = sizeof(cl_ulong);

size_t szGlobalWorkSize[1];// Global # of work items
size_t szLocalWorkSize[1];
size_t szParmDataBytes, szParmDataBytes2;
size_t szKernelLength, szKernelLength2;
static clock_t timeeeeee;
const char* cgenerate_random[] =
{
"__kernel void generate_random(",
"__global const int* random,",
"__global int* out) ",
"{",
"     int i = get_global_id(0);",
"    out[i] = random[i];",
"}",
};
const char* cgenerate_next_matrix[] =
{
"__kernel void generate_next_matrix(",
"__global const int* matrix,",
" __global int* bufferMatrix )",
"{",
"    int i = get_global_id(0);",
"    int surround[8];",
"	 const int HEIGHT = 768;" ,
"	 const int WIDTH = 1024;",
"    if ((i%WIDTH) > 0 && (i/WIDTH) > 0) {",
"        surround[0] = matrix[(((i%WIDTH) - 1) + ((i/WIDTH) - 1)*WIDTH)];",
"    }",
"    else surround[0] = 0;",
"    if ((i/WIDTH) > 0)",
"        surround[1] = matrix[((i%WIDTH)+ ((i/WIDTH) - 1)*WIDTH)];",
"    else surround[1] = 0;",
"    if ((i%WIDTH) < WIDTH && (i/WIDTH) > 0)",
"        surround[2] = matrix[(((i%WIDTH) + 1)+ ((i/WIDTH) - 1)*WIDTH)];",
"    else surround[2] = 0;",
"    if ((i%WIDTH) < WIDTH)",
"        surround[3] = matrix[(((i%WIDTH) + 1)+ (i/WIDTH)*WIDTH)];",
"    else surround[3] = 0;",
"    if ((i%WIDTH) < WIDTH && (i/WIDTH) < HEIGHT)",
"        surround[4] = matrix[(((i%WIDTH) + 1)+ ((i/WIDTH) + 1)*WIDTH)];",
"    else surround[4] = 0;",
"    if ((i/WIDTH) < HEIGHT)",
"        surround[5] = matrix[((i%WIDTH)+ ((i/WIDTH) + 1)*WIDTH)];",
"    else surround[5] = 0;",
"    if ((i%WIDTH) > 0 && (i/WIDTH) < HEIGHT)",
"        surround[6] = matrix[(((i%WIDTH) - 1)+ ((i/WIDTH) + 1)*WIDTH)];",
"    else surround[6] = 0;",
"    if ((i%WIDTH) > 0)",
"        surround[7] = matrix[(((i%WIDTH) - 1)+ (i/WIDTH)*WIDTH)];",
"    else surround[7] = 0;",
"    int live[4];",
"    for (int ii = 0; ii < 4; ii++) {",
"        int counter = 0;",
"        for (int i = 0; i < 8; i++) {",
"            if (surround[i] == ii + 1) {",
"                counter++;",
"            }",
"        }",
"        live[ii] = counter;",
"    }",
"    if (matrix[i] != 0) {",
"        if (live[matrix[i] - 1] < 2) {",
"            bufferMatrix[i] = 0;",
"        }",
"        else",
"            if ((live[matrix[i] - 1] == 2) || (live[matrix[i] - 1] == 3)) {",
"                bufferMatrix[i] = matrix[i];",
"            }",
"            else",
"                if (live[matrix[i] - 1] > 3) {",
"                    bufferMatrix[i] = 0;",
"                }",
"    }",
"    else if (matrix[i] == 0) {",
"        int c = 0;",
"        int d = 0;",
"        for (int jj = 0; jj < 4; jj++) {",
"            if (live[jj] == 3) {",
"                c = jj;",
"                d = 1;",
"                break;",
"            }",
"        }",
"        if (d == 1)",
"            bufferMatrix[i] = c + 1;",
"        else",
"            bufferMatrix[i] = 0;",
"    }",
"}",
};
int first = 1;
const int SOURCE_NUM_LINES = sizeof(cgenerate_next_matrix) / sizeof(cgenerate_next_matrix[0]);
const int SOURCE_NUM_LINEScpu = sizeof(cgenerate_random) / sizeof(cgenerate_random[0]);
void initMatrix(int matrix[HEIGHT * WIDTH], int size) {
	for (int i = 0; i < size; i++) {
		int random = (rand() % 5);
		if (random == 0) {
			matrix[i] = 0;
			
		}
		else if (random == 1) {

			matrix[i] = 1;
		}
		else if (random == 2) {

			matrix[i] = 2;
		}
		else if (random == 3) {

			matrix[i] = 3;
		}
		else if (random == 4) {

			matrix[i] = 4;
		}
	}
}

char title[] = "Assignment 1 Simulation";

static cl_int matrix[HEIGHT * WIDTH] = { 0 };
static cl_int bufferMatrix[HEIGHT * WIDTH] = { 0 };


int turn = 0;
// forward declarations
void clean_up();
void display();
void update(int);
void opencl_setup_scratch();
void dooWork(int[HEIGHT * WIDTH], int[HEIGHT * WIDTH]);
void dooWorkcpu();
void render(int[HEIGHT*WIDTH]);

void display()
{
	

	printf("%d", 1000 / (clock() - timeeeeee));
	cout << endl;
	timeeeeee = clock();
	
	
	
	if (turn == 0) {
		dooWork(matrix, bufferMatrix); // tell gpu to start
		render(matrix);
	}
	else {
		dooWork(bufferMatrix, matrix); // tell gpu to start
		render(bufferMatrix);
	}
	
	
	clFinish(queue_gpu);// tell gpu to stop
	clean_up();
	
	if (turn == 0) {
		turn = 1;
	}
	else {
		turn = 0;
	}

}
void render(int matrix[HEIGHT*WIDTH]) {
	glClear(GL_COLOR_BUFFER_BIT);
	glEnable(GL_POINT_SMOOTH);
	glPointSize(20);



	glBegin(GL_POINTS);
	for (cl_uint i = 0; i < HEIGHT * WIDTH; i++) {
		if (matrix[i] == 0) {
			glColor3f(1, 1, 1);


		}
		else if (matrix[i] == 1) {
			glColor3f(1, 0, 0);

		}
		else if (matrix[i] == 2) {
			glColor3f(0, 1, 0);

		}
		else if (matrix[i] == 3) {
			glColor3f(0, 0, 1);

		}
		else if (matrix[i] == 4) {
			glColor3f(0, 0, 0);

		}

		glVertex2i(xxCompute(i) % (WIDTH), yyCompute(i) % (HEIGHT)); // coords

	}
	glEnd();
	glFlush();
}

void clean_up()
{
	
	 clReleaseMemObject(matrixBuffer); 
	 clReleaseMemObject(dbuffer_matrix); 
	 

}
void clean_up_cpu() {
	clReleaseMemObject(bufferRandom);
	clReleaseMemObject(dbuffer_matrix2);
}
void big_clean() {
	free(device_ids);
	clReleaseKernel(generate_next_matrix);
	clReleaseCommandQueue(queue_gpu);
	clReleaseContext(m_context);
}
void opencl_setup_cpu() {
	cl_uint nd;
	err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_CPU, 1, &cdDevices2[0], &nd);
	//err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &cdDevices[1], &nd);
	
	m_contextcpu = clCreateContextFromType(0, CL_DEVICE_TYPE_CPU, NULL, NULL, &err);
	cout << err;
	clGetContextInfo(m_contextcpu, CL_CONTEXT_DEVICES, 0, NULL, &szParmDataBytes2);
	cdDevices2 = (cl_device_id*)malloc(szParmDataBytes2);
	clGetContextInfo(m_contextcpu, CL_CONTEXT_DEVICES, szParmDataBytes2, cdDevices2, NULL);
	queue_cpu = clCreateCommandQueue(m_contextcpu, cdDevices2[0], 0, &err);
	cout << err;
	
	bufferRandom = clCreateBuffer(m_contextcpu, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_int) * HEIGHT * WIDTH, bufferMatrix, &err);
	cout << err;
	dbuffer_matrix2 = clCreateBuffer(m_contextcpu, CL_MEM_WRITE_ONLY, sizeof(cl_int) * HEIGHT * WIDTH,
		NULL, &err);
	cout << err;

	programCpu = clCreateProgramWithSource(m_contextcpu, SOURCE_NUM_LINEScpu, cgenerate_random,
		NULL, &err);
	cout << err;
	err = clBuildProgram(programCpu, 0, NULL, NULL, NULL, NULL);
	cout << err;
	generate_random = clCreateKernel(programCpu, "generate_random", &err);
	cout << err;
	err = clSetKernelArg(generate_random, 0, sizeof(cl_mem), (void*)&bufferRandom);
	err |= clSetKernelArg(generate_random, 1, sizeof(cl_mem), (void*)&dbuffer_matrix2);
	szGlobalWorkSize[0] =	HEIGHT*WIDTH;

	szLocalWorkSize[0] = 1;
	err = clEnqueueNDRangeKernel(queue_cpu, generate_random, 1, NULL, szGlobalWorkSize,
		szLocalWorkSize, 0, NULL, NULL);
	
	err = clEnqueueReadBuffer(queue_cpu, dbuffer_matrix2, CL_TRUE, 0, HEIGHT * WIDTH * sizeof(cl_int),
		matrix, 0, NULL, NULL);
	clFinish(queue_cpu);
	clean_up_cpu();
}

void opencl_setup_scratch() {
	cl_uint nd;

	err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &cdDevices[0], &nd);

	m_context = clCreateContextFromType(0, CL_DEVICE_TYPE_GPU, NULL, NULL, NULL);

	clGetContextInfo(m_context, CL_CONTEXT_DEVICES, 0, NULL, &szParmDataBytes);

	cdDevices = (cl_device_id*)malloc(szParmDataBytes);

	clGetContextInfo(m_context, CL_CONTEXT_DEVICES, szParmDataBytes, cdDevices, NULL);

	queue_gpu = clCreateCommandQueue(m_context, cdDevices[0], 0, NULL);

	program = clCreateProgramWithSource(m_context, SOURCE_NUM_LINES, cgenerate_next_matrix,
		NULL, &err);
	
	cout << err;
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

	cout << err;
	generate_next_matrix = clCreateKernel(program, "generate_next_matrix", &err);
	
	cout << err;

	szGlobalWorkSize[0] = HEIGHT * WIDTH;

	szLocalWorkSize[0] = 1;

	cout << err << "heleloo";


	cout << err;

	matrixBuffer = clCreateBuffer(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_int) * HEIGHT * WIDTH, matrix, NULL);
	dbuffer_matrix = clCreateBuffer(m_context, CL_MEM_WRITE_ONLY, sizeof(cl_int) * HEIGHT * WIDTH,
		NULL, NULL);
	err = clSetKernelArg(generate_next_matrix, 0, sizeof(cl_mem), (void*)&matrixBuffer);

	err |= clSetKernelArg(generate_next_matrix, 1, sizeof(cl_mem), (void*)&dbuffer_matrix);





	err = clEnqueueNDRangeKernel(queue_gpu, generate_next_matrix, 1, NULL, szGlobalWorkSize,
		szLocalWorkSize, 0, NULL, NULL);

	err = clEnqueueReadBuffer(queue_gpu, dbuffer_matrix, CL_TRUE, 0, HEIGHT * WIDTH * sizeof(cl_int),
		bufferMatrix, 0, NULL, NULL);

	clFinish(queue_gpu);
	clean_up();

}

void dooWork(int m2[HEIGHT * WIDTH],int m[HEIGHT*WIDTH]) {
	
	matrixBuffer = clCreateBuffer(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_int) * HEIGHT * WIDTH, m2, NULL);
	dbuffer_matrix = clCreateBuffer(m_context, CL_MEM_WRITE_ONLY, sizeof(cl_int) * HEIGHT * WIDTH,
		NULL, NULL);
	err = clSetKernelArg(generate_next_matrix, 0, sizeof(cl_mem), (void*)&matrixBuffer);

	err |= clSetKernelArg(generate_next_matrix, 1, sizeof(cl_mem), (void*)&dbuffer_matrix);
	err = clEnqueueNDRangeKernel(queue_gpu, generate_next_matrix, 1, NULL, szGlobalWorkSize,
		szLocalWorkSize, 0, NULL, NULL);

	err = clEnqueueReadBuffer(queue_gpu, dbuffer_matrix, CL_TRUE, 0, HEIGHT * WIDTH * sizeof(cl_int),
		m, 0, NULL, NULL);
	
	
	
}

void dooWorkcpu() {

	bufferRandom = clCreateBuffer(m_contextcpu, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_int) * HEIGHT * WIDTH, bufferMatrix, &err);
	cout << err;
	dbuffer_matrix2 = clCreateBuffer(m_contextcpu, CL_MEM_WRITE_ONLY, sizeof(cl_int) * HEIGHT * WIDTH,
		NULL, &err);
	cout << err;
	err = clSetKernelArg(generate_random, 0, sizeof(cl_mem), (void*)&bufferRandom);
	err |= clSetKernelArg(generate_random, 1, sizeof(cl_mem), (void*)&dbuffer_matrix2);
	szGlobalWorkSize[0] = HEIGHT * WIDTH;

	szLocalWorkSize[0] = 1;
	err = clEnqueueNDRangeKernel(queue_cpu, generate_random, 1, NULL, szGlobalWorkSize,
		szLocalWorkSize, 0, NULL, NULL);

	err = clEnqueueReadBuffer(queue_cpu, dbuffer_matrix2, CL_TRUE, 0, HEIGHT * WIDTH * sizeof(cl_int),
		matrix, 0, NULL, NULL);
	clFinish(queue_cpu);
	clean_up_cpu();

}

void update(int value)
{
	glutPostRedisplay();
	glutTimerFunc(1000 / 30, update, 0);
}

int main(int argc, char** argv)
{
	

	
	
		

	initMatrix(bufferMatrix, HEIGHT * WIDTH);
	opencl_setup_cpu();
	opencl_setup_scratch();

	

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE | GLUT_ALPHA);
	glutInitWindowSize(static_cast<int>(WINDOW_WIDTH * 1.5), static_cast<int>(WINDOW_HEIGHT * 1.5));

	glutCreateWindow(title);
	

	glClearColor(0.0, 0.0, 0.0, 0.0);
	glMatrixMode(GL_PROJECTION);
	
	
	gluOrtho2D(-0.5f, ((float)WIDTH - 1) - 0.5f, -0.5f, ((float)HEIGHT - 1) - 0.5f);
	
	

	glutDisplayFunc(display);
	
	glutTimerFunc(1000, update, 0); 

#if defined (__APPLE__) || defined(MACOSX)
	atexit(cleanup);
#else
	glutCloseFunc(big_clean);
#endif

	glutMainLoop();
	
	
	
	
	
	return 0;

}