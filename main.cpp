// -----Overview-----
// This was a project focused on achieving maximum concurrency and parallelization while developing and scheduling the execution of GPU and CPU kernels using OpenCL. The program 
// executes a multi-species variation of Conway's Game of Life and renders it to the screen using OpenGL. This document is modified, with specific dependencies and imports removed
// to avoid cluttering the file for whoever is going to evaluate it.

// -----Multi-Species Conway's game of life rules-----
// The rules of the original game of life are as follows: If the cell is alive, then it stays alive if it has either 2 or 3 live neighbors. 
// If the cell is dead, then it springs to life only in the case that it has 3 live neighbors. Cells in this case are represented by pixels. 
// The only key difference here is that instead of cells representing simply a binary value, alive or dead, 
// while they are alive they are also part of a specific species of alive cells. So now, for a cell to stay alive, it needs 2 or 3 alive neighbors of the same species, 
// and for a dead cell to come back to life, it needs 3 live neighbors of the same species, and obviously will come alive as that species. 
// Each of the different species are visually represented by different colors.

// -----Program Structure Overview-----
// All logic regarding the translation of a current screen state to a future one given the rules of the game were encapsulated completely within a GPU kernel.
// The state of the 2D screen was represented by a one dimensional array, meaning that the computations within the kernal required some,
// albeit minimal, arithmetic gymnastics. The array had accompanying it a buffer array of the same size. During every iteration one array was being used to compute the next screen,
// while the other acted as a buffer storing the computed future state of the screen. The one being used to compute the future screen was used by OpenGL to render to the screen. 
// The two arrays will alternate acting as a buffer and as the representation of the current state of the screen. The purpose of this is to ensure maximum concurrency, so that while 
// the future state of the matrix is being computed, openGl can render the current state to the screen in parallel.
// All scheduling was done using simple logic within the main program file.


// 2D arrays were causing significant issues within the GPU kernel, and that is why I decided to go with 1D arrays instead. To make the conversion between
// 1D and 2D representations simple, I created these functions that extract from a 1D representation of a 2D point both its x and y coordinates. This is because OpenGL requires that I use 
// 2D points when defining what should be rendered to the screen.

// Extract x coordinate
int compute_x(int i)
{
	return i % WIDTH;
}
// Extract y coordinate
int compute_y(int i)
{
	return i / WIDTH;
}


// Where to store gpu device ids
cl_device_id *cd_devices_gpu;


// for returning errors
cl_int err; 

// declare context for gpu
cl_context m_context_gpu;

// two command queues, one for cpu, one for gpu
cl_command_queue queue_gpu; 

// declare program for gpu
cl_program program_gpu;

// declare kernel function
cl_kernel generate_next_matrix;

// declaring all the buffers I'm going to use as input and output for my cpu and gpu kernels. 
cl_mem buffer_matrix_1_gpu, buffer_matrix_2_gpu;

// Global size of work, basically how many total work-items need to be done
size_t global_work_size[1]; 
// Local size of work, basically how many work-items will be in a single work group
size_t local_work_size[1];
// Size in Bytes of the device id information
size_t mem_size_device_ids_gpu;


// This is responsible for computing the main game of life logic that translates a matrix state into a future one.
const char *cgenerate_next_matrix[] =
	{
		"__kernel void generate_next_matrix(", 
        // Takes in two matrixes, one being the current state of the matrix, used to compute the next state, which is the output
		"__global const int* matrix,", 
        // and the other as a buffer to put the output into
		" __global int* bufferMatrix )", 
		"{",
        // Get work-item id. Because work-items map directly to pixels, basically 
        // "which pixel am I working on right now?"
		"    int i = get_global_id(0);",
        // Declare array to store information about surrounding 8 pixels
		"    int surround[8];",
        // Initialize Height and Width constants to allow us to do the tedious arithmetic gymnastics I spoke about earlier
		"	 const int HEIGHT = 768;",
		"	 const int WIDTH = 1024;",
        // Below I set the values in surround to the values surrounding the current pixel
        // starting with the surrounding pixel above and to the left of the pixel, and then going clockwise
        // If there is no surrounding pixel in one of the positions, possibly the pixel is on the edge, I just
        // set that value to 0, which is equivalent to a dead pixel
		"    if ((i%WIDTH) > 0 && (i/WIDTH) > 0) {",
		"        surround[0] = matrix[(((i%WIDTH) - 1) + ((i/WIDTH) - 1)*WIDTH)];", //
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
        // initalize an array meant to store how many of each of the 4 species is alive around the current pixel
		"    int live[4];",
        // iterate through species
		"    for (int ii = 0; ii < 4; ii++) {",
		"        int counter = 0;",
        // iterate through surrounding pixels
		"        for (int i = 0; i < 8; i++) {",
        // if this surrounding pixel is alive and part of the species represented by ii+1, increment counter
		"            if (surround[i] == ii + 1) {",
		"                counter++;",
		"            }",
		"        }",
        // place the amount of live species represented by ii+1 into live[ii]
		"        live[ii] = counter;",
		"    }",
        // if this pixel is not dead:
        // and there are less than 2 live pixels of the same species
        // surrounding it, kill this pixel. 
        // if there are either 2 or 3 live pixels of the same species, keep it alive
        // if there are more than 3 live pixels of the same species surrounding it, also kill the pixel
        // I place the new value of the pixel into bufferMatrix
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
        // If this pixel is dead:
        // if there are exactly 3 of the same live species surrounding the pixel, set this pixel to that species
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

const int SOURCE_NUM_LINES = sizeof(cgenerate_next_matrix) / sizeof(cgenerate_next_matrix[0]);

// This function initalizes the matrix with random values between 0 and 4. 
void initMatrix(int matrix[HEIGHT * WIDTH], int size)
{
	for (int i = 0; i < size; i++)
	{
		int random = (rand() % 5);
		if (random == 0)
		{
			matrix[i] = 0;
		}
		else if (random == 1)
		{

			matrix[i] = 1;
		}
		else if (random == 2)
		{

			matrix[i] = 2;
		}
		else if (random == 3)
		{

			matrix[i] = 3;
		}
		else if (random == 4)
		{

			matrix[i] = 4;
		}
	}
}

char title[] = "OpenCL Game of Life";

// Initializing my two matrices. Neither one is totally a buffer matrix, as they take turns acting as a buffer.
static cl_int matrix_1[HEIGHT * WIDTH] = {0};
static cl_int matrix_2[HEIGHT * WIDTH] = {0};

int turn = 0;
// declaring my functions
void clean_up();
void display();
void update(int);
void opencl_setup_gpu();
void doWork(int[HEIGHT * WIDTH], int[HEIGHT * WIDTH]);
void render(int[HEIGHT * WIDTH]);

// This is my display function that is going to run on a loop. Every iteration, matrix_1 and 
// matrix_2 take turns acting as the input matrix, while the other acts as a buffer matrix
// recieving the output. The logic for this is implented below. Simply, I use a boolean integer
// to keep track of which matrix did what in the last iteration, and just make sure the opposite 
// is done in this iteration.
void display()
{

	if (turn == 0)
	{
		doWork(matrix_1, matrix_2); // tell gpu to start
		render(matrix_1);
	}
	else
	{
		doWork(matrix_2, matrix_1); // tell gpu to start
		render(matrix_2);
	}

	clFinish(queue_gpu); // tell gpu to stop
	clean_up();

	if (turn == 0)
	{
		turn = 1;
	}
	else
	{
		turn = 0;
	}
}
// This function renders to the screen based on what values exist in the current matrix
// If the pixel has value 0, render white
// if the pixel has value 1, render red
// if the pixel has value 2, render green
// if the pixel has value 3, render blue
// if the pixel has value 4, render black
void render(int matrix[HEIGHT * WIDTH])
{
	glClear(GL_COLOR_BUFFER_BIT);
	glEnable(GL_POINT_SMOOTH);
	glPointSize(20);

	glBegin(GL_POINTS);
	for (cl_uint i = 0; i < HEIGHT * WIDTH; i++)
	{
		if (matrix[i] == 0)
		{
			glColor3f(1, 1, 1);
		}
		else if (matrix[i] == 1)
		{
			glColor3f(1, 0, 0);
		}
		else if (matrix[i] == 2)
		{
			glColor3f(0, 1, 0);
		}
		else if (matrix[i] == 3)
		{
			glColor3f(0, 0, 1);
		}
		else if (matrix[i] == 4)
		{
			glColor3f(0, 0, 0);
		}
        // convert 1d coordinates into 2d coordinates
		glVertex2i(compute_x(i) % (WIDTH), compute_y(i) % (HEIGHT)); // coords
	}
	glEnd();
	glFlush();
}

void clean_up()
{

	clReleaseMemObject(buffer_matrix_1_gpu);
	clReleaseMemObject(buffer_matrix_2_gpu);
}

void big_clean()
{
	free(cd_devices_gpu);
    free(cd_devices_cpu)
	clReleaseKernel(generate_next_matrix);
	clReleaseCommandQueue(queue_gpu);
	clReleaseContext(m_context_gpu);
}

// Here I setup everything necessary for the gpu to run the kernel 
// this will be called only once
void opencl_setup_gpu()
{
	cl_uint nd;
    // This gets all device ids of type gpu and places their ids in cd_devices_gpu
	err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &cd_devices_gpu[0], &nd);
    // This creates a context
	m_context_gpu = clCreateContextFromType(0, CL_DEVICE_TYPE_GPU, NULL, NULL, NULL);
    // This gets the size of the deviceid information in bytes, and places it in mem_size_device_ids_gpu
	clGetContextInfo(m_context_gpu, CL_CONTEXT_DEVICES, 0, NULL, &mem_size_device_ids_gpu);
    // This allocates the necessary amount of memory for cd_devices_gpu based on what we found in the previous line
	cd_devices_gpu = (cl_device_id *)malloc(mem_size_device_ids_gpu);
    // this now places the device ids into cd_devices_gpu
	clGetContextInfo(m_context_gpu, CL_CONTEXT_DEVICES, mem_size_device_ids_gpu, cd_devices_gpu, NULL);
    // Create a command queue for the gpu
	queue_gpu = clCreateCommandQueue(m_context_gpu, cd_devices_gpu[0], 0, NULL);
    // Creates program based on kernel code written above
	program_gpu = clCreateProgramWithSource(m_context_gpu, SOURCE_NUM_LINES, cgenerate_next_matrix,
										NULL, &err);
    // builds program
	err = clBuildProgram(program_gpu, 0, NULL, NULL, NULL, NULL);
	// creates a kernel based on a function named 'generate_next_matrix' present in the program
	generate_next_matrix = clCreateKernel(program_gpu, "generate_next_matrix", &err);
    // set global work size to total number of pixels
	global_work_size[0] = HEIGHT * WIDTH;
    // set local work size to 1
	local_work_size[0] = 1;
    // create buffer based on matrix_1
	buffer_matrix_1_gpu = clCreateBuffer(m_context_gpu, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
								  sizeof(cl_int) * HEIGHT * WIDTH, matrix_1, NULL);
    // create empty buffer
	buffer_matrix_2_gpu = clCreateBuffer(m_context_gpu, CL_MEM_WRITE_ONLY, sizeof(cl_int) * HEIGHT * WIDTH,
									NULL, NULL);
    // set arguments for the kernel: the two buffers we just created
	err = clSetKernelArg(generate_next_matrix, 0, sizeof(cl_mem), (void *)&buffer_matrix_1_gpu);
	err |= clSetKernelArg(generate_next_matrix, 1, sizeof(cl_mem), (void *)&buffer_matrix_2_gpu);
    // push kernel execution to the queue
	err = clEnqueueNDRangeKernel(queue_gpu, generate_next_matrix, 1, NULL, global_work_size,
								 local_work_size, 0, NULL, NULL);
    // push reading the output buffer to the queue, placing it into matrix_2
	err = clEnqueueReadBuffer(queue_gpu, buffer_matrix_2_gpu, CL_TRUE, 0, HEIGHT * WIDTH * sizeof(cl_int),
							  matrix_2, 0, NULL, NULL);

	clFinish(queue_gpu);
	clean_up();
}
// This is the function that will execute once every iteration to compute the next screen.
void doWork(int m2[HEIGHT * WIDTH], int m[HEIGHT * WIDTH])
{
    // Create buffer based on matrix m2
	buffer_matrix_1_gpu = clCreateBuffer(m_context_gpu, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
								  sizeof(cl_int) * HEIGHT * WIDTH, m2, NULL);
    // Create empty buffer
	buffer_matrix_2_gpu = clCreateBuffer(m_context_gpu, CL_MEM_WRITE_ONLY, sizeof(cl_int) * HEIGHT * WIDTH,
									NULL, NULL);
    // Set kernel arguments
	err = clSetKernelArg(generate_next_matrix, 0, sizeof(cl_mem), (void *)&buffer_matrix_1_gpu);
	err |= clSetKernelArg(generate_next_matrix, 1, sizeof(cl_mem), (void *)&buffer_matrix_2_gpu);
    // push kernel execution to the queue
	err = clEnqueueNDRangeKernel(queue_gpu, generate_next_matrix, 1, NULL, global_work_size,
								 local_work_size, 0, NULL, NULL);
    // push reading the output buffer to the queue, placing it into m
	err = clEnqueueReadBuffer(queue_gpu, buffer_matrix_2_gpu, CL_TRUE, 0, HEIGHT * WIDTH * sizeof(cl_int),
							  m, 0, NULL, NULL);
}


void update(int value)
{
	glutPostRedisplay();
	glutTimerFunc(1000 / 30, update, 0);
}

int main(int argc, char **argv)
{
    // initialize matrix
	initMatrix(matrix_2, HEIGHT * WIDTH);
	opencl_setup_gpu();

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE | GLUT_ALPHA);
	glutInitWindowSize(static_cast<int>(WINDOW_WIDTH * 1.5), static_cast<int>(WINDOW_HEIGHT * 1.5));

	glutCreateWindow(title);

	glClearColor(0.0, 0.0, 0.0, 0.0);
	glMatrixMode(GL_PROJECTION);

	gluOrtho2D(-0.5f, ((float)WIDTH - 1) - 0.5f, -0.5f, ((float)HEIGHT - 1) - 0.5f);

    // Set display function to display(). Display() will run every iteration
	glutDisplayFunc(display);

	glutTimerFunc(1000, update, 0);

#if defined(__APPLE__) || defined(MACOSX)
	atexit(cleanup);
#else
	glutCloseFunc(big_clean);
#endif

	glutMainLoop();

	return 0;
}
