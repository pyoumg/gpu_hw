#pragma warning(disable : 4996)
//
//  main_WaveEquation_CL_CPU.cpp
//
//  Written for CSEG437_CSE5437
//  Department of Computer Science and Engineering
//  Copyright ©2021 Sogang University. All rights reserved.
//

#include<stdio.h>
#include<stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include <CL/cl.h>
#include <CL/cl_gl.h>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include "Util/my_OpenCL_util_2_2.h"
#include "Context_WE.h"
#include "Config_WE.h"
FILE* fp, * fp2;
void clean_up_OpenCL(CONTEXT_WE* cont) { 
    OCL_STUFFS* ocl_stuffs = &cont->ocl_stuffs;

    if (ocl_stuffs->prog_src.string) free(ocl_stuffs->prog_src.string);
    for (int i = 0; i < 3; i++) 
       if (ocl_stuffs->grid_data_OCL_BO[i])  
            clReleaseMemObject(ocl_stuffs->grid_data_OCL_BO[i]); 
    if (ocl_stuffs->kernel) clReleaseKernel(ocl_stuffs->kernel);
    if (ocl_stuffs->program) clReleaseProgram(ocl_stuffs->program);
    if (ocl_stuffs->cmd_queue) clReleaseCommandQueue(ocl_stuffs->cmd_queue);
    if (ocl_stuffs->device_id) clReleaseDevice(ocl_stuffs->device_id);
    if (ocl_stuffs->context) clReleaseContext(ocl_stuffs->context);
}

int run_CL_KERNEL_WE_IMPLICIT(CONTEXT_WE* cont) {
    cl_int errcode_ret;
    OCL_STUFFS* ocl_stuffs = &cont->ocl_stuffs;

    glFinish();

    errcode_ret = clEnqueueAcquireGLObjects(ocl_stuffs->cmd_queue, 3, 
        ocl_stuffs->grid_data_OCL_BO, 0, NULL, NULL);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;   

    int* buffer_next = &cont->wave_equation.next_buffer; 
    *buffer_next = (*buffer_next + 1) % 3;

    errcode_ret = clSetKernelArg(ocl_stuffs->kernel, 0, sizeof(cl_mem), &ocl_stuffs->grid_data_OCL_BO[*buffer_next]);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;
    errcode_ret = clSetKernelArg(ocl_stuffs->kernel, 1, sizeof(cl_mem), &ocl_stuffs->grid_data_OCL_BO[(*buffer_next + 2) % 3]);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;
    errcode_ret = clSetKernelArg(ocl_stuffs->kernel, 2, sizeof(cl_mem), &ocl_stuffs->grid_data_OCL_BO[(*buffer_next + 1) % 3]);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    for (int iter = 0; iter < cont->wave_equation.Jacobian_iterations; iter++) {  
        // The current implementaton requires Jacobian_iterations to be an even number.
        errcode_ret = clSetKernelArg(ocl_stuffs->kernel, 4, sizeof(int), &iter);
        if (CHECK_ERROR_CODE(errcode_ret)) return 1;

        errcode_ret = clEnqueueNDRangeKernel(ocl_stuffs->cmd_queue, ocl_stuffs->kernel, 2, NULL,
            ocl_stuffs->global_work_size, ocl_stuffs->local_work_size, 0, NULL, NULL);
        if (CHECK_ERROR_CODE(errcode_ret)) return 1;
        clFinish(ocl_stuffs->cmd_queue);
    }
    errcode_ret = clEnqueueReleaseGLObjects(ocl_stuffs->cmd_queue, 3, 
        ocl_stuffs->grid_data_OCL_BO, 0, NULL, NULL);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    clFinish(ocl_stuffs->cmd_queue);
    return 0;
}

int run_CL_KERNEL_WE_EXPLICIT(CONTEXT_WE* cont) {
    cl_int errcode_ret;
    OCL_STUFFS* ocl_stuffs = &cont->ocl_stuffs;

    glFinish();
    errcode_ret = clEnqueueAcquireGLObjects(ocl_stuffs->cmd_queue, 3, ocl_stuffs->grid_data_OCL_BO, 0, NULL, NULL);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    int* buffer_next = &cont->wave_equation.next_buffer;
    *buffer_next = (*buffer_next + 1) % 3;

    errcode_ret = clSetKernelArg(ocl_stuffs->kernel, 0, sizeof(cl_mem), &ocl_stuffs->grid_data_OCL_BO[*buffer_next]);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;
    errcode_ret = clSetKernelArg(ocl_stuffs->kernel, 1, sizeof(cl_mem), &ocl_stuffs->grid_data_OCL_BO[(*buffer_next + 2) % 3]);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;
    errcode_ret = clSetKernelArg(ocl_stuffs->kernel, 2, sizeof(cl_mem), &ocl_stuffs->grid_data_OCL_BO[(*buffer_next + 1) % 3]);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    errcode_ret = clEnqueueNDRangeKernel(ocl_stuffs->cmd_queue, ocl_stuffs->kernel, 2, NULL,
        ocl_stuffs->global_work_size, ocl_stuffs->local_work_size, 0, NULL, NULL);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;
    clFinish(ocl_stuffs->cmd_queue);

    errcode_ret = clEnqueueReleaseGLObjects(ocl_stuffs->cmd_queue, 3, 
        ocl_stuffs->grid_data_OCL_BO, 0, NULL, NULL);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    clFinish(ocl_stuffs->cmd_queue);
    return 0;
}

#define Y(index)  (4*(index) + 1)
#define K_MINUS_1	0
#define K				1
#define CPU_JACOBI_DEBUG    0
#define CPU_GaussSeidel_DEBUG    0

void Jacobi_WE(CONTEXT_WE* cont) {
    WAVE_EQUATION* we = &cont->wave_equation;
    int n_nodes_1d = we->n_nodes_1d;

    float* grid_next = we->grid_data[we->next_buffer];
    float* grid_cur = we->grid_data[(we->next_buffer + 2) % 3];
    float* grid_prev = we->grid_data[(we->next_buffer + 1) % 3];
    float* grid_tmp = we->grid_data_tmp;
    int index;
#if CPU_JACOBI_DEBUG == 1
    fp2 = fopen("jc.txt", "a");
    fprintf(fp2, "[Start]   %g\n", grid_cur[Y(64 * n_nodes_1d + 64)]);
#endif
    float diagonal = 1.0f + 4.0f * we->beta;
    for (int iter = 0; iter < we->Jacobian_iterations; iter++) { // Jacobian iterations
        for (int j = 1; j < n_nodes_1d - 1; j++) {
            for (int i = 1; i < n_nodes_1d - 1; i++) {
               // HW3
                index = j * n_nodes_1d + i;
               
                float* x[2];  // x(K_MINUS_1) --> x(K)
                if (iter == 0) {
                    x[K_MINUS_1] = grid_cur; x[K] = grid_tmp;
            }
                else if ((iter % 2) == 1) { // Jacobian_iter > 0 and odd
                    x[K_MINUS_1] = grid_tmp; x[K] = grid_next;
                }
                else { // Jacobian_iter > 0 and even
                    x[K_MINUS_1] = grid_next; x[K] = grid_tmp;
                }

                float b = 2 * grid_cur[Y(index)] - grid_prev[Y(index)];
                float sum = x[K_MINUS_1][Y(index - n_nodes_1d)] + x[K_MINUS_1][Y(index - 1)]
                    + x[K_MINUS_1][Y(index + 1)]+ x[K_MINUS_1][Y(index + n_nodes_1d)];

                x[K][Y(index)] = (b + we->beta * sum) / diagonal;

#if CPU_JACOBI_DEBUG == 1
                 if ((i==64) && (j==64))  fprintf(fp2, "[After iter = %2d]   %g\n", iter,  x[K][Y(index)]);
#endif
            }
        }
    }
#if CPU_JACOBI_DEBUG == 1
  fprintf(fp2, "[End]     %g\n\n", grid_next[Y(64 * n_nodes_1d + 64)]);
  fclose(fp2);
#endif
}

int run_CPU_Jacobi_WE(CONTEXT_WE* cont) {
    cont->wave_equation.next_buffer = (cont->wave_equation.next_buffer + 1) % 3;
    Jacobi_WE(cont);

    return 0;
}

void Jacobi_GS(CONTEXT_WE* cont) {
    WAVE_EQUATION* we = &cont->wave_equation;
    int n_nodes_1d = we->n_nodes_1d;

    float* grid_next = we->grid_data[we->next_buffer];
    float* grid_cur = we->grid_data[(we->next_buffer + 2) % 3];
    float* grid_prev = we->grid_data[(we->next_buffer + 1) % 3];
    float* grid_tmp = we->grid_data_tmp;
    int index;
    float ax_k, ax_k_minus_1;
#if CPU_GaussSeidel_DEBUG == 1
    fp = fopen("gs.txt", "a");
    fprintf(fp, "[Start]   %g\n", grid_cur[Y(64 * n_nodes_1d + 64)]);
#endif
    float diagonal = 1.0f + 4.0f * we->beta;
    for (int iter = 0; iter < we->Jacobian_iterations; iter++) { // Jacobian iterations
        for (int j = 1; j < n_nodes_1d - 1; j++) {
            ax_k = 0, ax_k_minus_1 = 1;
            for (int i = 1; i < n_nodes_1d - 1; i++) {//not edge
                // HW3
              
                index = j * n_nodes_1d + i;

                float* x[2];  // x(K_MINUS_1) --> x(K)

                if (iter == 0) {
                    x[K_MINUS_1] = grid_cur; x[K] = grid_tmp;
            }
                else if ((iter % 2) == 1) { // Jacobian_iter > 0 and odd
                    x[K_MINUS_1] = grid_tmp; x[K] = grid_next;
                }
                else { // Jacobian_iter > 0 and even
                    x[K_MINUS_1] = grid_next; x[K] = grid_tmp;
                }

                float b = 2 * grid_cur[Y(index)] - grid_prev[Y(index)];
                float sum = x[K][Y(index - n_nodes_1d)] + x[K][Y(index - 1)]
                    + x[K_MINUS_1][Y(index + 1)] + x[K_MINUS_1][Y(index + n_nodes_1d)];

                x[K][Y(index)] = (b + we->beta * sum) / diagonal;

#if CPU_GaussSeidel_DEBUG == 1
                if ((i == 64) && (j == 64))  fprintf(fp, "[After iter = %2d]   %g\n", iter, x[K][Y(index)]);
#endif
            }
        }
    }
#if CPU_GaussSeidel_DEBUG == 1
    fprintf(fp, "[End]     %g\n\n", grid_next[Y(64 * n_nodes_1d + 64)]);
    fclose(fp);
#endif
}

int run_CPU_GaussSeidel_WE(CONTEXT_WE* cont) {
    cont->wave_equation.next_buffer = (cont->wave_equation.next_buffer + 1) % 3;
    Jacobi_GS(cont);

    return 0;
}

int initialize_OpenCL(CONTEXT_WE* cont) {
    static const char* CL_KHR_GL_SHARING_EXTENSION = "cl_khr_gl_sharing";
    static const char* CL_KHR_GL_EVENT_EXTENSION = "cl_khr_gl_event";
    cl_int errcode_ret;

    OCL_STUFFS* ocl_stuffs = &cont->ocl_stuffs;
    OGL_STUFFS* ogl_stuffs = &cont->ogl_stuffs;
    WAVE_EQUATION* wave_equation = &cont->wave_equation;

    /* Get the first platform. */
    errcode_ret = clGetPlatformIDs(1, &ocl_stuffs->platform_id, NULL);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;  
    // You may skip error checking if you think it is unnecessary.

    /* Get the first GPU device. */
    errcode_ret = clGetDeviceIDs(ocl_stuffs->platform_id, CL_DEVICE_TYPE_GPU, 
        1, &ocl_stuffs->device_id, NULL);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    /* Get string that contains supported device extensions. */
    size_t extension_list_length;
    errcode_ret = clGetDeviceInfo(ocl_stuffs->device_id, CL_DEVICE_EXTENSIONS, 0, NULL, &extension_list_length);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;
 
    char* extension_list = (char*)malloc(extension_list_length);
    errcode_ret = clGetDeviceInfo(ocl_stuffs->device_id, CL_DEVICE_EXTENSIONS, extension_list_length, 
        extension_list, NULL);  
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    if (is_OpenCL_Extension_supported(CL_KHR_GL_SHARING_EXTENSION, extension_list)) {
        fprintf(stdout, "*** The extension %s is supported by the current OpenCL implementation.\n",
            CL_KHR_GL_SHARING_EXTENSION);
    }
    else {
        fprintf(stdout, "*** The extension %s is not supported by the current OpenCL implementation.\n",
            CL_KHR_GL_SHARING_EXTENSION);
        return 1; // Cannot proceed any longer...
    }
 
    if (is_OpenCL_Extension_supported(CL_KHR_GL_EVENT_EXTENSION, extension_list)) {
        fprintf(stdout, "*** The extension %s is supported by the current OpenCL implementation.\n",
            CL_KHR_GL_EVENT_EXTENSION);
    }
    else {
        fprintf(stdout, "*** The extension %s is not supported by the current OpenCL implementation.\n",
            CL_KHR_GL_EVENT_EXTENSION);
        // Currently, this extension is not used.
    }
 
    /* Create CL context properties, add WGL context & handle to DC. */
    cl_context_properties properties[] = {
        CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(), // WGL Context
        CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(), // WGL HDC
        CL_CONTEXT_PLATFORM, (cl_context_properties)ocl_stuffs->platform_id, // OpenCL platform
        0
    };

    /* Create a context with the devices. */
    ocl_stuffs->context = clCreateContext(properties, 1, &ocl_stuffs->device_id, NULL, NULL, &errcode_ret);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    /* Create a command-queue for the GPU device. */
    // Use clCreateCommandQueueWithProperties() for OpenCL 2.0.
    ocl_stuffs->cmd_queue = clCreateCommandQueue(ocl_stuffs->context, ocl_stuffs->device_id, 
        CL_QUEUE_PROFILING_ENABLE, &errcode_ret);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    /* Create a program from OpenCL C source code. */
    ocl_stuffs->prog_src.length = read_kernel_from_file("./Source/Kernel/wave_equation.cl", &ocl_stuffs->prog_src.string);

    ocl_stuffs->program = clCreateProgramWithSource(ocl_stuffs->context, 1, (const char**)&ocl_stuffs->prog_src.string, 
        &ocl_stuffs->prog_src.length, &errcode_ret);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;
 
    /* Build a program executable from the program object. */
    errcode_ret = clBuildProgram(ocl_stuffs->program, 1, &ocl_stuffs->device_id, "-cl-std=CL1.2", NULL, NULL);
    if (errcode_ret != CL_SUCCESS) {
        print_build_log(ocl_stuffs->program, ocl_stuffs->device_id, "GPU");
        free(ocl_stuffs->prog_src.string);
        return false;
    }
   
    /* Create the kernel from the program. */
#ifdef USE_GPU_EXPLICIT
    ocl_stuffs->kernel = clCreateKernel(ocl_stuffs->program, "wave_equation_EXPLICIT", &errcode_ret);
#elif defined USE_GPU_IMPLICIT_JACOBI
    ocl_stuffs->kernel = clCreateKernel(ocl_stuffs->program, "wave_equation_IMPLICIT", &errcode_ret);
#else
    fprintf(stderr, "^^^ Error: which method should we use for a simulation of the wave equation?");
    exit(-1);
#endif
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    for (int i = 0; i < 3; i++) {
        ocl_stuffs->grid_data_OCL_BO[i] = clCreateFromGLBuffer(ocl_stuffs->context, CL_MEM_READ_WRITE,
            ogl_stuffs->grid_data_OGL_BO[i], &errcode_ret);
        if (CHECK_ERROR_CODE(errcode_ret)) return 1;
    }

#ifdef USE_GPU_IMPLICIT_JACOBI
    ocl_stuffs->grid_data_tmp_OCL_BO = clCreateBuffer(ocl_stuffs->context, CL_MEM_READ_WRITE,
        4 * sizeof(float) * wave_equation->n_nodes_1d * wave_equation->n_nodes_1d, NULL, &errcode_ret);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1; 
    clEnqueueWriteBuffer(ocl_stuffs->cmd_queue, ocl_stuffs->grid_data_tmp_OCL_BO, CL_TRUE, 0,
        4 * sizeof(float) * wave_equation->n_nodes_1d * wave_equation->n_nodes_1d,
        wave_equation->grid_data_tmp, 0, NULL, NULL);
#endif

    ocl_stuffs->global_work_size[0] = GRID_RESOLUTION;
    ocl_stuffs->global_work_size[1] = GRID_RESOLUTION;

    ocl_stuffs->local_work_size[0] = LOCAL_WORK_SIZE_0;
    ocl_stuffs->local_work_size[1] = LOCAL_WORK_SIZE_1;

    errcode_ret = clSetKernelArg(ocl_stuffs->kernel, 3, sizeof(float), 
        &wave_equation->beta);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

#ifdef USE_GPU_IMPLICIT_JACOBI
    errcode_ret = clSetKernelArg(ocl_stuffs->kernel, 5, sizeof(cl_mem), 
        &ocl_stuffs->grid_data_tmp_OCL_BO);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;
#endif

    printf_KernelWorkGroupInfo(ocl_stuffs->kernel, ocl_stuffs->device_id);
    return 0;
}

void set_up_simulation_parameters(WAVE_EQUATION* we) {
    we->simulation_steps = NUMBER_OF_SIMULATION_STEPS;
    float scale_factor = GRID_RESOLUTION / 128.0;
    we->side_length = SIDE_LENGTH;

    we->n_intervals = GRID_RESOLUTION - 1;
    we->n_nodes_1d = GRID_RESOLUTION;

#if SIMULATION_PARAMETERS == 0
    we->alpha_square = 5.0;
    we->time_interval = 0.1 / scale_factor;
    we->sigma = 5.5;

    we->display_interval = 5; // 5 good
    we->Jacobian_iterations = NUMBER_OF_JACOBIAN_ITERATIONS;

    we->h = we->side_length / we->n_nodes_1d;
    we->beta = we->alpha_square * we->time_interval * we->time_interval / (we->h * we->h);

    we->x_mid = 2.5 * we->n_nodes_1d / 4;
    we->y_mid = 1.5 * we->n_nodes_1d / 5;
    we->amplitude = 20.0f;
    we->init_speed = 400.0;
#elif SIMULATION_PARAMETERS == 1
    we->alpha_square = 5.0;
    we->time_interval = 0.005 / scale_factor;
    we->sigma = 5.5;

    we->display_interval = 5; // 5 good
    we->Jacobian_iterations = NUMBER_OF_JACOBIAN_ITERATIONS;

    we->h = we->side_length / we->n_nodes_1d;
    we->beta = we->alpha_square * we->time_interval * we->time_interval / (we->h * we->h);

    we->x_mid = 2.5 * we->n_nodes_1d / 4;
    we->y_mid = 1.5 * we->n_nodes_1d / 5;
    we->amplitude = 20.0f;
    we->init_speed = 400.0;
#elif SIMULATION_PARAMETERS == 2
    we->alpha_square = 5.0;
    we->time_interval = 0.00075 / scale_factor;
    we->sigma = 5.5;

    we->display_interval = 5; // 5 good
    we->Jacobian_iterations = NUMBER_OF_JACOBIAN_ITERATIONS;

    we->h = we->side_length / we->n_nodes_1d;
    we->beta = we->alpha_square * we->time_interval * we->time_interval / (we->h * we->h);

    we->x_mid = 2.5 * we->n_nodes_1d / 4;
    we->y_mid = 1.5 * we->n_nodes_1d / 5;
    we->amplitude = 20.0f;
    we->init_speed = 400.0;
#endif

    we->next_buffer = 0;
}

void initialize_context_we(CONTEXT_WE* cont) {
    cont->window_width = WINDOW_WIDTH;
    cont->window_height = WINDOW_HEIGHT;
    set_up_simulation_parameters(&cont->wave_equation);
}

void initialize_OpenGL(CONTEXT_WE* cont);
void clean_up_OpenGL(CONTEXT_WE* cont);

int main(int argc, char* argv[]) {
    CONTEXT_WE context_we;

    char program_name[] = "hw4_20191571";
    fprintf(stdout, "\n###  %s\n\n", program_name);
    fprintf(stdout, "/////////////////////////////////////////////////////////////////////////\n");
    fprintf(stdout, "### GRID RESOLUTION = \t%d\n", GRID_RESOLUTION);
    fprintf(stdout, "### WORK-GROUP SIZE = \t(%d, %d)\n", LOCAL_WORK_SIZE_0, 
        LOCAL_WORK_SIZE_1);
    fprintf(stdout, "/////////////////////////////////////////////////////////////////////////\n\n");

    initialize_context_we(&context_we);

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA);
    glutInitWindowSize(context_we.window_width, context_we.window_height);
    glutInitContextVersion(4, 0);
    glutInitContextProfile(GLUT_CORE_PROFILE);
    glutCreateWindow(program_name);

    initialize_OpenGL(&context_we);

#if (defined USE_GPU_EXPLICIT) || (defined USE_GPU_IMPLICIT_JACOBI)
    if (initialize_OpenCL(&context_we)) {
        clean_up_OpenCL(&context_we);
        clean_up_OpenGL(&context_we);
        exit(-1);
    }
#elif defined (USE_CPU_IMPLICIT_JACOBI) || defined (USE_CPU_IMPLICIT_GAUSS_SEIDEL)
#endif

    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
    glutMainLoop();
}


 