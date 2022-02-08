#pragma once
//
//  Context_WE.h
//
//  Written for CSEG437_CSE5437
//  Department of Computer Science and Engineering
//  Copyright © 2020 Sogang University. All rights reserved.
//
// EXPLICIT NOT WORKING: GRID_RESOLUTION = 128 / SIMULATION_PARAMETERS = 1 

#ifndef __OPENCL_STUFFS_H__
#define __OPENCL_STUFFS_H__

#include <GL/glew.h>
#include <GL/freeglut.h>
//#include <FreeImage/FreeImage.h>

#include <CL/cl.h>
#include <CL/cl_gl.h>

typedef struct _OPENCL_C_PROG_SRC {
    size_t length;
    char* string;
} OPENCL_C_PROG_SRC;

typedef struct _OCL_STUFFS {
    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_context context;
    cl_command_queue cmd_queue;
    cl_kernel kernel;
    OPENCL_C_PROG_SRC prog_src;
    cl_program program;
    cl_mem grid_data_OCL_BO[3];
    cl_mem grid_data_tmp_OCL_BO;

    cl_uint work_dim;
    size_t global_work_size[3], local_work_size[3];
} OCL_STUFFS;

typedef struct _OGL_STUFFS {
    GLuint grid_data_OGL_BO[3], grid_index_OGL_BO; 
    GLuint VAO[3];

    int n_grid_indices;
    GLuint *grid_indices;
} OGL_STUFFS;

typedef struct _WAVE_EQUATION {
    float side_length, h, init_speed, alpha_square, time_interval, sigma;
    int n_intervals, n_nodes_1d, display_interval, Jacobian_iterations;

    int x_mid, y_mid;
    float amplitude;
    float beta;

    float* grid_data[3];
    float* grid_data_tmp;
    int gridMaxIdx = 0;

    int next_buffer;
    int simulation_steps;
} WAVE_EQUATION;

typedef struct _CONTEXT_WE {
    unsigned int window_width, window_height;

    OCL_STUFFS ocl_stuffs;
    WAVE_EQUATION wave_equation;
    OGL_STUFFS ogl_stuffs;
} CONTEXT_WE;

int run_CL_KERNEL_WE_IMPLICIT (CONTEXT_WE* cont);
int run_CL_KERNEL_WE_EXPLICIT(CONTEXT_WE* cont);
int run_CPU_Jacobi_WE(CONTEXT_WE* cont);
int run_CPU_GaussSeidel_WE(CONTEXT_WE* cont);
void initialize_OpenGL(CONTEXT_WE* cont);

#endif // __OPENCL_STUFFS_H__

 
 