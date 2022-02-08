#pragma warning(disable : 4996)

//
//  main_ConcurrentCopyCompute.cpp
//
//  Written for CSEG437_CSE5437
//  Department of Computer Science and Engineering
//  Copyright © 2021 Sogang University. All rights reserved.
//

#include<stdio.h>
#include<stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <string.h>
#include <time.h>

#include <CL/cl.h>

#include "Util/my_OpenCL_util_2_2.h"
#include "config_ConcurrentCopyCompute.h"
#include "context_ConcurrentCopyCompute.h"

Context context;

cl_int errcode_ret;
__int64 _start, _freq, _end;
float compute_time, total_time;
char tmp_string[256];

/////////////////////////////////////////////////////////////////////////////////////////////////////////
///// main /////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////
int main(void) {

    read_input_image_from_file32(INPUT_FILE_NAME);
    prepare_output_image();
    prepare_AoS_input_and_output();

    intialize_context();
     /* Get the first platform. */
    errcode_ret = clGetPlatformIDs(1, &context.platform_id, NULL);
    // You may skip error checking if you think it is unnecessary.
    if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);

    /* Get the first GPU device. */
    errcode_ret = clGetDeviceIDs(context.platform_id, CL_DEVICE_TYPE_GPU, 1, &context.device_id, NULL);
    if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);


    fprintf(stdout, "/////////////////////////////////////////////////////////////////////////\n");
    fprintf(stdout, "### INPUT FILE NAME = \t\t%s\n", INPUT_FILE_NAME);
    fprintf(stdout, "### OUTPUT FILE NAME = \t\t%s\n\n", OUTPUT_FILE_NAME);



    /* Assume the first device of the first plaform is a GPU. */

    fprintf(stdout, "\n^^^ The first GPU device on the platform ^^^\n");
    print_device_0(context.device_id);

    /* Create a context with the devices. */
    context.context = clCreateContext(NULL, 1, &context.device_id, NULL, NULL, &errcode_ret);
    if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);

    /* Create a command-queue for the GPU device. */
   // Use clCreateCommandQueueWithProperties() for OpenCL 2.0.
    for (int i = 0; i < MAXIMUM_COMMAND_QUEUES; i++) {
        context.cmd_queue[i] = clCreateCommandQueue(context.context, context.device_id, CL_QUEUE_PROFILING_ENABLE, &errcode_ret);
        if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);
    }


    /* Create input and output buffer objects on device. */
    context.buffer_input_dev = clCreateBuffer(context.context, CL_MEM_READ_ONLY,
        sizeof(BYTE) * context.image_data_bytes, NULL, &errcode_ret);//rgba
    if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);
    context.buffer_output_dev = clCreateBuffer(context.context, CL_MEM_WRITE_ONLY,
        sizeof(BYTE) * context.image_data_bytes, NULL, &errcode_ret);//rgba
    if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);

    context.sobel_x = clCreateBuffer(context.context, CL_MEM_READ_ONLY,
        sizeof(float) * 25, NULL, &errcode_ret);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;
    context.sobel_y = clCreateBuffer(context.context, CL_MEM_READ_ONLY,
        sizeof(float) * 25, NULL, &errcode_ret);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    fprintf(stdout, "\n^^^ Four buffer objects on device are created. ^^^\n");


    /* Create input and output buffer objects on host pinned memory. */
    /* Necessary for asynchronous data copy between host and device. */
    context.buffer_input_pinned = clCreateBuffer(context.context, CL_MEM_READ_WRITE |
        CL_MEM_ALLOC_HOST_PTR, sizeof(BYTE) * context.image_data_bytes, NULL, &errcode_ret);
    if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);

    context.buffer_output_pinned = clCreateBuffer(context.context, CL_MEM_READ_WRITE |
        CL_MEM_ALLOC_HOST_PTR, sizeof(BYTE) * context.image_data_bytes, NULL, &errcode_ret);
    if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);
    fprintf(stdout, "\n^^^ Two buffer objects on host pinned memory are created. ^^^\n");

    /* Get mapped (standard) pointers to buffer objects on host pinned memory. */
    context.data_input = (cl_uchar*)clEnqueueMapBuffer(context.cmd_queue[0], context.buffer_input_pinned, CL_TRUE,
        CL_MAP_WRITE, 0, sizeof(BYTE) * context.image_data_bytes, 0, NULL, NULL, &errcode_ret);
    if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);

    context.data_output = (cl_uchar*)clEnqueueMapBuffer(context.cmd_queue[0], context.buffer_output_pinned, CL_TRUE,
        CL_MAP_READ, 0, sizeof(BYTE) * context.image_data_bytes, 0, NULL, NULL, &errcode_ret);
    if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);
    fprintf(stdout, "\n^^^ Two standard pointers to host pinned memory are mapped. ^^^\n");

	for (int i = 0; i < context.image_height * context.image_width; i++) {
		context.data_input[i * 4] = context.AoS_image_input[i].R;
		context.data_input[i * 4 + 1] = context.AoS_image_input[i].G;
		context.data_input[i * 4 + 2] = context.AoS_image_input[i].B;
		context.data_input[i * 4 + 3] = context.AoS_image_input[i].A;
	}


    fprintf(stdout, "\n^^^ generate end... ^^^\n\n");

    /* Create a program from OpenCL C source code. */
    context.prog_src.length = read_kernel_from_file(OPENCL_C_PROG_FILE_NAME, &context.prog_src.string);
    context.program = clCreateProgramWithSource(context.context, 1, (const char**)&context.prog_src.string,
        &context.prog_src.length, &errcode_ret);
    if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);
    fprintf(stdout, "^^^ OpenCL C program file name = %s ^^^\n\n", OPENCL_C_PROG_FILE_NAME);

    /* Build a program executable from the program object. */
    const char options[] = "-cl-std=CL1.2";
    errcode_ret = clBuildProgram(context.program, 1, &context.device_id, options, NULL, NULL);
    if (errcode_ret != CL_SUCCESS) {
        print_build_log(context.program, context.device_id, "GPU");
        exit(EXIT_FAILURE);
    }

    /* Create the kernel from the program. */
    for (int i = 0; i < MAXIMUM_COMMAND_QUEUES; i++) {
        context.kernel[i] = clCreateKernel(context.program, KERNEL_NAME, &errcode_ret);
        if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);
    }
    fprintf(stdout, "^^^ Kernel name = %s ^^^\n\n", KERNEL_NAME);

    printf_KernelWorkGroupInfo(context.kernel[0], context.device_id);
 
    /* Warm the device up before starting profiling. */
    fprintf(stdout, "\n^^^ Just for warming the device up.\n");
    use_multiple_segments_and_three_command_queues_with_events_breadth(1);
    use_multiple_segments_and_three_command_queues_with_events_breadth(1);

    for (int i = 0; i < context.image_data_bytes; i++) {
        context.solution[i] = context.data_output[i];
    }
    fprintf(stdout, "\n");
    switch (context.copy_compute_type) {
    case COPY_COMPUTE_TYPE_THREE_QUEUES_WITH_EVENTS:
        fprintf(stdout, "^^^ Copy compute type: THREE QUEUES (EVENTS) ^^^\n\n");
        for (int j = 1; j <= MAXIMUM_COMMAND_QUEUES; j <<= 1) {
            // j : n_segments
            fprintf(stdout, "\n^^^ Use three queues and events for %d-segmented kernel execution.\n", j);
            use_multiple_segments_and_three_command_queues_with_events_breadth(j);
            check_correctness_on_host();
        }
        break;
    case COPY_COMPUTE_TYPE_MULTIPLE_QUEUES:
        fprintf(stdout, "^^^ Copy compute type: MULTIPLE QUEUES ^^^\n\n");
        for (int j = 1; j <= MAXIMUM_COMMAND_QUEUES; j <<= 1) {
            // j : n_segments
            fprintf(stdout, "\n^^^ Use multiple queues for %d-segmented kernel execution.\n", j);
            use_multiple_segments_and_multiple_command_queues_depth(j);
            check_correctness_on_host();
        }
        break;
    }

    /* Unmap the mapped regions of the three buffer objects. */
    errcode_ret = clEnqueueUnmapMemObject(context.cmd_queue[0], context.buffer_input_pinned, context.data_input, 
        0, NULL, NULL);
    if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);
    errcode_ret = clEnqueueUnmapMemObject(context.cmd_queue[0], context.buffer_output_pinned, context.data_output,
        0, NULL, NULL);
    if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);

    for (int i = 0; i < context.image_width*context.image_height;i++) {
        context.AoS_image_output[i].R = context.data_output[4*i];
        context.AoS_image_output[i].G = context.data_output[4*i+1];
        context.AoS_image_output[i].B = context.data_output[4*i+2];
        context.AoS_image_output[i].A = context.data_output[4*i+3];
    }
    convert_AoS_output_to_output_image_data();
    write_output_image_to_file32(OUTPUT_FILE_NAME);

    /* Free resources. */
    free( context.solution);
    free(context.prog_src.string);

    for (int i = 0; i < MAXIMUM_COMMAND_QUEUES; i++) {
        clReleaseEvent(context.event_write_input[i]);
        clReleaseEvent(context.event_compute[i]);
        clReleaseEvent(context.event_read_output[i]);
    }
    clReleaseMemObject(context.buffer_input_dev);
    clReleaseMemObject(context.buffer_output_dev);
    clReleaseMemObject(context.buffer_input_pinned);
    clReleaseMemObject(context.buffer_output_pinned);
    clReleaseMemObject(context.sobel_x);
    clReleaseMemObject(context.sobel_y);
    for (int i = 0; i < MAXIMUM_COMMAND_QUEUES; i++)  
        clReleaseKernel(context.kernel[i]);
    clReleaseProgram(context.program);
    for (int i = 0; i < MAXIMUM_COMMAND_QUEUES; i++)  
        clReleaseCommandQueue(context.cmd_queue[i]);
    clReleaseDevice(context.device_id);
    clReleaseContext(context.context);
    free(context.AoS_image_input);
    free(context.AoS_image_output);
    return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void intialize_context(void) {

    context.global_work_size[0] = context.image_width;
    context.global_work_size[1] = context.image_height;

    context.local_work_size[0] = LOCAL_WORK_SIZE_0;
    context.local_work_size[1] = LOCAL_WORK_SIZE_1;

    context.n_kernel_loop_iterations = N_KERNEL_LOOP_ITERATIONS;
    context.n_kernel_call_iterations = N_KERNEL_CALL_ITERATIONS;

    context.solution= (BYTE*)malloc(sizeof(BYTE)*context.image_data_bytes);
    if (!context.solution) {
        fprintf(stderr, "*** Error: cannot allocate memory for solution array on host....\n");
        exit(EXIT_FAILURE);
    }

    context.copy_compute_type = COPY_COMPUTE_TYPE;

    fprintf(stdout, "/***********************************************************************/\n");
    fprintf(stdout, "   - GLOBAL WORK SIZE = (%u, %u)\n", (unsigned int)context.global_work_size[0], (unsigned int)context.global_work_size[1]);
    fprintf(stdout, "   - LOCAL WORK SIZE = (%u, %u)\n", (unsigned int)context.local_work_size[0], (unsigned int)context.local_work_size[1]);
    fprintf(stdout, "   - KERNEL = %s in %s\n", KERNEL_NAME, OPENCL_C_PROG_FILE_NAME);
    fprintf(stdout, "   - NUMBER OF KERNEL LOOP ITERATIONS = %d\n", context.n_kernel_loop_iterations);
    fprintf(stdout, "   - NUMBER OF KERNEL CALL ITERATIONS = %d\n", context.n_kernel_call_iterations);
    fprintf(stdout, "   - NUMBER OF MAXIMUM COMMAND QUEUES = %d\n", MAXIMUM_COMMAND_QUEUES);
    fprintf(stdout, "   - COPY COMPUTE TYPE = %s\n",
        (context.copy_compute_type == 0) ? "THREE QUEUES WITH EVENTS" : "MULTIPLE QUEUES");
    fprintf(stdout, "/***********************************************************************/\n");
}

void initialize_buffers_for_kernel_execution(void) {
    const BYTE zero_b = 0;
    errcode_ret = clEnqueueFillBuffer(context.cmd_queue[0], context.buffer_input_dev, (void*)&zero_b, sizeof(zero_b), 0,
        context.image_data_bytes * sizeof(BYTE), 0, NULL, NULL);
    if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);
    errcode_ret = clEnqueueFillBuffer(context.cmd_queue[0], context.buffer_output_dev, (void*)&zero_b, sizeof(zero_b), 0,
        context.image_data_bytes * sizeof(BYTE), 0, NULL, NULL);
    if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);
    memset(context.data_output, 0, context.image_data_bytes * sizeof(BYTE));   // use for-loop for sure.
}


void check_correctness_on_host(void) {
    util_compair_two_byte_arrays(context.solution, context.data_output, context.image_data_bytes, 0);
}

void print_time_statistics_for_copy_and_compute(int n_segments) {
    cl_ulong submit_time, base_timestamp;

    for (int i = 0; i < n_segments; i++) {
        errcode_ret = clGetEventProfilingInfo(context.event_write_input[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong),
            &submit_time, NULL);
        if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);
        if (i == 0) {
            base_timestamp = submit_time;
        }
        else {
            if (base_timestamp > submit_time) base_timestamp = submit_time;
        }


        errcode_ret = clGetEventProfilingInfo(context.event_compute[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong),
            &submit_time, NULL);
        if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);
        if (base_timestamp > submit_time) base_timestamp = submit_time;

        errcode_ret = clGetEventProfilingInfo(context.event_read_output[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong),
            &submit_time, NULL);
        if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);
        if (base_timestamp > submit_time) base_timestamp = submit_time;
    }
    ///////////////////////////////////////////////////////////////////////////////////
    cl_ulong start_time, end_time;

    for (int i = 0; i < n_segments; i++) {
        errcode_ret = clGetEventProfilingInfo(context.event_write_input[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong),
            &submit_time, NULL);
        if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);
        errcode_ret = clGetEventProfilingInfo(context.event_write_input[i], CL_PROFILING_COMMAND_START, sizeof(cl_ulong),
            &start_time, NULL);
        if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);
        errcode_ret = clGetEventProfilingInfo(context.event_write_input[i], CL_PROFILING_COMMAND_END, sizeof(cl_ulong),
            &end_time, NULL);
        if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);

        fprintf(stdout, "         (A) Data transfer from host to device (%2d) = [", i);
        util_insert_commas_in_timestamp(start_time - base_timestamp, tmp_string);
        fprintf(stdout, "%12s ", tmp_string);
        util_insert_commas_in_timestamp(submit_time - base_timestamp, tmp_string);
        fprintf(stdout, "(%12s), ", tmp_string);
        util_insert_commas_in_timestamp(end_time - base_timestamp, tmp_string);
        fprintf(stdout, "%12s ", tmp_string);
        fprintf(stdout, "(%.3fms)]\n", (float) (end_time - start_time) * 1.0e-6f);
    }
    fprintf(stdout, "\n");
    for (int i = 0; i < n_segments; i++) {
        errcode_ret = clGetEventProfilingInfo(context.event_compute[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong),
            &submit_time, NULL);
        if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);
        errcode_ret = clGetEventProfilingInfo(context.event_compute[i], CL_PROFILING_COMMAND_START, sizeof(cl_ulong),
            &start_time, NULL);
        if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);
        errcode_ret = clGetEventProfilingInfo(context.event_compute[i], CL_PROFILING_COMMAND_END, sizeof(cl_ulong),
            &end_time, NULL);
        if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);

        fprintf(stdout, "         (B) Compute on device (%2d)                 = [", i);
        util_insert_commas_in_timestamp(start_time - base_timestamp, tmp_string);
        fprintf(stdout, "%12s ", tmp_string);
        util_insert_commas_in_timestamp(submit_time - base_timestamp, tmp_string);
        fprintf(stdout, "(%12s), ", tmp_string);
        util_insert_commas_in_timestamp(end_time - base_timestamp, tmp_string);
        fprintf(stdout, "%12s ", tmp_string);
        fprintf(stdout, "(%.3fms)]\n", (end_time - start_time) * 1.0e-6f);
    }
    fprintf(stdout, "\n");
    for (int i = 0; i < n_segments; i++) {
        errcode_ret = clGetEventProfilingInfo(context.event_read_output[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong),
            &submit_time, NULL);
        if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);
        errcode_ret = clGetEventProfilingInfo(context.event_read_output[i], CL_PROFILING_COMMAND_START, sizeof(cl_ulong),
            &start_time, NULL);
        if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);
        errcode_ret = clGetEventProfilingInfo(context.event_read_output[i], CL_PROFILING_COMMAND_END, sizeof(cl_ulong),
            &end_time, NULL);
        if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);

        fprintf(stdout, "         (C) Data transfer from device to host (%2d) = [", i);
        util_insert_commas_in_timestamp(start_time - base_timestamp, tmp_string);
        fprintf(stdout, "%12s ", tmp_string);
        util_insert_commas_in_timestamp(submit_time - base_timestamp, tmp_string);
        fprintf(stdout, "(%12s), ", tmp_string);
        util_insert_commas_in_timestamp(end_time - base_timestamp, tmp_string);
        fprintf(stdout, "%12s ", tmp_string);
        fprintf(stdout, "(%.3fms)]\n", (end_time - start_time) * 1.0e-6f);
    }
    fprintf(stdout, "\n");
}

#include "ThreeQueueingMethods.h"
