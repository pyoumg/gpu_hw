//
//  main_SoA_AoS.cpp
//
//  Written for CSEG437_CSE5437
//  Department of Computer Science and Engineering
//  Copyright © 2021 Sogang University. All rights reserved.
//
#pragma warning(disable: 4996)
#pragma warning(disable: 6386)

#include "Context_SoA_AoS.h"
#include "Config_SoA_AoS.h"

Context context;
cl_int errcode_ret;

FILE* fp_stat;
char tmp_string[512];
__int64 _start, _freq, _end;
float compute_time;

int main(int argc, char* argv[]) {
    int flag;
    char program_name[] = "Sogang CSEG475_5475 SoA_20191571";
    fprintf(stdout, "\n###  %s\n\n", program_name);
    fprintf(stdout, "/////////////////////////////////////////////////////////////////////////\n");
    fprintf(stdout, "### INPUT FILE NAME = \t\t%s\n", INPUT_FILE_NAME);
    fprintf(stdout, "### OUTPUT FILE NAME = \t\t%s\n\n", OUTPUT_FILE_NAME);

    fprintf(stdout, "### IMAGE OPERATION = ");
    switch (IMAGE_OPERATION) {
    case SoA_NAIVE:
        fprintf(stdout, "Structure of Arrays (SOBEL) on GPU (Kernel-Naive)\n");
        break;
    case SoA_OPTIMIZED:
        fprintf(stdout, "Structure of Arrays (SOBEL) on GPU (Kernel-Optimzied)\n");
        break;
   
    default:
        fprintf(stderr, "*** Error: unknown image operation...\n");
        exit(EXIT_FAILURE);
    }
    fprintf(stdout, "/////////////////////////////////////////////////////////////////////////\n\n");

    read_input_image_from_file32(INPUT_FILE_NAME);   
    prepare_output_image();

    if (IMAGE_OPERATION == SoA_NAIVE || IMAGE_OPERATION == SoA_OPTIMIZED) {
        prepare_SoA_input_and_output();

        flag = initialize_OpenCL();
        if (flag) goto finish1;
        flag = set_local_work_size_and_kernel_arguments();
        if (flag) goto finish1;
        flag = run_OpenCL_kernel();
        if (flag) goto finish1;
        convert_SoA_output_to_output_image_data();
        write_output_image_to_file32(OUTPUT_FILE_NAME);

        free(context.SoA_image_input.R_plane);
        free(context.SoA_image_input.G_plane);
        free(context.SoA_image_input.B_plane);
        free(context.SoA_image_input.A_plane);
        free(context.SoA_image_output.R_plane);
        free(context.SoA_image_output.G_plane);
        free(context.SoA_image_output.B_plane);
        free(context.SoA_image_output.A_plane);

    finish1:
        clean_up_system();
        return flag;
    }
    else {
        fprintf(stdout, "^^^ Nothing has been done!\n");
    }
    return 0;
}

void clean_up_system(void) {
    // Free OpenCL and other resources. 
    if (context.prog_src.string) free(context.prog_src.string);
    if (context.BO_input) clReleaseMemObject(context.BO_input);
    if (context.BO_input) clReleaseMemObject(context.BO_output);
    if (context.sobel_x) clReleaseMemObject(context.sobel_x);
    if (context.sobel_y) clReleaseMemObject(context.sobel_y);
    if (context.kernel) clReleaseKernel(context.kernel);
    if (context.program) clReleaseProgram(context.program);
    if (context.cmd_queue) clReleaseCommandQueue(context.cmd_queue);
    if (context.device_id) clReleaseDevice(context.device_id);
    if (context.context) clReleaseContext(context.context);
    if (context.event_for_timing) clReleaseEvent(context.event_for_timing);

};

int initialize_OpenCL(void) {
    /* Get the first platform. */
    errcode_ret = clGetPlatformIDs(1, &context.platform_id, NULL);
    // You may skip error checking if you think it is unnecessary.
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    /* Get the first GPU device. */
    errcode_ret = clGetDeviceIDs(context.platform_id, CL_DEVICE_TYPE_GPU, 1, &context.device_id, NULL);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    fprintf(stdout, "\n^^^ The first GPU device on the platform ^^^\n");
    print_device_0(context.device_id);

    /* Create a context with the devices. */
    context.context = clCreateContext(NULL, 1, &context.device_id, NULL, NULL, &errcode_ret);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    /* Create a command-queue for the GPU device. */
    // Use clCreateCommandQueueWithProperties() for OpenCL 2.0.
    context.cmd_queue = clCreateCommandQueue(context.context, context.device_id,
        CL_QUEUE_PROFILING_ENABLE, &errcode_ret);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    /* Create a program from OpenCL C source code. */
    context.prog_src.length = read_kernel_from_file(OPENCL_C_PROG_FILE_NAME,
        &context.prog_src.string);
    context.program = clCreateProgramWithSource(context.context, 1,
        (const char**)&context.prog_src.string, &context.prog_src.length, &errcode_ret);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    /* Build a program executable from the program object. */
    const char options[] = "-cl-std=CL1.2";
    errcode_ret = clBuildProgram(context.program, 1, &context.device_id, options, NULL, NULL);
    if (errcode_ret != CL_SUCCESS) {
        print_build_log(context.program, context.device_id, "GPU");
        return 1;
    }

    /* Create the kernel from the program. */
    context.kernel = clCreateKernel(context.program, KERNEL_NAME, &errcode_ret);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    /* Create input and output buffer objects. */
    context.BO_input = clCreateBuffer(context.context, CL_MEM_READ_ONLY,
        sizeof(BYTE) * context.image_data_bytes, NULL, &errcode_ret);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    context.BO_output = clCreateBuffer(context.context, CL_MEM_WRITE_ONLY,
        sizeof(BYTE) * context.image_data_bytes, NULL, &errcode_ret);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    context.sobel_x = clCreateBuffer(context.context, CL_MEM_READ_ONLY,
        sizeof(float) * 25, NULL, &errcode_ret);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    context.sobel_y = clCreateBuffer(context.context, CL_MEM_READ_ONLY,
        sizeof(float) * 25, NULL, &errcode_ret);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    fprintf(stdout, "    [Data Transfer to GPU] \n");

    CHECK_TIME_START(_start, _freq);
    // Move the input data from the host memory to the GPU device memory.
    UINT size = context.image_width * context.image_height;
    errcode_ret = clEnqueueWriteBuffer(context.cmd_queue, context.BO_input, CL_FALSE, 0,
        sizeof(BYTE) * size, context.SoA_image_input.R_plane, 0, NULL, NULL);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;
    errcode_ret = clEnqueueWriteBuffer(context.cmd_queue, context.BO_input, CL_FALSE, size,
        sizeof(BYTE) * size, context.SoA_image_input.G_plane, 0, NULL, NULL);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;
    errcode_ret = clEnqueueWriteBuffer(context.cmd_queue, context.BO_input, CL_FALSE, (size_t)2 * size,
        sizeof(BYTE) * size, context.SoA_image_input.B_plane, 0, NULL, NULL);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;
    errcode_ret = clEnqueueWriteBuffer(context.cmd_queue, context.BO_input, CL_FALSE, (size_t)3 * size,
        sizeof(BYTE) * size, context.SoA_image_input.A_plane, 0, NULL, NULL);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;
    errcode_ret = clEnqueueWriteBuffer(context.cmd_queue, context.sobel_x, CL_FALSE, 0,
        sizeof(float) * 25,
        Sobel_x, 0, NULL, NULL);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;
    errcode_ret = clEnqueueWriteBuffer(context.cmd_queue, context.sobel_y, CL_FALSE, 0,
        sizeof(float) * 25,
        Sobel_y, 0, NULL, NULL);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    /* Wait until all data transfers finish. */
    clFinish(context.cmd_queue);
    CHECK_TIME_END(_start, _end, _freq, compute_time);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    fprintf(stdout, "      * Time by host clock = %.3fms\n\n", compute_time);
    return 0;
}

int set_local_work_size_and_kernel_arguments(void) {
    context.global_work_size[0] = context.image_width;
    context.global_work_size[1] = context.image_height;

    context.local_work_size[0] = LOCAL_WORK_SIZE_0;
    context.local_work_size[1] = LOCAL_WORK_SIZE_1;

    /* Set the kenel arguments. */

    errcode_ret = clSetKernelArg(context.kernel, 0, sizeof(cl_mem), &context.BO_input);
    errcode_ret |= clSetKernelArg(context.kernel, 1, sizeof(cl_mem), &context.BO_output);
    errcode_ret |= clSetKernelArg(context.kernel, 2, sizeof(UINT), &context.image_width);
    errcode_ret |= clSetKernelArg(context.kernel, 3, sizeof(UINT), &context.image_height);
    errcode_ret |= clSetKernelArg(context.kernel, 4, sizeof(cl_mem), &context.sobel_x);
    errcode_ret |= clSetKernelArg(context.kernel, 5, sizeof(cl_mem), &context.sobel_y);
#if IMAGE_OPERATION == 1   //local memory
    int twice_half_filter_width = 4;
    size_t local_mem_size = sizeof(cl_uchar)
        * (context.local_work_size[0] + twice_half_filter_width)
        * (context.local_work_size[1] + twice_half_filter_width);
    errcode_ret |= clSetKernelArg(context.kernel, 6, local_mem_size, NULL);
    fprintf(stdout, "^^^ Necessary local memory = %d bytes (%d, %d, %d) ^^^\n\n", local_mem_size,
        sizeof(cl_uchar), context.local_work_size[0] + twice_half_filter_width,
        context.local_work_size[0] + twice_half_filter_width);
#endif
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    printf_KernelWorkGroupInfo(context.kernel, context.device_id);

    return 0;
}

int run_OpenCL_kernel(void) {
    fprintf(stdout, "    [Kernel Execution] \n");

    fp_stat = util_open_stat_file_append(STAT_FILE_NAME);
    util_stamp_stat_file_device_name_and_time(fp_stat, context.device_id);
    util_reset_event_time();

    CHECK_TIME_START(_start, _freq);
    /* Execute the kernel on the device. */
    for (int i = 0; i < N_EXECUTIONS; i++) {
        errcode_ret = clEnqueueNDRangeKernel(context.cmd_queue, context.kernel, 2, NULL,
            context.global_work_size, context.local_work_size, 0, NULL, &context.event_for_timing);
        if (CHECK_ERROR_CODE(errcode_ret)) return 1;
        clWaitForEvents(1, &context.event_for_timing);
        if (CHECK_ERROR_CODE(errcode_ret)) return 1;
        util_accumulate_event_times_1_2(context.event_for_timing);
    }
    CHECK_TIME_END(_start, _end, _freq, compute_time);

    fprintf(stdout, "      * Time by host clock = %.3fms\n\n", compute_time);
    util_print_accumulated_device_time_1_2(N_EXECUTIONS);
    MAKE_STAT_ITEM_LIST_CBO(tmp_string, context.global_work_size, context.local_work_size);
    util_stamp_stat_file_ave_device_time_START_to_END_1_2_string(fp_stat, tmp_string);
    util_close_stat_file_append(fp_stat);

    fprintf(stdout, "    [Data Transfer] \n");

    /* Read back the device buffer to the host array. */
    CHECK_TIME_START(_start, _freq);
    UINT size = context.image_width * context.image_height;
    errcode_ret = clEnqueueReadBuffer(context.cmd_queue, context.BO_output, CL_TRUE, 0,
        sizeof(BYTE) * size, context.SoA_image_output.R_plane, 0, NULL,
        &context.event_for_timing);
    errcode_ret = clEnqueueReadBuffer(context.cmd_queue, context.BO_output, CL_TRUE, (size_t)size,
        sizeof(BYTE) * size, context.SoA_image_output.G_plane, 0, NULL,
        &context.event_for_timing);
    errcode_ret = clEnqueueReadBuffer(context.cmd_queue, context.BO_output, CL_TRUE, (size_t)2 * size,
        sizeof(BYTE) * size, context.SoA_image_output.B_plane, 0, NULL,
        &context.event_for_timing);
    errcode_ret = clEnqueueReadBuffer(context.cmd_queue, context.BO_output, CL_TRUE, (size_t)3 * size,
        sizeof(BYTE) * size, context.SoA_image_output.A_plane, 0, NULL,
        &context.event_for_timing);
    CHECK_TIME_END(_start, _end, _freq, compute_time);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    fprintf(stdout, "      * Time by host clock = %.3fms\n\n", compute_time);
    print_device_time(context.event_for_timing);

    return 0;
}