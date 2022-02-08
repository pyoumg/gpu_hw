#pragma once

//
//  context_ConcurrentCopyCompute.h
//
//  Written for CSEG437_CSE5437
//  Department of Computer Science and Engineering
//  Copyright © 2021 Sogang University. All rights reserved.
//
#include <FreeImage/FreeImage.h>
#include "Util/my_OpenCL_util_2_2.h"
#include "config_ConcurrentCopyCompute.h"

typedef struct _OPENCL_C_PROG_SRC {
    size_t length;
    char* string;
} OPENCL_C_PROG_SRC;
//AoS
typedef struct _Pixel_Channels {
    BYTE R, G, B, A;
} Pixel_Channels;

typedef struct _My_Context {
    FREE_IMAGE_FORMAT image_format;
    unsigned int image_width, image_height, image_pitch;
    size_t image_data_bytes;

    struct {
        FIBITMAP* fi_bitmap_32;
        BYTE* image_data;
    } input;
    struct {
        FIBITMAP* fi_bitmap_32;
        BYTE* image_data;
    } output;

    Pixel_Channels* AoS_image_input, * AoS_image_output;

    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_context context;
    cl_command_queue cmd_queue[MAXIMUM_COMMAND_QUEUES];
    cl_kernel kernel[MAXIMUM_COMMAND_QUEUES];
    OPENCL_C_PROG_SRC prog_src;
    cl_program program;

    cl_mem buffer_input_dev, buffer_output_dev;
    cl_mem buffer_input_pinned, buffer_output_pinned;

    cl_event event_write_input[MAXIMUM_COMMAND_QUEUES];
    cl_event event_compute[MAXIMUM_COMMAND_QUEUES];
    cl_event event_read_output[MAXIMUM_COMMAND_QUEUES];

    cl_mem sobel_x, sobel_y;

    cl_uint work_dim;
    size_t global_work_offset[3], global_work_size[3], local_work_size[3];

    BYTE* data_input, * data_output; //host memory pointer
    int n_kernel_loop_iterations;
    int n_kernel_call_iterations;
    BYTE* solution;
    int copy_compute_type;
} Context;
extern Context context;

static const float Sobel_x[25] = {
    -0.0560561 ,-0.0448449 ,0.0000000 ,0.0448449 ,0.0560561 ,
-0.0896897 ,-0.1121121 ,0.0000000 ,0.1121121 ,0.0896897 ,
-0.1121121 ,-0.2242243 ,0.0000000 ,0.2242243 ,0.1121121 ,
-0.0896897 ,-0.1121121 ,0.0000000 ,0.1121121 ,0.0896897 ,
-0.0560561 ,-0.0448449 ,0.0000000 ,0.0448449 ,0.0560561 ,
};
static const float Sobel_y[25] = {
    -0.0560561 ,-0.0896897 ,-0.1121121 ,-0.0896897 ,-0.0560561 ,
-0.0448449 ,-0.1121121 ,-0.2242243 ,-0.1121121 ,-0.0448449 ,
0.0000000 ,0.0000000 ,0.0000000 ,0.0000000 ,0.0000000 ,
0.0448449 ,0.1121121 ,0.2242243 ,0.1121121 ,0.0448449 ,
0.0560561 ,0.0896897 ,0.1121121 ,0.0896897 ,0.0560561 ,
};

void intialize_context(void);
void initialize_buffers_for_kernel_execution(void);
void compute_solution_on_segment_1(void);
void check_correctness_on_host(void);
void use_multiple_segments_and_three_command_queues_with_events_breadth(int);
void use_multiple_segments_and_multiple_command_queues_depth(int);
void print_time_statistics_for_copy_and_compute(int);
////////////////////// Image_IO.cpp /////////////////////////////////////////

void read_input_image_from_file32(const char* filename);
void prepare_output_image(void);
void write_output_image_to_file32(const char* filename);
void prepare_AoS_input_and_output(void);
void convert_AoS_output_to_output_image_data(void);