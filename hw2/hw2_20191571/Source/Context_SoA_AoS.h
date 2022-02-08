//
//  Context_SoA_AoS.h
//
//  Written for CSEG437_CSE5437
//  Department of Computer Science and Engineering
//  Copyright © 2021 Sogang University. All rights reserved.
//

#ifndef __CONTEXT_SOA_AOS__
#define __CONTEXT_SOA_AOS__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

#include <FreeImage/FreeImage.h>

#include "Util/my_OpenCL_util_2_2.h"
#define     QUEUED_TO_END         0
#define     SUBMIT_TO_END          1
#define     START_TO_END            2

typedef struct _OPENCL_C_PROG_SRC {
    size_t length;
    char* string;
} OPENCL_C_PROG_SRC;


typedef struct _Pixel_Planes {
    BYTE *R_plane, *G_plane, *B_plane, *A_plane;
} Pixel_Planes;

typedef struct _Context {
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
    Pixel_Planes SoA_image_input, SoA_image_output;


    //cl
    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_context context;
    cl_command_queue cmd_queue;
    cl_program program;
    cl_kernel kernel;
    OPENCL_C_PROG_SRC prog_src;
    cl_mem BO_input, BO_output;
    cl_mem sobel_x, sobel_y;
    cl_event event_for_timing;

    cl_uint work_dim;
    size_t global_work_offset[3], global_work_size[3], local_work_size[3];

} Context;

extern Context context;

static const float Sobel_x[25]={
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


////////////////////// Image_IO.cpp /////////////////////////////////////////
void read_input_image_from_file32(const char* filename);
void prepare_output_image(void);
void write_output_image_to_file32(const char* filename);
void prepare_SoA_input_and_output(void);
void convert_SoA_output_to_output_image_data(void);
///////////// main_SoA_AoS.cpp ///////////////////////////////////
int initialize_OpenCL(void);
int set_local_work_size_and_kernel_arguments(void);
int run_OpenCL_kernel(void);
void clean_up_system(void);


#endif // __CONTEXT_SOA_AOS__