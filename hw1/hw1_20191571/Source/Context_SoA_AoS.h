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
#include <CL/cl.h>
#include <CL/cl_gl.h>
#include <FreeImage/FreeImage.h>

#include "Util/my_OpenCL_util_2_2.h"

#define     QUEUED_TO_END         0
#define     SUBMIT_TO_END          1
#define     START_TO_END            2

typedef struct _OPENCL_C_PROG_SRC {
	size_t length;
	char* string;
} OPENCL_C_PROG_SRC;

typedef struct _Pixel_Channels {
	BYTE R, G, B, A;
} Pixel_Channels;

typedef struct _Pixel_Planes {
	BYTE* R_plane, * G_plane, * B_plane, * A_plane;
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

	Pixel_Channels* AoS_image_input, * AoS_image_output;
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
	cl_mem sobel_x,sobel_y;
	cl_event event_for_timing;

	cl_uint work_dim;
	size_t global_work_offset[3], global_work_size[3], local_work_size[3];

} Context;

extern Context context;

static const float Sobel_x[9] = {
	0.1767767,0.0,-0.1767767,
	0.3535534,0.0,-0.3535534,
	0.1767767,0.0,-0.1767767
};
static const float Sobel_y[9] = {
	0.1767767,0.3535534,0.1767767,
	0.0,0.0,0.0,
	-0.1767767,-0.3535534,-0.1767767
};
/*
static const float Sobel_x[9] = {
	0.3,0.0,-0.3,
	0.6,0.0,-0.6,
	0.3,0.0,-0.3
};
static const float Sobel_y[9] = {
	0.3,0.6,0.3,
	0.0,0.0,0.0,
	-0.3,-0.6,-0.3
};

*/
////////////////////// Image_IO.cpp /////////////////////////////////////////
void read_input_image_from_file32(const char* filename);
void prepare_output_image(void);
void write_output_image_to_file32(const char* filename);
void prepare_SoA_input_and_output(void);
void prepare_AoS_input_and_output(void);
void convert_SoA_output_to_output_image_data(void);
void convert_AoS_output_to_output_image_data(void);

///////////// My_Image_Filtering_Codes.cpp ///////////////////////////////////
void convert_to_greyscale_image_SoA_CPU(void);
void convert_to_greyscale_image_AoS_CPU(void);
void convert_to_sobel_image_SoA_CPU();
void convert_to_sobel_image_AoS_CPU();
///////////// main_SoA_AoS.cpp ///////////////////////////////////
int initialize_OpenCL(void);
int set_local_work_size_and_kernel_arguments(void);
int run_OpenCL_kernel(void);
void clean_up_system(void);



#endif // __CONTEXT_SOA_AOS__