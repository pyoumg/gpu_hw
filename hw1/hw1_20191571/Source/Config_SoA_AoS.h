//
//  Config_SoA_AoS.h
//
//  Written for CSEG437_CSE5437
//  Department of Computer Science and Engineering
//  Copyright © 2021 Sogang University. All rights reserved.
//

/////////////////////////////////////////////////////////////////////////////////////
#define		LOCAL_WORK_SIZE_0			32		// Dim 0 (x)
#define		LOCAL_WORK_SIZE_1			16		// Dim 1 (y)

/////////////////////////////////////////////////////////////////////////////////////
#define		SoA_GS_CPU		0		// SoA Grayscale on CPU
#define		AoS_GS_CPU		1		// AoS Grayscale on CPU 
#define		SoA_SO_CPU		2		// SoA Sobel operator on CPU
#define		AoS_SO_CPU		3		// AoS Sobel operator on CPU : 여기까지  cpu
#define		SoA_GS_GPU		4		// SoA Grayscale on GPU
#define		AoS_GS_GPU		5		// AoS Grayscale on GPU
#define		SoA_SO_GPU		6		// SoA Sobel operator on GPU
#define		AoS_SO_GPU		7		// AoS Sobel operator on GPU


#define INPUT_FILE_0			"Image_0_7360_4832"
#define INPUT_FILE_1			"Image_1_9984_6400"
#define INPUT_FILE_2			"Image_2_7680_4320"
#define INPUT_FILE_3			"Image_3_8960_5408"
#define INPUT_FILE_4			"Image_4_6304_4192"
#define INPUT_FILE_5			"Image_5_1856_1376"
#define INPUT_FILE_8			"Grass_texture_2048_2048"
#define INPUT_FILE_9			"Tiger_texture_512_512"
#define INPUT_FILE_10		"Plain_color_128_192_64_1024_1024"
/////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////
#define		INPUT_IMAGE					0
#define		IMAGE_OPERATION			AoS_GS_GPU
#define		N_EXECUTIONS				10

#define OPENCL_C_PROG_FILE_NAME "Source/Kernel/grayscale_SoA.cl"
#define KERNEL_NAME "grayscale_SoA"
/////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////
#if   IMAGE_OPERATION == SoA_GS_GPU
#define OPENCL_C_PROG_FILE_NAME "Source/Kernel/grayscale_SoA.cl"
#define KERNEL_NAME "grayscale_SoA"
#elif IMAGE_OPERATION == AoS_GS_GPU
#define OPENCL_C_PROG_FILE_NAME "Source/Kernel/grayscale_AoS.cl"
#define KERNEL_NAME "grayscale_AoS"
#elif IMAGE_OPERATION == SoA_SO_GPU
#define OPENCL_C_PROG_FILE_NAME "Source/Kernel/sobel_SoA.cl"
#define KERNEL_NAME "sobel_SoA"
#elif IMAGE_OPERATION == AoS_SO_GPU
#define OPENCL_C_PROG_FILE_NAME "Source/Kernel/sobel_AoS.cl"
#define KERNEL_NAME "sobel_AoS"
#endif
/////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////
#if INPUT_IMAGE == 0
#define	INPUT_FILE 			INPUT_FILE_0
#elif INPUT_IMAGE == 1
#define	INPUT_FILE 			INPUT_FILE_1
#elif INPUT_IMAGE == 2
#define	INPUT_FILE 			INPUT_FILE_2
#elif INPUT_IMAGE == 3
#define	INPUT_FILE 			INPUT_FILE_3
#elif INPUT_IMAGE == 4
#define	INPUT_FILE 			INPUT_FILE_4
#elif INPUT_IMAGE == 5
#define	INPUT_FILE 			INPUT_FILE_5
#elif INPUT_IMAGE == 8
#define	INPUT_FILE 			INPUT_FILE_8
#elif INPUT_IMAGE == 9
#define	INPUT_FILE 			INPUT_FILE_9
#else INPUT_IMAGE == 10
#define	INPUT_FILE 			INPUT_FILE_10
#endif

#define	INPUT_FILE_NAME			"Data/Input/" INPUT_FILE ".jpg"
#define	OUTPUT_FILE_NAME		"Data/Output/" INPUT_FILE "_out.png"
/////////////////////////////////////////////////////////////////////////////////////
#define MAKE_STAT_ITEM_LIST_CPU(string, time)  sprintf((string), "\n*** Host(CPU):: KER = %d, Time: %.3fms", \
			IMAGE_OPERATION, time);
#define MAKE_STAT_ITEM_LIST(string, gws, lws) sprintf((string), "IMAGE = %d(%s)," \
                    "    KER = %d, GWS = (%d, %d), LWS = (%d, %d)", INPUT_IMAGE, INPUT_FILE_NAME, \
					 IMAGE_OPERATION, (gws)[0], (gws)[1], (lws)[0], (lws)[1]);