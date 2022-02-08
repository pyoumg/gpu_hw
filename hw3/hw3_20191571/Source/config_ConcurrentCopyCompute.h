//
//  config_ConcurrentCopyCompute.h
//
//  Written for CSEG437_CSE5437
//  Department of Computer Science and Engineering
//  Copyright © 2021 Sogang University. All rights reserved.
//

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define NO_437		0
#define YES_437		1
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define MAXIMUM_COMMAND_QUEUES			32
#define N_KERNEL_LOOP_ITERATIONS		100 // Determine the relative workload of kernel
#define N_KERNEL_CALL_ITERATIONS		100
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GLOBAL_WORK_SIZE = n_elements must be multiple of MAXIMUM_COMMAND_QUEUES.

#define	LOCAL_WORK_SIZE_0			64		// Dim 0 (x)
#define	LOCAL_WORK_SIZE_1			16	// Dim 1 (y)
#define OPENCL_C_PROG_FILE_NAME		"Source/Kernel/kernel_Optimized.cl"
#define KERNEL_NAME							"Kernel_Optimized"
#define	INPUT_IMAGE					9
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define COPY_COMPUTE_TYPE_THREE_QUEUES_WITH_EVENTS			0
#define COPY_COMPUTE_TYPE_MULTIPLE_QUEUES 					1
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define COPY_COMPUTE_TYPE	 COPY_COMPUTE_TYPE_MULTIPLE_QUEUES 
#define COPY_COMPUTE_COMMANDS_PROFILING    NO_437
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define INPUT_FILE_0			"Image_0_7360_4832"
#define INPUT_FILE_1			"Image_1_9984_6400"
#define INPUT_FILE_2			"Image_2_7680_4320"
#define INPUT_FILE_3			"Image_3_8960_5408"
#define INPUT_FILE_4			"Image_4_6304_4192"
#define INPUT_FILE_5			"Image_5_1856_1376"
#define INPUT_FILE_8			"Grass_texture_2048_2048"
#define INPUT_FILE_9			"Tiger_texture_512_512"
#define INPUT_FILE_10		"Plain_color_128_192_64_1024_1024"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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
