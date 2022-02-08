#pragma once
//
//  Config_WE.h
//
//  Written for CSEG437_CSE5437
//  Department of Computer Science and Engineering
//  Copyright © 2021 Sogang University. All rights reserved.
//
// EXPLICIT NOT WORKING: GRID_RESOLUTION = 128 / SIMULATION_PARAMETERS = 1 

////////////////////////////////////////////////////////////////////////////////
#define		WINDOW_WIDTH				1200
#define		WINDOW_HEIGHT				800

#define		GRID_RESOLUTION			128 // Must be a multiple of 32
#define		SIDE_LENGTH					100.0f // Do not modify this paramenter.

// 0: GPU_EXPLICIT, 1: GPU_IMPLICIT_JACOBI, 2: CPU_IMPLICIT_JACOBI 3: CPU_IMPLICIT_GAUSS_SEIDEL
#define		SIMULATION_METHOD					3	
// 0: FAST FOWARD, 1: REGULAR SPEED, 2: SLOW MOTION
#define		SIMULATION_PARAMETERS					0		 

#define		NUMBER_OF_SIMULATION_STEPS			5000
#define		NUMBER_OF_JACOBIAN_ITERATIONS		10	// MUST be an even number.

#define		LOCAL_WORK_SIZE_0			32		// Dim 0 (x)
#define		LOCAL_WORK_SIZE_1			8		// Dim 1 (y)
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
#if SIMULATION_METHOD == 0
#define USE_GPU_EXPLICIT
#elif SIMULATION_METHOD == 1
#define USE_GPU_IMPLICIT_JACOBI
#elif SIMULATION_METHOD == 2
#define USE_CPU_IMPLICIT_JACOBI 
#elif SIMULATION_METHOD == 3
#define USE_CPU_IMPLICIT_GAUSS_SEIDEL 
#endif
////////////////////////////////////////////////////////////////////////////////
 