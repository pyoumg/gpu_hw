//
//  wave_equation.cl
//
//  Written for CSEG437_CSE5437
//  Department of Computer Science and Engineering
//  Copyright ©2020 Sogang University. All rights reserved.
//

#define K_MINUS_1	0
#define K				1
__kernel void wave_equation_IMPLICIT(
	__global float4* grid_next, __global float4* grid_cur,
	__global float4* grid_prev, float beta,
	int Jacobian_iter, __global float4* grid_tmp) {
	int global_size_0 = get_global_size(0), global_size_1 = get_global_size(1);
	int global_id_0 = get_global_id(0), global_id_1 = get_global_id(1);

	// Check the boundary conditions first.
	if ((global_id_0 == 0) || (global_id_0 == global_size_0 - 1) ||
		(global_id_1 == 0) || (global_id_1 == global_size_1 - 1))
		return;

	__global float4* x[2];  // x(K_MINUS_1) --> x(K)
	if (Jacobian_iter == 0) {
		x[K_MINUS_1] = grid_cur; x[K] = grid_tmp;
	}
	else if ((Jacobian_iter % 2) == 1) { // Jacobian_iter > 0 and odd
		x[K_MINUS_1] = grid_tmp; x[K] = grid_next;
	}
	else { // Jacobian_iter > 0 and even
		x[K_MINUS_1] = grid_next; x[K] = grid_tmp;
	}

	int index = global_id_1 * global_size_0 + global_id_0;

	float diagonal = 1.0f + 4.0f * beta, b = 2 * grid_cur[index].y - grid_prev[index].y;
	float sum = x[K_MINUS_1][index - global_size_0].y + x[K_MINUS_1][index - 1].y
		+ x[K_MINUS_1][index + 1].y + x[K_MINUS_1][index + global_size_0].y;

	x[K][index].y = (b + beta * sum) / diagonal;
}

__kernel void wave_equation_IMPLICIT_DEBUG(
	__global float4* grid_next, __global float4* grid_cur,
	__global float4* grid_prev, float beta,
	int Jacobian_iter, __global float4* grid_tmp) {
	int global_size_0 = get_global_size(0), global_size_1 = get_global_size(1);
	int global_id_0 = get_global_id(0), global_id_1 = get_global_id(1);

	// Check the boundary conditions first.
	if ((global_id_0 == 0) || (global_id_0 == global_size_0 - 1) ||
		(global_id_1 == 0) || (global_id_1 == global_size_1 - 1))
		return;

	__global float4* x[2];  // x(K_MINUS_1) --> x(K)
	if (Jacobian_iter == 0) {
		x[K_MINUS_1] = grid_cur; x[K] = grid_tmp;
	}
	else if ((Jacobian_iter % 2) == 1) { // Jacobian_iter > 0 and odd
		x[K_MINUS_1] = grid_tmp; x[K] = grid_next;
	}
	else { // Jacobian_iter > 0 and even
		x[K_MINUS_1] = grid_next; x[K] = grid_tmp;
	}

	int index = global_id_1 * global_size_0 + global_id_0;
	float diagonal = 1.0f + 4.0f * beta, b = 2 * grid_cur[index].y - grid_prev[index].y;

	if ((global_id_0 == 64) && (global_id_1 == 64)) printf("[Before iter = %2d]   %g\n", Jacobian_iter, x[K_MINUS_1][index].y);

	float sum = x[K_MINUS_1][index - global_size_0].y + x[K_MINUS_1][index - 1].y
		+ x[K_MINUS_1][index + 1].y + x[K_MINUS_1][index + global_size_0].y;
	x[K][index].y = (b + beta * sum) / diagonal;
				 
	if ((global_id_0 == 64) && (global_id_1 == 64)) printf("[After iter  = %2d]          %g\n", Jacobian_iter, x[K][index].y);
}

__kernel void wave_equation_EXPLICIT(
	__global float4* grid_next, __global float4* grid_cur,
	__global float4* grid_prev, float beta ) {
	int global_size_0 = get_global_size(0), global_size_1 = get_global_size(1);
	int global_id_0 = get_global_id(0), global_id_1 = get_global_id(1);

	// Check the boundary conditions first.
	if ((global_id_0 == 0) || (global_id_0 == global_size_0 - 1) ||
		(global_id_1 == 0) || (global_id_1 == global_size_1 - 1))
		return;

   // HW3

	__global float4* x[2];  // x(K_MINUS_1) --> x(K)

	x[K_MINUS_1] = grid_cur; x[K] = grid_next;
	

	int index = global_id_1 * global_size_0 + global_id_0;

	x[K][index].y = 2.0 * (1.0 - 2.0 * beta) * x[K_MINUS_1][index].y + beta * (x[K_MINUS_1][index + 1].y + 
		x[K_MINUS_1][index - 1].y + x[K_MINUS_1][index + global_size_0].y + x[K_MINUS_1][index - global_size_0].y) - grid_prev[index].y;
}
