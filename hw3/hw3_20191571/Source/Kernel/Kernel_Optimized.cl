#define HFS						2
#define GRAYSCALE(x, y)			(uchar)(0.299f *input_data[offset_2+((n_columns * (y) + (x))*4)] +0.587f*input_data[offset_2+((n_columns * (y) + (x))*4)+1] +0.114f * input_data[offset_2+((n_columns * (y) + (x))*4)+2] )
#define SHARED_MEM(x, y)		shared_mem[SMW * (y) + (x)]


//LU, local memory
//shared_mem : store grayscale value


void process_boundary_work_groups_AoS(
	const __global uchar* input_data, __global uchar* output_data,
	int n_columns, int n_rows, // image width & height
	__constant float* sobel_x, __constant float* sobel_y, __local uchar* shared_mem,
	int offset_2, uchar last_flag) {

	int column = get_global_id(0), row = get_global_id(1);
	int loc_column = get_local_id(0), loc_row = get_local_id(1);
	int SMW = get_local_size(0) + 2 * HFS;

	int offset_2_h = offset_2 / n_columns / 4;//rgba

	SHARED_MEM(loc_column + HFS, loc_row + HFS) = GRAYSCALE(column, row);

	int side_left = 0, side_right = 0;
	int x_coord, y_coord;
	if (loc_column < HFS) {
		x_coord = (column - HFS < 0) ? 0 : column - HFS;
		SHARED_MEM(loc_column, loc_row + HFS) = GRAYSCALE(x_coord, row);
		side_left = 1;
	}
	else if (loc_column >= get_local_size(0) - HFS) {
		x_coord = (column + HFS >= n_columns) ? n_columns - 1 : column + HFS;
		SHARED_MEM(loc_column + 2 * HFS, loc_row + HFS) = GRAYSCALE(x_coord, row);
		side_right = 1;
	}

	if (loc_row < HFS) {
		y_coord = (row - HFS < 0) ? 0 : row - HFS;
		SHARED_MEM(loc_column + HFS, loc_row) = GRAYSCALE(column, y_coord);
		if (side_left == 1) {
			x_coord = (column - HFS < 0) ? 0 : column - HFS;
			y_coord = (row - HFS < 0) ? 0 : row - HFS;
			SHARED_MEM(loc_column, loc_row) = GRAYSCALE(x_coord, y_coord);
		}
		if (side_right == 1) {
			x_coord = (column + HFS >= n_columns) ? n_columns - 1 : column + HFS;
			y_coord = (row - HFS < 0) ? 0 : row - HFS;
			SHARED_MEM(loc_column + 2 * HFS, loc_row) = GRAYSCALE(x_coord, y_coord);
		}
	}
	else if (loc_row >= get_local_size(1) - HFS) {

		y_coord = (row + HFS + offset_2_h >= n_rows) ? n_rows - offset_2_h - 1 : row + HFS;
		SHARED_MEM(loc_column + HFS, loc_row + 2 * HFS) = GRAYSCALE(column, y_coord);

		if (side_left == 1) {
			x_coord = (column - HFS < 0) ? 0 : column - HFS;
			y_coord = (row + HFS + offset_2_h >= n_rows) ? n_rows - offset_2_h - 1 : row + HFS;
			SHARED_MEM(loc_column, loc_row + 2 * HFS) = GRAYSCALE(x_coord, y_coord);
		}
		if (side_right == 1) {
			x_coord = (column + HFS >= n_columns) ? n_columns - 1 : column + HFS;
			y_coord = (row + HFS + offset_2_h >= n_rows) ? n_rows - offset_2_h - 1 : row + HFS;
			SHARED_MEM(loc_column + 2 * HFS, loc_row + 2 * HFS) = GRAYSCALE(x_coord, y_coord);
		}

	}
	barrier(CLK_LOCAL_MEM_FENCE);


	int filter_index = 0;

	int offset;

	int g_x = 0; //sobel result 
	int g_y = 0;

	if (((get_group_id(0) == 0 || get_group_id(0) == get_num_groups(0) - 1) && (get_group_id(1) != 0 && get_group_id(1) != get_num_groups(1) - 1)) ||
		(offset_2 == 0 && get_group_id(1) == 0) || (last_flag == 1 && get_group_id(1) == get_num_groups(1) - 1)) {

		for (int row = loc_row - HFS; row <= loc_row + HFS; row++) {
			//for (int col = loc_column - HFS; col <= loc_column + HFS; col++) {
				//sum += convert_float4(SHARED_MEM(col + HFS, row + HFS)) * filter_weights[filter_index++];
			//}
			int col = loc_column - HFS;		// -2
			g_x += sobel_x[filter_index] * SHARED_MEM(col + HFS, row + HFS);
			g_y += sobel_y[filter_index++] * SHARED_MEM(col + HFS, row + HFS);

			col++;		// -1
			g_x += sobel_x[filter_index] * SHARED_MEM(col + HFS, row + HFS);
			g_y += sobel_y[filter_index++] * SHARED_MEM(col + HFS, row + HFS);

			col++;		// 0
			g_x += sobel_x[filter_index] * SHARED_MEM(col + HFS, row + HFS);
			g_y += sobel_y[filter_index++] * SHARED_MEM(col + HFS, row + HFS);

			col++;		// +1
			g_x += sobel_x[filter_index] * SHARED_MEM(col + HFS, row + HFS);
			g_y += sobel_y[filter_index++] * SHARED_MEM(col + HFS, row + HFS);

			col++;		// +2
			g_x += sobel_x[filter_index] * SHARED_MEM(col + HFS, row + HFS);
			g_y += sobel_y[filter_index++] * SHARED_MEM(col + HFS, row + HFS);
		}
		uchar intensity = sqrt((double)g_x * g_x + g_y * g_y);
		offset = (row * n_columns + column) * 4;


		for (int i = 0; i < 3; i++)
			output_data[offset_2 + offset + i] = intensity; //rgb
		output_data[offset_2 + offset + 3] = input_data[offset_2 + offset + 3]; //a 

	}

}


__kernel void Kernel_Optimized(
	const __global uchar* input_data, __global uchar* output_data,
	int n_columns, int n_rows, // image width & height
	__constant float* sobel_x, __constant float* sobel_y, __local uchar* shared_mem,
	unsigned int offset_2, uchar last_flag, int n_kernel_loop_iterations) {

	for (int iter = 0; iter < n_kernel_loop_iterations; iter++) {


		if (get_group_id(0) == 0 || get_group_id(0) == get_num_groups(0) - 1 ||
			get_group_id(1) == 0 || get_group_id(1) == get_num_groups(1) - 1) {

			process_boundary_work_groups_AoS(input_data, output_data, n_columns, n_rows, sobel_x, sobel_y, shared_mem, offset_2, last_flag);
			if (iter == n_kernel_loop_iterations - 1)
				return;
			else
				continue;
		}

		int column = get_global_id(0), row = get_global_id(1);
		int loc_column = get_local_id(0), loc_row = get_local_id(1);
		int SMW = get_local_size(0) + 2 * HFS;

		SHARED_MEM(loc_column + HFS, loc_row + HFS) = GRAYSCALE(column, row);

		int side_left = 0, side_right = 0;
		if (loc_column < HFS) {
			SHARED_MEM(loc_column, loc_row + HFS) = GRAYSCALE(column - HFS, row);
			side_left = 1;
		}
		else if (loc_column >= get_local_size(0) - HFS) {
			SHARED_MEM(loc_column + 2 * HFS, loc_row + HFS) = GRAYSCALE(column + HFS, row);
			side_right = 1;
		}

		if (loc_row < HFS) {
			SHARED_MEM(loc_column + HFS, loc_row) = GRAYSCALE(column, row - HFS);
			if (side_left == 1)
				SHARED_MEM(loc_column, loc_row) = GRAYSCALE(column - HFS, row - HFS);
			if (side_right == 1)
				SHARED_MEM(loc_column + 2 * HFS, loc_row) = GRAYSCALE(column + HFS, row - HFS);
		}
		else if (loc_row >= get_local_size(1) - HFS) {
			SHARED_MEM(loc_column + HFS, loc_row + 2 * HFS) = GRAYSCALE(column, row + HFS);
			if (side_left == 1)
				SHARED_MEM(loc_column, loc_row + 2 * HFS) = GRAYSCALE(column - HFS, row + HFS);
			if (side_right == 1)
				SHARED_MEM(loc_column + 2 * HFS, loc_row + 2 * HFS) = GRAYSCALE(column + HFS, row + HFS);
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		int filter_index = 0;

		int offset;

		int g_x = 0; //sobel result 
		int g_y = 0;


		for (int row = loc_row - HFS; row <= loc_row + HFS; row++) {
			//for (int col = loc_column - HFS; col <= loc_column + HFS; col++) {
				//sum += convert_float4(SHARED_MEM(col + HFS, row + HFS)) * filter_weights[filter_index++];
			//}
			int col = loc_column - HFS;		// -2
			g_x += sobel_x[filter_index] * SHARED_MEM(col + HFS, row + HFS);
			g_y += sobel_y[filter_index++] * SHARED_MEM(col + HFS, row + HFS);

			col++;		// -1
			g_x += sobel_x[filter_index] * SHARED_MEM(col + HFS, row + HFS);
			g_y += sobel_y[filter_index++] * SHARED_MEM(col + HFS, row + HFS);

			col++;		// 0
			g_x += sobel_x[filter_index] * SHARED_MEM(col + HFS, row + HFS);
			g_y += sobel_y[filter_index++] * SHARED_MEM(col + HFS, row + HFS);

			col++;		// +1
			g_x += sobel_x[filter_index] * SHARED_MEM(col + HFS, row + HFS);
			g_y += sobel_y[filter_index++] * SHARED_MEM(col + HFS, row + HFS);

			col++;		// +2
			g_x += sobel_x[filter_index] * SHARED_MEM(col + HFS, row + HFS);
			g_y += sobel_y[filter_index++] * SHARED_MEM(col + HFS, row + HFS);
		}
		uchar intensity = sqrt((double)g_x * g_x + g_y * g_y);
		offset = (row * n_columns + column) * 4;

		for (int i = 0; i < 3; i++)
			output_data[offset_2 + offset + i] = intensity; //rgb
		output_data[offset_2 + offset + 3] = input_data[offset_2 + offset + 3]; //a 
	}
}
