
#define		FS		5	// Filter Size
#define		HFS		2	// Half Filter Size

__kernel void Kernel_Naive(
 __global uchar* input_data, __global uchar* output_data,
	int n_columns, int n_rows, // image width & height
__constant float* sobel_x, __constant float* sobel_y) {


	int data_size = n_columns * n_rows;
	int column = get_global_id(0);
	int row = get_global_id(1);

	int output_row=row, output_col=column;
	int2 cur_pixel_coord;
	int offset;

	int g_x = 0; //sobel result 
	int g_y = 0;

	//sobel

	int filter_index = 0;

	for (int r = -HFS; r <= HFS; r++) {
		cur_pixel_coord.y = row + r;
		if (cur_pixel_coord.y < 0) cur_pixel_coord.y = 0;
		if (cur_pixel_coord.y > n_rows - 1) cur_pixel_coord.y = n_rows - 1;

		for (int c = -HFS; c <= HFS; c++) {
			cur_pixel_coord.x = column + c;
			if (cur_pixel_coord.x < 0) cur_pixel_coord.x = 0;
			if (cur_pixel_coord.x > n_columns - 1) cur_pixel_coord.x = n_columns - 1;
			
			offset = cur_pixel_coord.y * n_columns + cur_pixel_coord.x;
			g_x += sobel_x[filter_index] * (uchar)(0.299f *input_data[offset]+0.587f*input_data[offset+data_size]+0.114f * input_data[offset+2*data_size]);
			g_y += sobel_y[filter_index] * (uchar)(0.299f *input_data[offset]+0.587f*input_data[offset+data_size]+0.114f * input_data[offset+2*data_size]);
			filter_index++;

		}
	}

	uchar intensity = sqrt((double)g_x * g_x + g_y * g_y);
	offset = row * n_columns + column;
	for(int i=0;i<3;i++)
		output_data[offset+i*data_size]=intensity; //rgb
	output_data[offset+3*data_size] = input_data[offset+3*data_size]; //a 
	
}