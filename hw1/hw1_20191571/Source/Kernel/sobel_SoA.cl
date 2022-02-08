__kernel void sobel_SoA(
 __global uchar* input_data, __global uchar* output_data,
	int n_columns, int n_rows, // image width & height
__constant float* sobel_x, __constant float* sobel_y) {

	int data_size=n_columns*n_rows;
	int column = get_global_id(0);
	int row = get_global_id(1);

	int output_row=row,output_col=column;

	
	//sobel

	int filter_index = 0;
	int g_x = 0;
	int g_y = 0;

	
	if (row == 0) //edge
		row++;
	else if (row == n_rows - 1)
		row--;
	if (column == 0)
		column++;
	else if (column == n_columns - 1)
		column--; 

	int idx = row*n_columns+column;

	for (int r = -1; r <= 1; r++) {
		for (int c = -1; c <= 1; c++) {
			int temp_idx = idx + (n_columns * r + c);
			g_x += sobel_x[filter_index] * (uchar)(0.299f *input_data[temp_idx]+0.587f*input_data[temp_idx+data_size]+0.114f * input_data[temp_idx+2*data_size]);
			g_y += sobel_y[filter_index] * (uchar)(0.299f *input_data[temp_idx]+0.587f*input_data[temp_idx+data_size]+0.114f * input_data[temp_idx+2*data_size]);
			filter_index++;
		}
	}

	uchar intensity = sqrt((double)g_x * g_x + g_y * g_y);
	idx=output_row*n_columns+output_col;
	for(int i=0;i<3;i++)
		output_data[idx+i*data_size]=intensity;
	output_data[idx+3*data_size] = input_data[idx+3*data_size]; 
		
}