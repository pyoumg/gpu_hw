__kernel void grayscale_SoA(
const __global uchar* input_data, __global uchar* output_data,
	int n_columns, int n_rows // image width & height
){
	int data_size=n_columns*n_rows;
	int column = get_global_id(0);
	int row = get_global_id(1);

	int idx=row*n_columns+column;
	uchar intensity = (uchar)(0.299f *input_data[idx]+0.587f*input_data[idx+data_size]+0.114f * input_data[idx+2*data_size]);
	
	for(int i=0;i<3;i++)
		output_data[idx+i*data_size] = intensity;
	output_data[idx+3*data_size]=input_data[idx+3*data_size];
}