__kernel void grayscale_AoS(
const __global uchar* input_data, __global uchar* output_data,
	int n_columns, int n_rows // image width & height
){

	int column = get_global_id(0);
	int row = get_global_id(1);

	int idx=4*(row*n_columns+column);
	uchar intensity = (uchar)(0.299f *input_data[idx]+0.587f*input_data[idx+1]+0.114f * input_data[idx+2]);
	
	for(int i=0;i<3;i++)
		output_data[idx+i] = intensity;
	output_data[idx+3]=input_data[idx+3];
}