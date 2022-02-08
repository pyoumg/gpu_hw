#include "Context_SoA_AoS.h"

void convert_to_greyscale_image_SoA_CPU(void) {
	for (unsigned int i = 0; i < context.image_width * context.image_height; i++) {

		BYTE intensity = BYTE(0.299f * context.SoA_image_input.R_plane[i]  // R
			+ 0.587f * context.SoA_image_input.G_plane[i]  // G
			+ 0.114f * context.SoA_image_input.B_plane[i]);  // B
		context.SoA_image_output.R_plane[i] = intensity;
		context.SoA_image_output.G_plane[i] = intensity;
		context.SoA_image_output.B_plane[i] = intensity;
		context.SoA_image_output.A_plane[i] = context.SoA_image_input.A_plane[i];
	}
}

void convert_to_greyscale_image_AoS_CPU(void) {
	Pixel_Channels* tmp_ptr_input = context.AoS_image_input;
	Pixel_Channels* tmp_ptr_output = context.AoS_image_output;
	for (unsigned int i = 0; i < context.image_width * context.image_height; i++) {
		BYTE intensity = BYTE(0.299f * tmp_ptr_input->R // R
			+ 0.587f * tmp_ptr_input->G  // G
			+ 0.114f * tmp_ptr_input->B);  // B
		tmp_ptr_output->R = intensity;
		tmp_ptr_output->G = intensity;
		tmp_ptr_output->B = intensity;
		tmp_ptr_output->A = tmp_ptr_input->A;

		tmp_ptr_input++; tmp_ptr_output++;
	}
}

void convert_to_sobel_image_SoA_CPU() {

	/*grayscale input*/
	for (unsigned int i = 0; i < context.image_width * context.image_height; i++) {

		BYTE intensity = BYTE(0.299f * context.SoA_image_input.R_plane[i]  // R
			+ 0.587f * context.SoA_image_input.G_plane[i]  // G
			+ 0.114f * context.SoA_image_input.B_plane[i]);  // B
		context.SoA_image_input.R_plane[i] = intensity;
		context.SoA_image_input.G_plane[i] = intensity;
		context.SoA_image_input.B_plane[i] = intensity;

	}

	UINT width = context.image_width;
	UINT height = context.image_height;


	for (unsigned int i = 0; i < context.image_width * context.image_height; i++) {

		int filter_index = 0;
		int g_x = 0;
		int g_y = 0;


		UINT idx_r = i / width;
		UINT idx_c = i % width;

		if (idx_r == 0) //edge
			idx_r++;
		else if (idx_r == height - 1)
			idx_r--;
		if (idx_c == 0)
			idx_c++;
		else if (idx_c == width - 1)
			idx_c--;
		UINT idx = idx_r * width + idx_c;

		for (int r = -1; r <= 1; r++) {
			for (int c = -1; c <= 1; c++) {
				g_x += Sobel_x[filter_index] * context.SoA_image_input.R_plane[idx + width * r + c];
				g_y += Sobel_y[filter_index] * context.SoA_image_input.R_plane[idx + width * r + c];
				filter_index++;
			}
		}

		BYTE intensity = sqrt((double)g_x * g_x + g_y * g_y);

		context.SoA_image_output.R_plane[i] = intensity;
		context.SoA_image_output.G_plane[i] = intensity;
		context.SoA_image_output.B_plane[i] = intensity;
		context.SoA_image_output.A_plane[i] = context.SoA_image_input.A_plane[i];
	}
}

void convert_to_sobel_image_AoS_CPU() {
	Pixel_Channels* tmp_ptr_input = context.AoS_image_input;
	Pixel_Channels* tmp_ptr_output = context.AoS_image_output;

	//grayscale 
	for (unsigned int i = 0; i < context.image_width * context.image_height; i++) {
		BYTE intensity = BYTE(0.299f * tmp_ptr_input->R // R
			+ 0.587f * tmp_ptr_input->G  // G
			+ 0.114f * tmp_ptr_input->B);  // B
		tmp_ptr_input->R = intensity;
		tmp_ptr_input->G = intensity;
		tmp_ptr_input->B = intensity;

		tmp_ptr_input++;
	}


	UINT width = context.image_width;
	UINT height = context.image_height;

	for (unsigned int i = 0; i < context.image_width * context.image_height; i++) {
		tmp_ptr_input = context.AoS_image_input;
		int filter_index = 0;
		int g_x = 0;
		int g_y = 0;


		UINT idx_r = i / width;
		UINT idx_c = i % width;

		if (idx_r == 0) //edge
			idx_r++;
		else if (idx_r == height - 1)
			idx_r--;
		if (idx_c == 0)
			idx_c++;
		else if (idx_c == width - 1)
			idx_c--;
		UINT idx = idx_r * width + idx_c;

		for (int r = -1; r <= 1; r++) {
			for (int c = -1; c <= 1; c++) {
				g_x += Sobel_x[filter_index] * (tmp_ptr_input[idx + width * r + c]).R;
				g_y += Sobel_y[filter_index] * (tmp_ptr_input[idx + width * r + c]).R;
				filter_index++;
			}
		}

		BYTE intensity = sqrt((double)g_x * g_x + g_y * g_y);

		tmp_ptr_output->R = intensity;
		tmp_ptr_output->G = intensity;
		tmp_ptr_output->B = intensity;
		tmp_ptr_output->A = (context.AoS_image_input + i)->A;
		tmp_ptr_output++;
	}

}

