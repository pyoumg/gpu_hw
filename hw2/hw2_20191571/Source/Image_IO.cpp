#include "Context_SoA_AoS.h"

void read_input_image_from_file32(const char* filename) {
    // Assume everything is fine with reading image from file: no error checking is done.
    context.image_format = FreeImage_GetFileType(filename, 0);
    FIBITMAP* fi_bitmap = FreeImage_Load(context.image_format, filename);
    context.image_width = FreeImage_GetWidth(fi_bitmap);
    context.image_height = FreeImage_GetHeight(fi_bitmap);
    int bits_per_pixel = FreeImage_GetBPP(fi_bitmap);

    fprintf(stdout, "*** A %d-bit image of %d x %d pixels was read from %s.\n", bits_per_pixel,
        context.image_width, context.image_height, filename);

    if (bits_per_pixel == 32) {
        context.input.fi_bitmap_32 = fi_bitmap;
    }
    else {
        fprintf(stdout, "*** Converting texture from %d bits to 32 bits...\n", bits_per_pixel);
        context.input.fi_bitmap_32 = FreeImage_ConvertTo32Bits(fi_bitmap);
    }
    FreeImage_Unload(fi_bitmap);

    context.image_pitch = FreeImage_GetPitch(context.input.fi_bitmap_32);
    printf("*** input image: width = %d, height = %d, bpp = %d, pitch = %d\n", context.image_width,
        context.image_height, 32, context.image_pitch);

    context.input.image_data = FreeImage_GetBits(context.input.fi_bitmap_32);
}

void prepare_output_image(void) {
    context.image_data_bytes
        = (size_t)context.image_width * (size_t)context.image_height * sizeof(unsigned char) * 4;

    if (context.image_data_bytes != (size_t)context.image_pitch * (size_t)context.image_height) {
        fprintf(stderr, "*** Error: do something!\n");
        exit(EXIT_FAILURE);
    }

    context.output.image_data = (BYTE*)malloc(sizeof(BYTE) * context.image_data_bytes);
    if (context.output.image_data == NULL) {
        fprintf(stderr, "*** Error: cannot allocate memory for output image...\n");
        exit(EXIT_FAILURE);
    }
}

void write_output_image_to_file32(const char* filename) {
    // Assume everything is fine with writing image into file: no error checking is done.
    context.output.fi_bitmap_32 = FreeImage_ConvertFromRawBits(context.output.image_data,
        context.image_width, context.image_height, context.image_pitch, 32,
        FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, FALSE);

    FreeImage_Save(FIF_PNG, context.output.fi_bitmap_32, filename, 0);
    fprintf(stdout, "*** A 32-bit image of %d x %d pixels was written to %s.\n",
        context.image_width, context.image_height, filename);
}

void prepare_SoA_input_and_output(void) {
    context.SoA_image_input.R_plane
        = (BYTE*)malloc(sizeof(BYTE) * context.image_width * context.image_height);
    context.SoA_image_input.G_plane
        = (BYTE*)malloc(sizeof(BYTE) * context.image_width * context.image_height);
    context.SoA_image_input.B_plane
        = (BYTE*)malloc(sizeof(BYTE) * context.image_width * context.image_height);
    context.SoA_image_input.A_plane
        = (BYTE*)malloc(sizeof(BYTE) * context.image_width * context.image_height);

    if (!context.SoA_image_input.R_plane || !context.SoA_image_input.G_plane
        || !context.SoA_image_input.B_plane || !context.SoA_image_input.A_plane) {
        fprintf(stderr, "*** Error: cannot allocate memory for SoA_image_input...\n");
        exit(EXIT_FAILURE);
    }

    context.SoA_image_output.R_plane
        = (BYTE*)malloc(sizeof(BYTE) * context.image_width * context.image_height);
    context.SoA_image_output.G_plane
        = (BYTE*)malloc(sizeof(BYTE) * context.image_width * context.image_height);
    context.SoA_image_output.B_plane
        = (BYTE*)malloc(sizeof(BYTE) * context.image_width * context.image_height);
    context.SoA_image_output.A_plane
        = (BYTE*)malloc(sizeof(BYTE) * context.image_width * context.image_height);

    if (!context.SoA_image_output.R_plane || !context.SoA_image_output.G_plane
        || !context.SoA_image_output.B_plane || !context.SoA_image_output.A_plane) {
        fprintf(stderr, "*** Error: cannot allocate memory for SoA_image_input...\n");
        exit(EXIT_FAILURE);
    }

    int offset, index = 0;
    for (unsigned int i = 0; i < context.image_height; i++) {
        offset = i * context.image_pitch;
        for (unsigned int j = 0; j < context.image_width; j++) {
            context.SoA_image_input.B_plane[index] = (BYTE)context.input.image_data[offset];
            context.SoA_image_input.G_plane[index] = (BYTE)context.input.image_data[offset + 1];
            context.SoA_image_input.R_plane[index] = (BYTE)context.input.image_data[offset + 2];
            context.SoA_image_input.A_plane[index] = (BYTE)context.input.image_data[offset + 3];
            offset += 4; index++;
        }
    }
}


void convert_SoA_output_to_output_image_data(void) {
    int offset = 0, index = 0;
    for (unsigned int i = 0; i < context.image_height; i++) {
        offset = i * context.image_pitch;
        for (unsigned int j = 0; j < context.image_width; j++) {
            context.output.image_data[offset] = context.SoA_image_output.B_plane[index];
            context.output.image_data[offset + 1] = context.SoA_image_output.G_plane[index];
            context.output.image_data[offset + 2] = context.SoA_image_output.R_plane[index];
            context.output.image_data[offset + 3] = context.SoA_image_output.A_plane[index];
            offset += 4; index++;
        }
    }
}

