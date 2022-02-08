#pragma warning(disable : 4996)

//
//  my_OpenCL_util_2_2.cpp
//
//  Written for CSEG437_CSE5437
//  Department of Computer Science and Engineering
//  Copyright © 2021 Sogang University. All rights reserved.
//

#include "my_OpenCL_util_2_2.h"

/******************************************************************************************************/
char *get_error_flag(cl_int errcode) {
    switch (errcode) {
        case CL_SUCCESS:
            return (char*) "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND:
            return (char*) "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE:
            return (char*) "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE:
            return (char*) "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
            return (char*) "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES:
            return (char*) "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY:
            return (char*) "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE:
            return (char*) "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP:
            return (char*) "CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH:
            return (char*) "CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:
            return (char*) "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE:
            return (char*) "CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE:
            return (char*) "CL_MAP_FAILURE";
        case CL_MISALIGNED_SUB_BUFFER_OFFSET:
            return (char*) "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
            return (char*) "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case CL_COMPILE_PROGRAM_FAILURE:
            return (char*) "CL_COMPILE_PROGRAM_FAILURE";
        case CL_LINKER_NOT_AVAILABLE:
            return (char*) "CL_LINKER_NOT_AVAILABLE";
        case CL_LINK_PROGRAM_FAILURE:
            return (char*) "CL_LINK_PROGRAM_FAILURE";
        case CL_DEVICE_PARTITION_FAILED:
            return (char*) "CL_DEVICE_PARTITION_FAILED";
        case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
            return (char*) "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
        case CL_INVALID_VALUE:
            return (char*) "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE:
            return (char*) "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM:
            return (char*) "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE:
            return (char*) "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT:
            return (char*) "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES:
            return (char*) "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE:
            return (char*) "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR:
            return (char*) "CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT:
            return (char*) "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
            return (char*) "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case CL_INVALID_IMAGE_SIZE:
            return (char*) "CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_SAMPLER:
            return (char*) "CL_INVALID_SAMPLER";
        case CL_INVALID_BINARY:
            return (char*) "CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS:
            return (char*) "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM:
            return (char*) "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE:
            return (char*) "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME:
            return (char*) "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION:
            return (char*) "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL:
            return (char*) "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX:
            return (char*) "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE:
            return (char*) "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE:
            return (char*) "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS:
            return (char*) "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION:
            return (char*) "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE:
            return (char*) "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE:
            return (char*) "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET:
            return (char*) "CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_EVENT_WAIT_LIST:
            return (char*) "CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_EVENT:
            return (char*) "CL_INVALID_EVENT";
        case CL_INVALID_OPERATION:
            return (char*) "CL_INVALID_OPERATION";
        case CL_INVALID_GL_OBJECT:
            return (char*) "CL_INVALID_GL_OBJECT";
        case CL_INVALID_BUFFER_SIZE:
            return (char*) "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_MIP_LEVEL:
            return (char*) "CL_INVALID_MIP_LEVEL";
        case CL_INVALID_GLOBAL_WORK_SIZE:
            return (char*) "CL_INVALID_GLOBAL_WORK_SIZE";
        case CL_INVALID_PROPERTY:
            return (char*) "CL_INVALID_PROPERTY";
        case CL_INVALID_IMAGE_DESCRIPTOR:
            return (char*) "CL_INVALID_IMAGE_DESCRIPTOR";
        case CL_INVALID_COMPILER_OPTIONS:
            return (char*) "CL_INVALID_COMPILER_OPTIONS";
        case CL_INVALID_LINKER_OPTIONS:
            return (char*) "CL_INVALID_LINKER_OPTIONS";
        case CL_INVALID_DEVICE_PARTITION_COUNT:
            return (char*) "CL_INVALID_DEVICE_PARTITION_COUNT";
/*        case CL_INVALID_PIPE_SIZE:
            return (char*) "CL_INVALID_PIPE_SIZE";
        case CL_INVALID_DEVICE_QUEUE:
            return (char*) "CL_INVALID_DEVICE_QUEUE";
*/
        default:
            return (char*)"UNKNOWN ERROR CODE";
    }
}

int check_error_code(cl_int errcode, int line, const char *file) {
    if (errcode != CL_SUCCESS) {
        fprintf(stderr, "^^^ OpenCL error in Line %d of FILE %s: %s(%d)\n\n",
            line, file, get_error_flag(errcode), errcode);
        return 1;
    }
    else
        return 0;
}
/******************************************************************************************************/

/******************************************************************************************************/
 //#define _SHOW_OPENCL_C_PROGRAM
size_t read_kernel_from_file(const char *filename, char **source_str) {
    FILE *fp;
    size_t count;
    
    if ((fp = fopen(filename, "rb")) == NULL) {
        fprintf(stderr, "Error: cannot open the file %s for reading...\n", filename);
        exit(EXIT_FAILURE);
    }
    
    fseek(fp, 0, SEEK_END);
    count = ftell(fp);

    fseek(fp, 0, SEEK_SET);
    
    if (!source_str) free(source_str);

    *source_str = (char *)malloc(count + 1);
    if (*source_str == NULL) {
        fprintf(stderr, "Error: cannot allocate memory for reading the file %s for reading...\n", filename);
        exit(EXIT_FAILURE);
    }
    
    fread(*source_str, sizeof(char), count, fp);
    *(*source_str + count) = '\0';
    
    fclose(fp);
    
#ifdef _SHOW_OPENCL_C_PROGRAM
    fprintf(stdout, "\n^^^^^^^^^^^^^^ The OpenCL C program ^^^^^^^^^^^^^^\n%s\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n\n", *source_str);
#endif
    
    return count;
}
/******************************************************************************************************/

/******************************************************************************************************/
int print_build_log(cl_program program, cl_device_id device, const char *title_suppl) {
    cl_int errcode_ret;
    char *string;
    size_t string_length;
    
    fprintf(stderr, "\n^^^^^^^^^^^^ Program build log (%s) ^^^^^^^^^^^^\n", title_suppl);
    errcode_ret = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &string_length);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;
    
    string = (char *)malloc(string_length);
    if (string == NULL) {
        fprintf(stderr, "Error: cannot allocate memory for holding a program build log...\n");
        return 1;
    }
    
    errcode_ret = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, string_length, string, NULL);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;
    
    fprintf(stderr, "%s", string);
    fprintf(stderr, "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n");
    free(string);

    return 0;
}
/******************************************************************************************************/

/******************************************************************************************************/
int compute_elapsed_time(cl_event event, cl_ulong* time, cl_profiling_info from, cl_profiling_info to) {
    cl_ulong from_time, to_time;
    cl_int errcode_ret;
    
    errcode_ret = clGetEventProfilingInfo(event, from, sizeof(cl_ulong), &from_time, NULL);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    errcode_ret = clGetEventProfilingInfo(event, to, sizeof(cl_ulong), &to_time, NULL);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    *time = (cl_ulong)(to_time - from_time);

    return 0;
}

int print_device_time(cl_event event) {
    // Consider to use CL_PROFILING_COMMAND_END for OpenCL 2.0 or higher.
    cl_ulong time_elapsed;

    fprintf(stdout, "      * Time by device clock:\n");
    if (compute_elapsed_time(event, &time_elapsed, CL_PROFILING_COMMAND_QUEUED, CL_PROFILING_COMMAND_END))
        return 1;
    fprintf(stdout, "        - Time from QUEUED to END = %.3fms\n", time_elapsed * 1.0e-6f);

    if (compute_elapsed_time(event, &time_elapsed, CL_PROFILING_COMMAND_SUBMIT, CL_PROFILING_COMMAND_END))
        return 1;
    fprintf(stdout, "        - Time from SUBMIT to END = %.3fms\n", time_elapsed * 1.0e-6f);

    if (compute_elapsed_time(event, &time_elapsed, CL_PROFILING_COMMAND_START, CL_PROFILING_COMMAND_END))
        return 1;
    fprintf(stdout, "        - Time from START to END = %.3fms\n\n", time_elapsed * 1.0e-6f);

    return 0;
}
/******************************************************************************************************/

/******************************************************************************************************/
void print_device_0(cl_device_id device) {
#define MAX_BUFFER_SIZE 1024
    char _buffer[MAX_BUFFER_SIZE]; // Use a char buffer of enough size for convenience.
    
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(_buffer), _buffer, NULL);
    fprintf(stdout, "     - CL_DEVICE_NAME:\t\t%s\n", _buffer);
    
    clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(_buffer), _buffer, NULL);
    fprintf(stdout, "     - CL_DEVICE_TYPE:\t\t");
    {
        cl_device_type tmp = *((cl_device_type *)_buffer);
        if (tmp & CL_DEVICE_TYPE_CPU) fprintf(stdout, "%s ", "CL_DEVICE_TYPE_CPU");
        if (tmp & CL_DEVICE_TYPE_GPU) fprintf(stdout, "%s ", "CL_DEVICE_TYPE_GPU");
        if (tmp & CL_DEVICE_TYPE_ACCELERATOR) fprintf(stdout, "%s ", "CL_DEVICE_TYPE_ACCELERATOR");
        if (tmp & CL_DEVICE_TYPE_DEFAULT) fprintf(stdout, "%s ", "CL_DEVICE_TYPE_DEFAULT");
        if (tmp & CL_DEVICE_TYPE_CUSTOM) fprintf(stdout, "%s ", "CL_DEVICE_TYPE_CUSTOM");
    }
    fprintf(stdout, "\n");
    
    clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(_buffer), _buffer, NULL);
    fprintf(stdout, "     - CL_DEVICE_VENDOR:\t%s\n", _buffer);
    
    clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(_buffer), _buffer, NULL);
    fprintf(stdout, "     - CL_DEVICE_VERSION:\t%s\n", _buffer);
    
    clGetDeviceInfo(device, CL_DEVICE_PROFILE, sizeof(_buffer), _buffer, NULL);
    fprintf(stdout, "     - CL_DEVICE_PROFILE:\t%s\n", _buffer);
    
    clGetDeviceInfo(device, CL_DRIVER_VERSION, sizeof(_buffer), _buffer, NULL);
    fprintf(stdout, "     - CL_DRIVER_VERSION:\t%s\n", _buffer);
    
    fprintf(stdout, "\n");
}

/******************************************************************************************************/

/******************************************************************************************************/
void print_platform(cl_platform_id *platforms, int i) {
    // No error checking is made.
    char *param_value;
    size_t param_value_size;
    
    clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, NULL, NULL, &param_value_size);
    param_value = (char *)malloc(param_value_size);
    clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, param_value_size, param_value, NULL);
    fprintf(stdout, "  * CL_PLATFORM_NAME:\t\t\t\t%s\n", param_value);
    free(param_value);
    
    clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, NULL, NULL, &param_value_size);
    param_value = (char *)malloc(param_value_size);
    clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, param_value_size, param_value, NULL);
    fprintf(stdout, "  * CL_PLATFORM_VENDOR:\t\t\t\t%s\n", param_value);
    free(param_value);
    
    clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, NULL, NULL, &param_value_size);
    param_value = (char *)malloc(param_value_size);
    clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, param_value_size, param_value, NULL);
    fprintf(stdout, "  * CL_PLATFORM_VERSION:\t\t\t%s\n", param_value);
    free(param_value);
    
    clGetPlatformInfo(platforms[i], CL_PLATFORM_PROFILE, NULL, NULL, &param_value_size);
    param_value = (char *)malloc(param_value_size);
    clGetPlatformInfo(platforms[i], CL_PLATFORM_PROFILE, param_value_size, param_value, NULL);
    fprintf(stdout, "  * CL_PLATFORM_PROFILE:\t\t\t%s\n", param_value);
    free(param_value);
    
    clGetPlatformInfo(platforms[i], CL_PLATFORM_EXTENSIONS, NULL, NULL, &param_value_size);
    param_value = (char *)malloc(param_value_size);
    clGetPlatformInfo(platforms[i], CL_PLATFORM_EXTENSIONS, param_value_size, param_value, NULL);
    fprintf(stdout, "  * CL_PLATFORM_EXTENSIONS:\t\t\t%s\n", param_value);
    free(param_value);
}

void print_device(cl_device_id *devices, int j) {
        // No error checking is made.
#define MAX_BUFFER_SIZE 1024
    char _buffer[MAX_BUFFER_SIZE]; // Use a char buffer of enough size for convenience.
    cl_device_id device;
    
    device = devices[j];

    if (0) {
        clGetDeviceInfo(device, 0x4004, sizeof(cl_bool), _buffer, NULL);
        fprintf(stdout, "   - CL_DEVICE_GPU_OVERLAP_NV :\t\t\t\t%d\n", *((cl_bool *)_buffer));
    }
    
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(_buffer), _buffer, NULL);
    fprintf(stdout, "   - CL_DEVICE_NAME:\t\t\t\t%s\n", _buffer);
    
    clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(_buffer), _buffer, NULL);
    fprintf(stdout, "   - CL_DEVICE_TYPE:\t\t\t\t");
    {
        cl_device_type tmp = *((cl_device_type *)_buffer);
        if (tmp & CL_DEVICE_TYPE_CPU) fprintf(stdout, "%s ", "CL_DEVICE_TYPE_CPU");
        if (tmp & CL_DEVICE_TYPE_GPU) fprintf(stdout, "%s ", "CL_DEVICE_TYPE_GPU");
        if (tmp & CL_DEVICE_TYPE_ACCELERATOR) fprintf(stdout, "%s ", "CL_DEVICE_TYPE_ACCELERATOR");
        if (tmp & CL_DEVICE_TYPE_DEFAULT) fprintf(stdout, "%s ", "CL_DEVICE_TYPE_DEFAULT");
        if (tmp & CL_DEVICE_TYPE_CUSTOM) fprintf(stdout, "%s ", "CL_DEVICE_TYPE_CUSTOM");
    }
    fprintf(stdout, "\n");
    
    clGetDeviceInfo(device, CL_DEVICE_AVAILABLE, sizeof(_buffer), _buffer, NULL);
    fprintf(stdout, "   - CL_DEVICE_AVAILABLE:\t\t\t%s\n", *((cl_bool *)_buffer) == CL_TRUE ? "YES" : "NO");
    
    clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(_buffer), _buffer, NULL);
    fprintf(stdout, "   - CL_DEVICE_VENDOR:\t\t\t\t%s\n", _buffer);
    
    clGetDeviceInfo(device, CL_DEVICE_VENDOR_ID, sizeof(_buffer), _buffer, NULL);
    fprintf(stdout, "   - CL_DEVICE_VENDOR_ID:\t\t\t%u\n", *((cl_uint *)_buffer));
    
    clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(_buffer), _buffer, NULL);
    fprintf(stdout, "   - CL_DEVICE_VERSION:\t\t\t\t%s\n", _buffer);
    
    clGetDeviceInfo(device, CL_DEVICE_PROFILE, sizeof(_buffer), _buffer, NULL);
    fprintf(stdout, "   - CL_DEVICE_PROFILE:\t\t\t\t%s\n", _buffer);
    
    clGetDeviceInfo(device, CL_DRIVER_VERSION, sizeof(_buffer), _buffer, NULL);
    fprintf(stdout, "   - CL_DRIVER_VERSION:\t\t\t\t%s\n", _buffer);
    
    clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, sizeof(_buffer), _buffer, NULL);
    fprintf(stdout, "   - CL_DEVICE_EXTENSIONS:\t\t\t%s\n", _buffer);
    
    fprintf(stdout, "\n");
    
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(_buffer), _buffer, NULL);
    fprintf(stdout, "   - CL_DEVICE_MAX_COMPUTE_UNITS:\t\t\t%u\n", *((cl_uint *)_buffer));
    
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(_buffer), _buffer, NULL);
    fprintf(stdout, "   - CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS:\t%u\n", *((cl_uint *)_buffer));
    
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(_buffer), _buffer, NULL);
    printf("   - CL_DEVICE_MAX_WORK_ITEM_SIZES:\t\t\t%zu / %zu / %zu \n",
           *((size_t *) _buffer), *(((size_t *) _buffer) + 1), *(((size_t *)_buffer) + 2));
    
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(_buffer), _buffer, NULL);
    fprintf(stdout, "   - CL_DEVICE_MAX_WORK_GROUP_SIZE:\t\t\t%zu\n", *((size_t *)_buffer));
    
    clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(_buffer), _buffer, NULL);
    fprintf(stdout, "   - CL_DEVICE_MAX_CLOCK_FREQUENCY:\t\t\t%u MHz\n", *((cl_uint *)_buffer));
    
    clGetDeviceInfo(device, CL_DEVICE_ADDRESS_BITS, sizeof(_buffer), _buffer, NULL);
    fprintf(stdout, "   - CL_DEVICE_ADDRESS_BITS:\t\t\t\t%u\n", *((cl_uint *)_buffer));
    
    clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(_buffer), _buffer, NULL);
    fprintf(stdout, "   - CL_DEVICE_MAX_MEM_ALLOC_SIZE:\t\t\t%llu MBytes\n", *((cl_ulong *)_buffer) >> 20);
    
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(_buffer), _buffer, NULL);
    fprintf(stdout, "   - CL_DEVICE_GLOBAL_MEM_SIZE:\t\t\t\t%llu MBytes\n", *((cl_ulong *)_buffer) >> 20);
    
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, sizeof(_buffer), _buffer, NULL);
    fprintf(stdout, "   - CL_DEVICE_GLOBAL_MEM_CACHE_TYPE:\t\t");
    switch (*((cl_device_mem_cache_type *)_buffer)) {
        case CL_NONE:
            fprintf(stdout, "CL_NONE\n");
            break;
        case CL_READ_ONLY_CACHE:
            fprintf(stdout, "CL_READ_ONLY_CACHE\n");
            break;
        case CL_READ_WRITE_CACHE:
            fprintf(stdout, "CL_READ_WRITE_CACHE\n");
            break;
    }
    
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(_buffer), _buffer, NULL);
    fprintf(stdout, "   - CL_DEVICE_GLOBAL_MEM_CACHE_SIZE:\t\t%lluKBytes\n", *((cl_ulong *)_buffer) >> 10);
    
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(_buffer), _buffer, NULL);
    fprintf(stdout, "   - CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE:\t%u Bytes\n", *((cl_int *)_buffer));
    
    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(_buffer), _buffer, NULL);
    fprintf(stdout, "   - CL_DEVICE_LOCAL_MEM_TYPE:\t\t\t\t%s\n", *((cl_device_local_mem_type *)_buffer) == CL_LOCAL ? "LOCAL" : "GLOBAL");
    
    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(_buffer), _buffer, NULL);
    fprintf(stdout, "   - CL_DEVICE_LOCAL_MEM_SIZE:\t\t\t\t%llu KByte\n", *((cl_ulong *)_buffer) >> 10);
    
    clGetDeviceInfo(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(_buffer), _buffer, NULL);
    fprintf(stdout, "   - CL_DEVICE_MAX_CONSTANT__buffer_SIZE:\t%llu MBytes\n", *((cl_ulong *)_buffer) >> 20);
    
    clGetDeviceInfo(device, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(_buffer), _buffer, NULL);
    fprintf(stdout, "   - CL_DEVICE_MEM_BASE_ADDR_ALIGN:\t\t\t%u Bits\n", *((cl_uint *)_buffer));
    
    clGetDeviceInfo(device, CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, sizeof(_buffer), _buffer, NULL);
    fprintf(stdout, "   - CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE:\t%u Bytes\n", *((cl_uint *)_buffer));
    
    fprintf(stdout, "\n");
    
    clGetDeviceInfo(device, CL_DEVICE_EXECUTION_CAPABILITIES, sizeof(_buffer), _buffer, NULL);
    fprintf(stdout, "   - CL_DEVICE_EXECUTION_CAPABILITIES:\t\t");
    {
        cl_device_exec_capabilities tmp = *((cl_device_exec_capabilities *)_buffer);
        if (tmp & CL_EXEC_KERNEL) fprintf(stdout, "%s ", "CL_EXEC_KERNEL");
        if (tmp & CL_EXEC_NATIVE_KERNEL) fprintf(stdout, "%s ", "CL_EXEC_NATIVE_KERNEL");
    }
    fprintf(stdout, "\n");
    
    clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES, sizeof(_buffer), _buffer, NULL);
    fprintf(stdout, "   - CL_DEVICE_QUEUE_PROPERTIES:\t\t\t");
    {
        cl_command_queue_properties tmp = *((cl_command_queue_properties *)_buffer);
        if (tmp & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
            fprintf(stdout, "%s ", "CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE");
        if (tmp & CL_QUEUE_PROFILING_ENABLE)
            fprintf(stdout, "%s ", "CL_QUEUE_PROFILING_ENABLE");
    }
    fprintf(stdout, "\n");
    
    clGetDeviceInfo(device, CL_DEVICE_ERROR_CORRECTION_SUPPORT, sizeof(_buffer), _buffer, NULL);
    fprintf(stdout, "   - CL_DEVICE_ERROR_CORRECTION_SUPPORT:\t%s\n", *((cl_bool *)_buffer) == CL_TRUE ? "YES" : "NO");
    
    clGetDeviceInfo(device, CL_DEVICE_ENDIAN_LITTLE, sizeof(_buffer), _buffer, NULL);
    fprintf(stdout, "   - CL_DEVICE_ENDIAN_LITTLE:\t\t\t\t%s\n", *((cl_bool *)_buffer) == CL_TRUE ? "YES" : "NO");
    
    clGetDeviceInfo(device, CL_DEVICE_COMPILER_AVAILABLE, sizeof(_buffer), _buffer, NULL);
    fprintf(stdout, "   - CL_DEVICE_COMPILER_AVAILABLE:\t\t\t%s\n", *((cl_bool *)_buffer) == CL_TRUE ? "YES" : "NO");
    
    clGetDeviceInfo(device, CL_DEVICE_PROFILING_TIMER_RESOLUTION, sizeof(_buffer), _buffer, NULL);
    fprintf(stdout, "   - CL_DEVICE_PROFILING_TIMER_RESOLUTION:\t%zu nanosecond(s)\n", *((size_t *)_buffer));
    
    clGetDeviceInfo(device, CL_DEVICE_MAX_PARAMETER_SIZE, sizeof(_buffer), _buffer, NULL);
    fprintf(stdout, "   - CL_DEVICE_MAX_PARAMETER_SIZE:\t\t\t%zu Bytes\n", *((size_t *)_buffer));
    
    clGetDeviceInfo(device, CL_DEVICE_MAX_CONSTANT_ARGS, sizeof(_buffer), _buffer, NULL);
    fprintf(stdout, "   - CL_DEVICE_MAX_CONSTANT_ARGS:\t\t\t%u\n", *((cl_uint *)_buffer));
    
    fprintf(stdout, "\n");
    
    clGetDeviceInfo(device, CL_DEVICE_IMAGE_SUPPORT, sizeof(_buffer), _buffer, NULL);
    fprintf(stdout, "   - CL_DEVICE_IMAGE_SUPPORT:\t\t\t\t%s\n", *((cl_bool *)_buffer) == CL_TRUE ? "YES" : "NO");
    
    clGetDeviceInfo(device, CL_DEVICE_MAX_SAMPLERS, sizeof(_buffer), _buffer, NULL);
    fprintf(stdout, "   - CL_DEVICE_MAX_SAMPLERS:\t\t\t\t%u\n", *((cl_uint *)_buffer));
    
    clGetDeviceInfo(device, CL_DEVICE_MAX_READ_IMAGE_ARGS, sizeof(_buffer), _buffer, NULL);
    fprintf(stdout, "   - CL_DEVICE_MAX_READ_IMAGE_ARGS:\t\t\t%u\n", *((cl_uint *)_buffer));
    
    clGetDeviceInfo(device, CL_DEVICE_MAX_WRITE_IMAGE_ARGS, sizeof(_buffer), _buffer, NULL);
    fprintf(stdout, "   - CL_DEVICE_MAX_WRITE_IMAGE_ARGS:\t\t%u\n", *((cl_uint *)_buffer));
    
    clGetDeviceInfo(device, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(_buffer), _buffer, NULL);
    fprintf(stdout, "   - CL_DEVICE_IMAGE2D_MAX_WIDTH:\t\t\t%zu\n", *((size_t *)_buffer));
    
    clGetDeviceInfo(device, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(_buffer), _buffer, NULL);
    fprintf(stdout, "   - CL_DEVICE_IMAGE2D_MAX_HEIGHT:\t\t\t%zu\n", *((size_t *)_buffer));
    
    clGetDeviceInfo(device, CL_DEVICE_IMAGE3D_MAX_WIDTH, sizeof(_buffer), _buffer, NULL);
    fprintf(stdout, "   - CL_DEVICE_IMAGE3D_MAX_WIDTH:\t\t\t%zu\n", *((size_t *)_buffer));
    
    clGetDeviceInfo(device, CL_DEVICE_IMAGE3D_MAX_HEIGHT, sizeof(_buffer), _buffer, NULL);
    fprintf(stdout, "   - CL_DEVICE_IMAGE3D_MAX_HEIGHT:\t\t\t%zu\n", *((size_t *)_buffer));
    
    clGetDeviceInfo(device, CL_DEVICE_IMAGE3D_MAX_DEPTH, sizeof(_buffer), _buffer, NULL);
    fprintf(stdout, "   - CL_DEVICE_IMAGE3D_MAX_DEPTH:\t\t\t%zu\n", *((size_t *)_buffer));
    
    fprintf(stdout, "\n");
    
    clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, sizeof(_buffer), _buffer, NULL);
    fprintf(stdout, "   - CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR:\t\t%u\n", *((cl_uint *)_buffer));
    
    clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, sizeof(_buffer), _buffer, NULL);
    fprintf(stdout, "   - CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT:\t%u\n", *((cl_uint *)_buffer));
    
    clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, sizeof(_buffer), _buffer, NULL);
    fprintf(stdout, "   - CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT:\t\t%u\n", *((cl_uint *)_buffer));
    
    clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, sizeof(_buffer), _buffer, NULL);
    fprintf(stdout, "   - CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG:\t\t%u\n", *((cl_uint *)_buffer));
    
    clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, sizeof(_buffer), _buffer, NULL);
    fprintf(stdout, "   - CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT:\t%u\n", *((cl_uint *)_buffer));
    
    clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, sizeof(_buffer), _buffer, NULL);
    fprintf(stdout, "   - CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE:\t%u\n", *((cl_uint *)_buffer));
    
    fprintf(stdout, "\n");
    
    clGetDeviceInfo(device, CL_DEVICE_SINGLE_FP_CONFIG, sizeof(_buffer), _buffer, NULL);
    fprintf(stdout, "   - CL_DEVICE_SINGLE_FP_CONFIG:\t\t");
    {
        cl_device_fp_config tmp = *((cl_device_fp_config *)_buffer);
        if (tmp & CL_FP_DENORM) fprintf(stdout, "%s ", "CL_FP_DENORM");
        if (tmp & CL_FP_INF_NAN) fprintf(stdout, "%s ", "CL_FP_INF_NAN");
        if (tmp & CL_FP_ROUND_TO_NEAREST) fprintf(stdout, "%s ", "CL_FP_ROUND_TO_NEAREST");
        if (tmp & CL_FP_ROUND_TO_ZERO) fprintf(stdout, "%s ", "CL_FP_ROUND_TO_ZERO");
        if (tmp & CL_FP_ROUND_TO_INF) fprintf(stdout, "%s ", "CL_FP_ROUND_TO_INF");
        if (tmp & CL_FP_FMA) fprintf(stdout, "%s ", "CL_FP_FMA");
    }
    fprintf(stdout, "\n");
    /*
     clGetDeviceInfo(device, CL_DEVICE_HALF_FP_CONFIG, sizeof(_buffer), _buffer, NULL);
     fprintf(stdout, "   - CL_DEVICE_SINGLE_FP_CONFIG:\t\t");
     {
     cl_device_fp_config tmp = *((cl_device_fp_config *)_buffer);
     if (tmp & CL_FP_DENORM) fprintf(stdout, "%s ", "CL_FP_DENORM");
     if (tmp & CL_FP_INF_NAN) fprintf(stdout, "%s ", "CL_FP_INF_NAN");
     if (tmp & CL_FP_ROUND_TO_NEAREST) fprintf(stdout, "%s ", "CL_FP_ROUND_TO_NEAREST");
     if (tmp & CL_FP_ROUND_TO_ZERO) fprintf(stdout, "%s ", "CL_FP_ROUND_TO_ZERO");
     if (tmp & CL_FP_ROUND_TO_INF) fprintf(stdout, "%s ", "CL_FP_ROUND_TO_INF");
     if (tmp & CL_FP_FMA) fprintf(stdout, "%s ", "CL_FP_FMA");
     }
     fprintf(stdout, "\n");
     */
    
    clGetDeviceInfo(device, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(_buffer), _buffer, NULL);
    fprintf(stdout, "   - CL_DEVICE_DOUBLE_FP_CONFIG:\t\t");
    {
        cl_device_fp_config tmp = *((cl_device_fp_config *)_buffer);
        if (tmp & CL_FP_DENORM) fprintf(stdout, "%s ", "CL_FP_DENORM");
        if (tmp & CL_FP_INF_NAN) fprintf(stdout, "%s ", "CL_FP_INF_NAN");
        if (tmp & CL_FP_ROUND_TO_NEAREST) fprintf(stdout, "%s ", "CL_FP_ROUND_TO_NEAREST");
        if (tmp & CL_FP_ROUND_TO_ZERO) fprintf(stdout, "%s ", "CL_FP_ROUND_TO_ZERO");
        if (tmp & CL_FP_ROUND_TO_INF) fprintf(stdout, "%s ", "CL_FP_ROUND_TO_INF");
        if (tmp & CL_FP_FMA) fprintf(stdout, "%s ", "CL_FP_FMA");
    }
    fprintf(stdout, "\n");
}

void print_devices(cl_platform_id *platforms, int i) {
    // Assurm there are no errors in calling cl functions.
    cl_uint n_devices;
    cl_device_id *devices;
    cl_int errcode_ret;
    
    errcode_ret = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, NULL, NULL, &n_devices);
    CHECK_ERROR_CODE(errcode_ret);
    
    devices = (cl_device_id *)malloc(sizeof(cl_device_id) * n_devices);
    errcode_ret = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, n_devices, devices, NULL);
    CHECK_ERROR_CODE(errcode_ret);
    
    for (int j = 0; j < (int) n_devices; j++) {
        fprintf(stdout, "----- [Begin of Device %d(%d) of Platform %d] ----------------------------------------------------------------------\n\n",
                j, n_devices, i);
        print_device(devices, j);
        fprintf(stdout, "\n----- [End of Device %d(%d) of Platform %d] ------------------------------------------------------------------------\n\n",
                j, n_devices, i);
    }
    free(devices);
}

int show_OpenCL_platform(void) {
    cl_uint n_platforms;
    cl_platform_id *platforms;
    cl_int errcode_ret;
    
    errcode_ret = clGetPlatformIDs(NULL, NULL, &n_platforms);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;

    platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * n_platforms);
    
    errcode_ret = clGetPlatformIDs(n_platforms, platforms, NULL);
    if (CHECK_ERROR_CODE(errcode_ret)) return 1;
    
    fprintf(stdout, "\n");
    for (int i = 0; i < (int) n_platforms; i++) {
        fprintf(stdout, "===== [Begin of Platform %d(%d)] ========================================================================================\n\n",
                i, n_platforms);
        print_platform(platforms, i);
        fprintf(stdout, "\n");
        print_devices(platforms, i);
        fprintf(stdout, "===== [End of Platform %d(%d)] ==========================================================================================\n\n",
                i, n_platforms);
    }
    free(platforms);

    return 0;
}
/******************************************************************************************************/

/******************************************************************************************************/
void printf_KernelWorkGroupInfo(cl_kernel kernel, cl_device_id device) {
    // Assurm there are no errors in calling cl functions.
    cl_int errcode_ret;
    size_t tmp_size[3];
    cl_ulong tmp_ulong = 0;

    fprintf(stdout, "\n");

    errcode_ret = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                                           sizeof(size_t), (void *)tmp_size, NULL);
    CHECK_ERROR_CODE(errcode_ret);
    fprintf(stdout,    "   # The preferred multiple of workgroup size for launch (hint) is %zu.\n", tmp_size[0]);
    
    errcode_ret = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE,
                                       sizeof(size_t), (void *)tmp_size, NULL);
    CHECK_ERROR_CODE(errcode_ret);
    fprintf(stdout,    "   # The maximum work-group size that can be used to execute this kernel on this device is %zu.\n", tmp_size[0]);
    
    errcode_ret = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_LOCAL_MEM_SIZE,
                                           sizeof(cl_ulong), (void *)tmp_ulong, NULL);
    CHECK_ERROR_CODE(errcode_ret);
    fprintf(stdout,    "   # The amount of local memory in bytes being used by this kernel is %llu.\n", tmp_ulong);
    
    errcode_ret = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_PRIVATE_MEM_SIZE,
                                           sizeof(cl_ulong), (void *)tmp_ulong, NULL);
    CHECK_ERROR_CODE(errcode_ret);
    fprintf(stdout,    "   # The minimum amount of private memory in bytes used by each workitem in the kernel is %llu.\n", tmp_ulong);
    
    errcode_ret = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_COMPILE_WORK_GROUP_SIZE,
                                           sizeof(size_t)*3, (void *)tmp_size, NULL);
    CHECK_ERROR_CODE(errcode_ret);
    fprintf(stdout,    "   # The work-group size specified by the __attribute__((reqd_work_group_size(X, Y, Z))) qualifier is (%zu, %zu, %zu).\n",
            tmp_size[0], tmp_size[1], tmp_size[2]);
    
    fprintf(stdout, "\n");
}
/******************************************************************************************************/

int is_OpenCL_Extension_supported(const char* wanted_extension, char* extension_list) {
    char cur_extension[256];
    int extension_list_index = 0, cur_extension_index;

    do {
        cur_extension_index = 0;
        while (extension_list[extension_list_index] == ' ')
            extension_list_index++;
        if (extension_list[extension_list_index] == '\0')
            break;
        do {
            cur_extension[cur_extension_index++] = extension_list[extension_list_index++];
        } while ((extension_list[extension_list_index] != ' ') && (extension_list[extension_list_index] != '\0'));
        cur_extension[cur_extension_index] = '\0';
        if (!strcmp(wanted_extension, cur_extension))
            return 1;
    } while (extension_list[extension_list_index] != '\0');
    return 0;
}
/******************************************************************************************************/
#define     QUEUED_TO_END         0
#define     SUBMIT_TO_END          1
#define     START_TO_END            2

static float event_time[3];

void util_reset_event_time(void) {
    event_time[0] = event_time[1] = event_time[2] = 0.0f;
}

void util_accumulate_event_times_1_2(cl_event event) {
    // Consider to use CL_PROFILING_COMMAND_END for OpenCL 2.0 or higher.
    cl_ulong time_elapsed;

    compute_elapsed_time(event, &time_elapsed, CL_PROFILING_COMMAND_QUEUED, CL_PROFILING_COMMAND_END);
    event_time[QUEUED_TO_END] += time_elapsed * 1.0e-6f;
    compute_elapsed_time(event, &time_elapsed, CL_PROFILING_COMMAND_SUBMIT, CL_PROFILING_COMMAND_END);
    event_time[SUBMIT_TO_END] += time_elapsed * 1.0e-6f;
    compute_elapsed_time(event, &time_elapsed, CL_PROFILING_COMMAND_START, CL_PROFILING_COMMAND_END);
    event_time[START_TO_END] += time_elapsed * 1.0e-6f;
}

void util_print_accumulated_device_time_1_2(int n_executions) {
    event_time[0] /= (float)n_executions; 
    event_time[1] /= (float)n_executions; 
    event_time[2] /= (float)n_executions;

    fprintf(stdout, "      * Average time by device clock (No. of executions = %2d):\n", n_executions);
    fprintf(stdout, "        - From QUEUED to END = %.3fms\n",    event_time[QUEUED_TO_END]);
    fprintf(stdout, "        - From SUBMIT to END = %.3fms\n",     event_time[SUBMIT_TO_END]);
    fprintf(stdout, "        - From START to END = %.3fms\n\n",   event_time[START_TO_END]);
}

/******************************************************************************************************/

FILE* util_open_stat_file_append(const char* filename) {
    FILE* fp = fopen(filename, "a");
    if (fp == NULL) {
        fprintf(stderr, "^^^ Error: cannot open file %s for appending...\n", filename);
        exit(-1);
    }
    return fp;
}

void util_close_stat_file_append(FILE* fp) {
    fclose(fp);
}

/******************************************************************************************************/
static char current_time[32];

char* util_get_current_time(void) {
    time_t t = time(NULL);
    struct tm tm = *localtime(&t);

    sprintf(current_time, "%4d-%2d-%2d %2d:%2d:%2d",
        tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
        tm.tm_hour, tm.tm_min, tm.tm_sec);

    return current_time;
}

/******************************************************************************************************/
static char device_name[256];
 
char* util_get_device_name(cl_device_id device) {
    clGetDeviceInfo(device, CL_DEVICE_NAME, 128, device_name, NULL);

    return device_name;
}

/******************************************************************************************************/
void util_stamp_stat_file_device_name_and_time(FILE* fp, cl_device_id device_id) {
    fprintf(fp, "\n*** DEVICE (%s)", util_get_device_name(device_id));
    fprintf(fp, " (%s)\n", util_get_current_time());
}

void util_stamp_stat_file_ave_device_time_START_to_END_1_2(FILE* fp, int kernel_selection, int n_executions) {
    fprintf(fp, "--- KERNEL = %2d,  N_EXECUTIONS = %2d,  Ave_time(S_to_E): %.3fms\n",
        kernel_selection, n_executions, event_time[START_TO_END]);
}

void util_stamp_stat_file_ave_device_time_START_to_END_1_2_string(FILE* fp, char* string) {
    fprintf(fp, "--- %s, Ave_time(S_to_E): %.3fms\n", string, event_time[START_TO_END]);
}

void util_stamp_stat_file_string_and_time(FILE* fp, char* string) {
    fprintf(fp, "%s (%s)\n", string, util_get_current_time());
}

/******************************************************************************************************/

void util_set_seed_for_rand(unsigned int seed) {
    if (seed == 0)
        srand((unsigned int) time(NULL));
    else
        srand(seed);
}

void util_generate_random_float_array(float* array, int n) {
    // Assume sufficient memory space has been allocated.
    for (int i = 0; i < n; i++) {
        array[i] = (float)rand() / RAND_MAX;
    }
}

void util_compair_two_float_arrays(float* A, float* B, int n_elements, int dump_discrepancies) {
    int n_differs = 0;

    for (int i = 0; i < n_elements; i++) {
        if ((double)fabs(A[i] - B[i]) / fabs(A[i]) > 0.00001) { // Assume A[i] is not zero.
            n_differs++;
            if (dump_discrepancies) // for debugging
                fprintf(stdout, "   %f(A[%d]) != %f(B[%d])\n", A[i], i, B[i], i);
        }
    }
    if (n_differs == 0)
        fprintf(stdout, "      * The two arrays are the same.\n");
    else
        fprintf(stdout, "      * There are %d discrepancies between the two arrays.\n", n_differs);
}
/******************************************************************************************************/

void util_insert_commas_in_timestamp(cl_ulong n, char* out) {
    // modified from a public code.
    int c;
    char buf[256], * p;

    sprintf(buf, "%llu", n);
    c = 2 - strlen(buf) % 3;
    for (p = buf; *p != 0; p++) {
        *out++ = *p;
        if (c == 1) {
            *out++ = ',';
        }
        c = (c + 1) % 3;
    }
    *--out = 0;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
