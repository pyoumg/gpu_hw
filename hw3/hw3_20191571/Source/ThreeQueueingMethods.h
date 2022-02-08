#pragma once
//
//  ThreeQueueingMethods.h
//
//  Written for CSEG437_CSE5437
//  Department of Computer Science and Engineering
//  Copyright © 2021 Sogang University. All rights reserved.
//


/////////////////////////////////////////////////////////////////////////////////////////////////////////
///// 1. use_multiple_segments_and_three_command_queues_with_events_breadth //////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////
void use_multiple_segments_and_three_command_queues_with_events_breadth(int n_segments) {
    context.global_work_size[0] = context.image_width; //x
    if ((context.image_height / n_segments) % context.local_work_size[1] != 0) {
        printf("invalid local work size\n");
        return;
    }
    int height = context.image_height / n_segments; 
    int image_size = context.image_height * context.image_width;
    int start_idx = 0; //offset
    int size = 0;
    BYTE last_flag = 0;
    /* Set the kenel arguments for kernels. */

    for (int j = 0; j < n_segments; j++) {
        if (j == 0) 
            start_idx = 0;
        else 
            start_idx = (height * j - context.local_work_size[1]) * context.image_width * 4;

        if (j == n_segments - 1)
            last_flag = 1;
        else
            last_flag = 0;

        int read_idx = j * context.image_data_bytes / n_segments;
        errcode_ret = clSetKernelArg(context.kernel[j], 0, sizeof(cl_mem), (void*)&context.buffer_input_dev);
        if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);
        errcode_ret |= clSetKernelArg(context.kernel[j], 1, sizeof(cl_mem), (void*)&context.buffer_output_dev);
        if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);
        errcode_ret |= clSetKernelArg(context.kernel[j], 2, sizeof(UINT), &context.image_width);
        if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);
        errcode_ret |= clSetKernelArg(context.kernel[j], 3, sizeof(UINT), &context.image_height);
        if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);
        errcode_ret |= clSetKernelArg(context.kernel[j], 4, sizeof(cl_mem), &context.sobel_x);
        if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);
        errcode_ret |= clSetKernelArg(context.kernel[j], 5, sizeof(cl_mem), &context.sobel_y);
        if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);
        int twice_half_filter_width = 4;
        size_t local_mem_size = sizeof(cl_uchar)
            * (context.local_work_size[0] + twice_half_filter_width)
            * (context.local_work_size[1] + twice_half_filter_width);
        errcode_ret |= clSetKernelArg(context.kernel[j], 6, local_mem_size, NULL);

        errcode_ret = clSetKernelArg(context.kernel[j], 7, sizeof(unsigned int), (void*)&start_idx);
        if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);
        errcode_ret |= clSetKernelArg(context.kernel[j], 8, sizeof(BYTE), (void*)&last_flag);
        if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);
        errcode_ret = clSetKernelArg(context.kernel[j], 9, sizeof(int), (void*)&context.n_kernel_loop_iterations);
        if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);
    }

    /* Make sure that all previous commands have finished. */
    for (int j = 0; j < 3; j++) {
        clFinish(context.cmd_queue[j]);
    }

    total_time = 0.0f;
    for (int i = 0; i < context.n_kernel_call_iterations; i++) {
        initialize_buffers_for_kernel_execution();

        CHECK_TIME_START(_start, _freq);
        for (int j = 0; j < n_segments; j++) {
            // Move the input data from the host memory to the GPU device memory.
            
            if (n_segments >= 2) {
                if (j == 0) {
                    start_idx = 0;
                    size = sizeof(BYTE) * (height + context.local_work_size[1]) * context.image_width * 4;
                    context.global_work_size[1] = height + context.local_work_size[1];
 
                }
                else if (j == n_segments - 1) {
                    start_idx = (height * j - context.local_work_size[1]) * context.image_width * 4;
                    size = sizeof(BYTE) * (height + context.local_work_size[1]) * context.image_width * 4;
                    context.global_work_size[1] = height + context.local_work_size[1];
                }
                else {
                    start_idx = (height * j - context.local_work_size[1]) * context.image_width * 4;
                    size = sizeof(BYTE) * (height + 2 * context.local_work_size[1]) * context.image_width * 4;
                    context.global_work_size[1] = height + 2 * context.local_work_size[1];
                }
            }
            else if (n_segments == 1) {
                start_idx = 0;
                size = sizeof(BYTE) * (height) * context.image_width * 4;//rgba
                context.global_work_size[1] = height ;
            }

            errcode_ret = clEnqueueWriteBuffer(context.cmd_queue[0], context.sobel_x, CL_FALSE, 0,
                sizeof(float) * 25,
                Sobel_x, 0, NULL, NULL);
            if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);
            errcode_ret = clEnqueueWriteBuffer(context.cmd_queue[0], context.sobel_y, CL_FALSE, 0,
                sizeof(float) * 25,
                Sobel_y, 0, NULL, NULL);
            if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);

            errcode_ret = clEnqueueWriteBuffer(context.cmd_queue[0], context.buffer_input_dev, CL_FALSE, start_idx * sizeof(BYTE),size, 
                (void*)&context.data_input[start_idx], 0, NULL, &context.event_write_input[j]);
            if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);

            clFlush(context.cmd_queue[0]);

            /* Execute the kernel on the device. */
            errcode_ret = clEnqueueNDRangeKernel(context.cmd_queue[1], context.kernel[j], 2, NULL,
                context.global_work_size, context.local_work_size, 1, &context.event_write_input[j], &context.event_compute[j]);
            if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);
            clFlush(context.cmd_queue[1]);

            int read_idx = j * context.image_data_bytes / n_segments;

            /* Read back the device buffer to the host array. */
#if COPY_COMPUTE_COMMANDS_PROFILING  
            errcode_ret = clEnqueueReadBuffer(context.cmd_queue[2], context.buffer_output_dev, CL_FALSE, read_idx*sizeof(BYTE),
                sizeof(BYTE) *context.image_data_bytes/n_segments, (void*)&context.data_output[read_idx], 1, &context.event_compute[j], &context.event_read_output[j]);
            if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);
#else
            errcode_ret = clEnqueueReadBuffer(context.cmd_queue[2], context.buffer_output_dev, CL_FALSE, read_idx * sizeof(BYTE),
                sizeof(BYTE) * context.image_data_bytes / n_segments, (void*)&context.data_output[read_idx], 1, &context.event_compute[j],NULL);
            if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);
#endif
            clFlush(context.cmd_queue[2]);
        }
        for (int j = 0; j < 3; j++) {
            clFinish(context.cmd_queue[j]);
        }
        CHECK_TIME_END(_start, _end, _freq, compute_time);
        total_time += compute_time;
    }
    fprintf(stdout, "      * Time by host clock = %.3fms\n", total_time / context.n_kernel_call_iterations);

#if COPY_COMPUTE_COMMANDS_PROFILING
    // For the last execution
    print_time_statistics_for_copy_and_compute(n_segments);
#endif


}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///// 2. void use_multiple_segments_and_multiple_command_queues_depth(int n_cmd_queues) /////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void use_multiple_segments_and_multiple_command_queues_depth(int n_cmd_queues) {
    context.global_work_size[0] = context.image_width; //x
    if ((context.image_height / n_cmd_queues) % context.local_work_size[1] != 0) {
        printf("invalid local work size\n");
        return;
    }
    int height = context.image_height / n_cmd_queues;
    int image_size = context.image_height * context.image_width;
    int start_idx = 0; //offset
    int size = 0;
    BYTE last_flag = 0;
    /* Set the kenel arguments for kernels. */

    for (int j = 0; j < n_cmd_queues; j++) {
        if (j == 0)
            start_idx = 0;
        else
            start_idx = (height * j - context.local_work_size[1]) * context.image_width * 4;

        if (j == n_cmd_queues - 1)
            last_flag = 1;
        else
            last_flag = 0;

        int read_idx = j * context.image_data_bytes / n_cmd_queues;
        errcode_ret = clSetKernelArg(context.kernel[j], 0, sizeof(cl_mem), (void*)&context.buffer_input_dev);
        if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);
        errcode_ret |= clSetKernelArg(context.kernel[j], 1, sizeof(cl_mem), (void*)&context.buffer_output_dev);
        if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);
        errcode_ret |= clSetKernelArg(context.kernel[j], 2, sizeof(UINT), &context.image_width);
        if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);
        errcode_ret |= clSetKernelArg(context.kernel[j], 3, sizeof(UINT), &context.image_height);
        if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);
        errcode_ret |= clSetKernelArg(context.kernel[j], 4, sizeof(cl_mem), &context.sobel_x);
        if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);
        errcode_ret |= clSetKernelArg(context.kernel[j], 5, sizeof(cl_mem), &context.sobel_y);
        if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);
        int twice_half_filter_width = 4;
        size_t local_mem_size = sizeof(cl_uchar)
            * (context.local_work_size[0] + twice_half_filter_width)
            * (context.local_work_size[1] + twice_half_filter_width);
        errcode_ret |= clSetKernelArg(context.kernel[j], 6, local_mem_size, NULL);

        errcode_ret = clSetKernelArg(context.kernel[j], 7, sizeof(unsigned int), (void*)&start_idx);
        if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);
        errcode_ret |= clSetKernelArg(context.kernel[j], 8, sizeof(BYTE), (void*)&last_flag);
        if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);
        errcode_ret = clSetKernelArg(context.kernel[j], 9, sizeof(int), (void*)&context.n_kernel_loop_iterations);
        if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);
    }

    /* Make sure that all previous commands have finished. */
    for (int j = 0; j < n_cmd_queues; j++) {
        clFinish(context.cmd_queue[j]);
    }

    total_time = 0.0f;
    for (int i = 0; i < context.n_kernel_call_iterations; i++) {
        initialize_buffers_for_kernel_execution();

        CHECK_TIME_START(_start, _freq);
#if COPY_COMPUTE_COMMANDS_PROFILING
        for (int j = 0; j < n_cmd_queues; j++) {
            // Move the input data from the host memory to the GPU device memory.

            if (n_cmd_queues >= 2) {
                if (j == 0) {
                    start_idx = 0;
                    size = sizeof(BYTE) * (height + context.local_work_size[1]) * context.image_width * 4;
                    context.global_work_size[1] = height + context.local_work_size[1];

                }
                else if (j == n_cmd_queues - 1) {
                    start_idx = (height * j - context.local_work_size[1]) * context.image_width * 4;
                    size = sizeof(BYTE) * (height + context.local_work_size[1]) * context.image_width * 4;
                    context.global_work_size[1] = height + context.local_work_size[1];
                }
                else {
                    start_idx = (height * j - context.local_work_size[1]) * context.image_width * 4;
                    size = sizeof(BYTE) * (height + 2 * context.local_work_size[1]) * context.image_width * 4;
                    context.global_work_size[1] = height + 2 * context.local_work_size[1];
                }
            }
            else if (n_cmd_queues == 1) {
                start_idx = 0;
                size = sizeof(BYTE) * (height)*context.image_width * 4;//rgba
                context.global_work_size[1] = height;
            }

            errcode_ret = clEnqueueWriteBuffer(context.cmd_queue[j], context.sobel_x, CL_FALSE, 0,
                sizeof(float) * 25,
                Sobel_x, 0, NULL, NULL);
            if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);
            errcode_ret = clEnqueueWriteBuffer(context.cmd_queue[j], context.sobel_y, CL_FALSE, 0,
                sizeof(float) * 25,
                Sobel_y, 0, NULL, NULL);
            if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);

            if (j == 0) {
                errcode_ret = clEnqueueWriteBuffer(context.cmd_queue[j], context.buffer_input_dev, CL_FALSE, start_idx * sizeof(BYTE), size,
                    (void*)&context.data_input[start_idx], 0, NULL, &context.event_write_input[j]);
            }
            else {
                errcode_ret = clEnqueueWriteBuffer(context.cmd_queue[j], context.buffer_input_dev, CL_FALSE, start_idx * sizeof(BYTE), size,
                    (void*)&context.data_input[start_idx], 1, &context.event_write_input[j-1], &context.event_write_input[j]);
            }
            if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);
            clFlush(context.cmd_queue[j]);

            if (j == 0) {
                /* Execute the kernel on the device. */
                errcode_ret = clEnqueueNDRangeKernel(context.cmd_queue[j], context.kernel[j], 2, NULL,
                    context.global_work_size, context.local_work_size, 0, NULL, &context.event_compute[j]);
            }
            else {
                errcode_ret = clEnqueueNDRangeKernel(context.cmd_queue[j], context.kernel[j], 2, NULL,
                    context.global_work_size, context.local_work_size, 1, &context.event_compute[j-1], &context.event_compute[j]);
            }
            
            if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);
            clFlush(context.cmd_queue[j]);

            int read_idx = j * context.image_data_bytes / n_cmd_queues;

            /* Read back the device buffer to the host array. */
            if (j == 0) {
                errcode_ret = clEnqueueReadBuffer(context.cmd_queue[j], context.buffer_output_dev, CL_FALSE, read_idx * sizeof(BYTE),
                    sizeof(BYTE) * context.image_data_bytes / n_cmd_queues, (void*)&context.data_output[read_idx], 0, NULL, &context.event_read_output[j]);
            }
            else {
                errcode_ret = clEnqueueReadBuffer(context.cmd_queue[j], context.buffer_output_dev, CL_FALSE, read_idx * sizeof(BYTE),
                    sizeof(BYTE) * context.image_data_bytes / n_cmd_queues, (void*)&context.data_output[read_idx], 1, &context.event_read_output[j-1], &context.event_read_output[j]);
            }
            if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);

            clFlush(context.cmd_queue[j]);
        }
        for (int j = 0; j < n_cmd_queues; j++) {
            clFinish(context.cmd_queue[j]);
        }
#else
        for (int j = 0; j < n_cmd_queues; j++) {
            // Move the input data from the host memory to the GPU device memory.

            if (n_cmd_queues >= 2) {
                if (j == 0) {
                    start_idx = 0;
                    size = sizeof(BYTE) * (height + context.local_work_size[1]) * context.image_width * 4;
                    context.global_work_size[1] = height + context.local_work_size[1];

                }
                else if (j == n_cmd_queues - 1) {
                    start_idx = (height * j - context.local_work_size[1]) * context.image_width * 4;
                    size = sizeof(BYTE) * (height + context.local_work_size[1]) * context.image_width * 4;
                    context.global_work_size[1] = height + context.local_work_size[1];
                }
                else {
                    start_idx = (height * j - context.local_work_size[1]) * context.image_width * 4;
                    size = sizeof(BYTE) * (height + 2 * context.local_work_size[1]) * context.image_width * 4;
                    context.global_work_size[1] = height + 2 * context.local_work_size[1];
                }
            }
            else if (n_cmd_queues == 1) {
                start_idx = 0;
                size = sizeof(BYTE) * (height)*context.image_width * 4;//rgba
                context.global_work_size[1] = height;
            }

            errcode_ret = clEnqueueWriteBuffer(context.cmd_queue[j], context.sobel_x, CL_FALSE, 0,
                sizeof(float) * 25,
                Sobel_x, 0, NULL, NULL);
            if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);
            errcode_ret = clEnqueueWriteBuffer(context.cmd_queue[j], context.sobel_y, CL_FALSE, 0,
                sizeof(float) * 25,
                Sobel_y, 0, NULL, NULL);
            if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);

            errcode_ret = clEnqueueWriteBuffer(context.cmd_queue[j], context.buffer_input_dev, CL_FALSE, start_idx * sizeof(BYTE), size,
                (void*)&context.data_input[start_idx], 0, NULL, NULL);
            if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);

            clFlush(context.cmd_queue[j]);

            /* Execute the kernel on the device. */
            errcode_ret = clEnqueueNDRangeKernel(context.cmd_queue[j], context.kernel[j], 2, NULL,
                context.global_work_size, context.local_work_size, 0, NULL, NULL);
            if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);
            clFlush(context.cmd_queue[j]);

            int read_idx = j * context.image_data_bytes / n_cmd_queues;

            /* Read back the device buffer to the host array. */

            errcode_ret = clEnqueueReadBuffer(context.cmd_queue[j], context.buffer_output_dev, CL_FALSE, read_idx * sizeof(BYTE),
                sizeof(BYTE) * context.image_data_bytes / n_cmd_queues, (void*)&context.data_output[read_idx], 0, NULL, NULL);
            if (CHECK_ERROR_CODE(errcode_ret)) exit(EXIT_FAILURE);

            clFlush(context.cmd_queue[j]);
        }
        for (int j = 0; j < n_cmd_queues; j++) {
            clFinish(context.cmd_queue[j]);
        }
#endif
        CHECK_TIME_END(_start, _end, _freq, compute_time);
        total_time += compute_time;
    }
    fprintf(stdout, "      * Time by host clock = %.3fms\n", total_time / context.n_kernel_call_iterations);

#if COPY_COMPUTE_COMMANDS_PROFILING
    // For the last execution
    print_time_statistics_for_copy_and_compute(n_cmd_queues);
#endif

}
