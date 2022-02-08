//
//  dummy.cl
//
//  Written for CSEG437_CSE5437
//  Department of Computer Science and Engineering
//  Copyright © 2021 Sogang University. All rights reserved.
//

__kernel
void dummy(__global float* A, __global float* B, __global float* C) {
    int i = get_global_id(0);

    C[i] = A[i] + B[i];
}
