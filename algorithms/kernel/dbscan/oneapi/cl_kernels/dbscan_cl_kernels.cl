/* file: dbscan_cl_kernels.cl */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/*
//++
//  Implementation of DBSCAN OpenCL kernels.
//--
*/

#ifndef __KNN_CL_KERNELS_CL__
#define __KNN_CL_KERNELS_CL__

#include <string.h>

#define DECLARE_SOURCE(name, src) static const char* name = #src;

DECLARE_SOURCE(dbscan_cl_kernels,

__kernel void init_distances(__global const algorithmFPType *querySq,
                             __global const algorithmFPType *dataSq,
                             __global       algorithmFPType *distances,
                             int N) {

    const int global_id_0  = get_global_id(0);
    const int global_id_1  = get_global_id(1);

    distances[global_id_0  + global_id_1 * N] = dataSq[global_id_0] + querySq[global_id_1];
}

__kernel void partition( __global  algorithmFPType *distances_in,
                __global int *indices_out,
                __global int *counters_out,
                int N,
                int M,
                algorithmFPType pivot)
 {
    const int global_id  = get_global_id(0);
    const int local_id = get_local_id(1);
    const int local_size = get_local_size(1);
    const int local_size_0 = get_local_size(0);

    __global  algorithmFPType *distances = &distances_in[global_id * N];
    __global int *indices = &indices_out[global_id * N];
    __global int *counters = &counters_out[0];

    for(int i = local_id; i < N; i += local_size)
        indices[i] = i;
    /*if(local_id == 0 && global_id == 1) {
        int small = 0;
        for(int i = 0; i < N; i++) 
        {
            if(distances[i] < pivot) 
            {
                small++;
                printf(" %d ", i);
            }
        }
        printf("row %d small %d \n", global_id, small);
    }*/
/*
    if(global_id == 1 && local_id == 0)
        printf("N %d M %d pivot %f local_size %d local_size_0 %d\n", N, M, pivot, local_size, local_size_0);
*/
    int count = sub_group_reduce_add(1);
/*    if(global_id == 1 && local_id == 0)
        printf("Count=%d\n", count); 
*/
    int last_group_size = N % local_size;
    int full_group_size = N -last_group_size;
    int split = 0;
    for(int i = local_id; i < N; i += local_size) 
    {
        sub_group_barrier(CLK_GLOBAL_MEM_FENCE);
        algorithmFPType curVal = distances[i];
        int curLabel = indices[i];
        unsigned char flag = curVal < pivot ? 1 : 0;
        unsigned char incr = sub_group_reduce_add(flag);
        int min_ind = sub_group_reduce_min(i);
        if(incr > 0) 
        {
            unsigned char shift = sub_group_scan_exclusive_add(flag);
            unsigned char old_shift= sub_group_scan_exclusive_add(flag > 0 ? 0 : 1);
            algorithmFPType exVal = flag > 0 ? distances[split + shift] : -1.0;
            int exLabel = flag > 0 ? indices[split + shift] : -1;
            int cur_size = i > full_group_size - 1 ? last_group_size : local_size;
            if(flag) 
            {
                distances[split + shift] = curVal;
                indices[split + shift] = curLabel;
                if(split + shift < min_ind) {
                    distances[min_ind + cur_size - 1 - shift] = exVal;
                    indices[min_ind + cur_size - 1 - shift] = exLabel;
                }
            } 
            else 
            {
                distances[min_ind + cur_size - 1 - old_shift] = curVal;
                indices[min_ind + cur_size - 1 - old_shift] = curLabel;
            }
        }
        sub_group_barrier(CLK_GLOBAL_MEM_FENCE);
        split += incr;
/*        if(global_id == 1)
            printf("index %d local_id %d value %f flag %d incr %d split %d \n", curLabel, local_id, curVal, flag, incr, split);
*/
    }
    split = - sub_group_reduce_min(-(split));
    if(local_id == 0)
        counters[global_id] = split;
    /*
    if(local_id == 0 && global_id == 1) {
        for(int i = 0; i < split; i++) 
        {
            printf(" res %d ", indices[i]);
        }
    }*/
}

);

#endif
