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

__kernel void query_row(__global const algorithmFPType *data,
                             __global const algorithmFPType *dist,
                             int Q,
                             int M,
                             int D,
                             int N) 
{

    const int index = get_global_id(0) * get_max_sub_group_num() + get_sub_group_id();
    if(index >= N)
        return;
    const int size = get_sub_group_size();
    const int id = get_sub_group_local_id();
    algorithmFPType sum = 0.0;
    for(int i = id; i < D; i += size)
    {
        algorithmFPType val = fabs(data[Q * D + i] - data[index * D + i]);
        algorithmFPType dm = 1.0;
        for(int j = 0; j < M; j++)
        {
            dm *= val;
        }
        sum += dm;
    }
    algorithmFPType ret = sub_group_reduce_add(sum);
    if(id == 0)
    {
        dist[index] = ret;
    }
}

__kernel void query_queue(__global const algorithmFPType *data,
                             __global const int *queue,
                             __global const algorithmFPType *dist,
                             int queue_begin,
                             int queue_size,
                             int M,
                             int D,
                             int N) 
{

    const int queue_pos = get_global_id(0);
    if(queue_pos >= queue_size)
        return;
    const int id = queue[queue_begin + queue_pos];
    const int group_num = get_sub_group_num()
    const int group_id = get_sub_group_id()
    const int group_size = get_sub_group_size();
    const int local_id = get_sub_group_local_id();
    __global const algorithmFPType * cur_dist = &dist[queue_pos * N];
    for(int j = group_id; j < N; j += group_num)
    {
        algorithmFPType sum = 0.0;
        for(int i = local_id; i < D; i += group_size)
        {
            algorithmFPType val = fabs(data[id * D + i] - data[j * D + i]);
            algorithmFPType dm = 1.0;
            for(int j = 0; j < M; j++)
            {
                dm *= val;
            }
            sum += dm;
        }
        algorithmFPType ret = sub_group_reduce_add(sum);
        if(local_id == 0)
        {
            cur_dist[j] = ret;
        }
    }
}

__kernel void query_nbrs(__global const algorithmFPType *data,
                         __global  algorithmFPType *dist,
                         int M,
                         int D,
                         int N) 
{

    const int row_id = get_global_id(0) * get_max_sub_group_num() + get_sub_group_id();
    const int size = get_sub_group_size();
    const int id = get_sub_group_local_id();
    const int index = indices[row_id];
    for(int j = 0; j < N; j++)
    {
            algorithmFPType sum = 0.0;
            for(int i = id; i < D; i += size)
            {
                algorithmFPType val = fabs(data[index * D + i] - data[j *D + i]);
                algorithmFPType dm = 1.0;
                for(int j = 0; j < M; j++)
                {
                    dm *= val;
                }
                sum += dm;
            }
            algorithmFPType ret = sub_group_reduce_add(sum);
            if(id == 0)
            {
                dist[row_id * N * D + j] = ret;
            }
        }
    }
}


__kernel void count_nbrs(__global const algorithmFPType *data,
                             __global int *counters,
                             algorithmFPType eps,
                             int M,
                             int N) 
{

    const int index = get_global_id(0) * get_max_sub_group_num() + get_sub_group_id();
    const int offset = index * M;
    const int number = N - offset < M ? N - offset : M;
    if(index >= N)
        return;
    const int size = get_sub_group_size();
    const int id = get_sub_group_local_id();
    __global const algorithmFPType * input = &data[offset];
    int count = 0;
    for(int i = id; i < number; i += size)
    {
        count += input[i] <= eps ? 1 : 0;
    }
    int ret = sub_group_reduce_add(count);
    if(local_id == 0)
    {
        counters[index] = ret;
    }
}

__kernel void copy_nbrs(__global const algorithmFPType *data,
                            __global int *counters,
                             __global int *nbr_indices,
                             algorithmFPType eps,
                             int M,
                             int N) 
{

    const int index = get_global_id(0) * get_max_sub_group_num() + get_sub_group_id();
    const int offset = index * M;
    const int number = N - offset < M ? N - offset : M;
    if(index >= N)
        return;
    const int size = get_sub_group_size();
    const int id = get_sub_group_local_id();
    __global const algorithmFPType * input = &data[offset];
    int count = 0;
    int out_offset = counters[index];
    for(int i = id; i < number; i += size)
    {
        int val = input[i] <= eps ? 1 : 0;
        int shift = sub_group_scan_exclusive(val);
        if(val)
        {
            nbr_indices[out_offset + shift] = i + offset;
        }
        out_offset += sub_group_reduce_add(val);
    }
}





);

#endif
