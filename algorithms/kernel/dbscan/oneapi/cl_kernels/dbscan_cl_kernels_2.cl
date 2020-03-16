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
                             __global algorithmFPType *dist,
                             int row_id,
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
        algorithmFPType val = fabs(data[row_id * D + i] - data[index * D + i]);
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
                             __global  algorithmFPType *dist,
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

__kernel void count_neighbors(__global const int *assignments,
                            __global const algorithmFPType *data,
                            int chunk_size, 
                            int N, 
                            algorithmFPType eps,
                             __global int *counters,
                             __global int *undefCounters,
                            ) 
{

    const int index = get_global_id(0) * get_max_sub_group_num() + get_sub_group_id();
    const int offset = index * chunk_size;
    const int number = N - offset < M ? N - offset : M;
    if(index >= N)
        return;
    const int size = get_sub_group_size();
    const int id = get_sub_group_local_id();
    __global const algorithmFPType * input = &data[offset];
    __global const int * assigned = &assignments[offset];
    int count = 0;
    int newCount = 0;
    for(int i = id; i < number; i += chunk_size)
    {
        int nbrFlag = input[i] <= eps ? 1 : 0;
        int newNbrFlag = nbrFlag > 0 && assigned[i] == _UNDEFINED_ ? 1 : 0;
        count += nbrFlag;
        newCount += newNbrFlag;
    }
    int retCount = sub_group_reduce_add(count);
    int retNewCount = sub_group_reduce_add(newCount);
    if(local_id == 0)
    {
        counters[index] = retCount;
        undefCounters[index] = retNewCount;
    }
}


__kernel void setBufferValue(__global int *buffer,
                             int index,
                             int value)
{
    const int global_id = get_global_id(0);
    const int global_id = get_local_id(1);
    if(local_id == 0 & global_id == 0)
        buffer[index] = value;
}
__kernel void set_buffer_value_by_queue_index(__global int *queue,
                            __global int *buffer,
                             int pos,
                             int value)
{
    const int global_id = get_global_id(0);
    const int global_id = get_local_id(1);
    if(local_id == 0 & global_id == 0)
        buffer[queue[pos]] = value;
}

__kernel void process_row_neighbors(__global const algorithmFPType * distances,
                                    __global const int * offsets,
                                    __global const int * assignments,
                                    __global const int * queue,
                                    __global const int * queue_end,
                                    int row_id,
                                    int cluster_id,
                                    int chunk_size, 
                                    algorithmFPType eps,
                                    int N)
{
    const int index = get_global_id(0) * get_max_sub_group_num() + get_sub_group_id();
    const int offset = index * chunk_size;
    if(offset >= N)
        return;
    const int number = N - offset < chunk_size ? N - offset : chunk_size;
    const int size = get_sub_group_size();
    const int id = get_sub_group_local_id();
    __global const algorithmFPType * input = &distances[offset];
    __global const int * assigned = &assignments[offset];
    const int out_offset = counters[index];
    int local_offset = 0;
    for(int i = id; i < number; i++) {
        int nbrFlag = input[i] <= eps ? 1 : 0;
        int newNbrFlag = nbrFlag > 0 && assigned[i] == _UNDEFINED_ ? 1 : 0;
        int pos = sub_group_scan_exclusive(newNbrFlag);
        if(newNbrFlag)
        {
            queue[*queue_end + offset + local_offset + pos] = offset + i;
        }
        if(nbrFlag && assigned[i] == _NOISE_)
        {
            assigned[i] = cluster_id;
        }
        local_offset += sub_group_reduce_add(newNbrFlag);
    }
    if(index == 0 && id == 0)
    {
        assignments[row_id] = cluster_id;
    }
}

__kernel void process_queue_neighbors(__global const algorithmFPType * distances,
                                    __global const int * offsets,
                                    __global const int * assignments,
                                    __global const int * queue,
                                    __global const int * queue_end,
                                    int queue_pos,
                                    int cluster_id,
                                    int chunk_size, 
                                    algorithmFPType eps,
                                    int N)
{
    const int index = get_global_id(0) * get_max_sub_group_num() + get_sub_group_id();
    const int offset = index * chunk_size;
    if(offset >= N)
        return;
    const int number = N - offset < chunk_size ? N - offset : chunk_size;
    const int size = get_sub_group_size();
    const int id = get_sub_group_local_id();
    __global const algorithmFPType * input = &distances[offset];
    __global const int * assigned = &assignments[offset];
    const int out_offset = counters[index];
    int local_offset = 0;
    for(int i = id; i < number; i++) {
        int nbrFlag = input[i] <= eps ? 1 : 0;
        int newNbrFlag = nbrFlag > 0 && assigned[i] == _UNDEFINED_ ? 1 : 0;
        int pos = sub_group_scan_exclusive(newNbrFlag);
        if(newNbrFlag)
        {
            queue[*queue_end + offset + local_offset + pos] = offset + i;
        }
        if(assigned[i] == _NOISE_)
        {
            assigned[i] = cluster_id;
        }
        local_offset += sub_group_reduce_add(newNbrFlag);
    }
    if(index == 0 && id == 0)
    {
        assignments[queue[queue_pos]] = cluster_id;
    }
}

);

#endif
