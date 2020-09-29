/* file: dbscan_cl_kernels.cl */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef __DBSCAN_CL_KERNELS_CL__
#define __DBSCAN_CL_KERNELS_CL__

#define DECLARE_SOURCE(name, src) static const char * name = #src;

DECLARE_SOURCE(
    dbscan_cl_kernels,

    algorithmFPType pow_by_mult(algorithmFPType value, int power) {
        algorithmFPType result = 1.0;
        for (int i = 0; i < power; i++)
        {
            result *= value;
        }
        return result;
    }

    void point_to_nbr_distance(__global const algorithmFPType * points, int point_id, int nbr_id, int power, int local_id, int subgroup_size,
                               int num_features, __global algorithmFPType * dist) {
        algorithmFPType sum = 0.0;
        for (int i = local_id; i < num_features; i += subgroup_size)
        {
            algorithmFPType val = fabs(points[point_id * num_features + i] - points[nbr_id * num_features + i]);
            sum += pow_by_mult(val, power);
        }
        algorithmFPType cur_nbr_distance = sub_group_reduce_add(sum);
        if (local_id == 0)
        {
            dist[nbr_id] = cur_nbr_distance;
        }
    }

    __kernel void compute_point_distances(__global const algorithmFPType * points, int point_id, int power, int num_features, int num_points,
                                          __global algorithmFPType * dist) {
        const int subgroup_index = get_global_id(0) * get_max_sub_group_size() + get_sub_group_id();
        if (subgroup_index >= num_points) return;

        const int subgroup_size = get_sub_group_size();
        const int local_id      = get_sub_group_local_id();
        point_to_nbr_distance(points, point_id, subgroup_index, power, local_id, subgroup_size, num_features, dist);
    }

    __kernel void compute_queue_block_distances(__global const algorithmFPType * points, __global const int * queue, int queue_begin, int queue_size,
                                                int power, int num_features, int num_points, __global algorithmFPType * queue_dist) {
        const int queue_pos = get_global_id(0);
        if (queue_pos >= queue_size) return;

        const int point_id                        = queue[queue_begin + queue_pos];
        const int group_num                       = get_num_sub_groups();
        const int group_id                        = get_sub_group_id();
        const int subgroup_size                   = get_sub_group_size();
        const int local_id                        = get_sub_group_local_id();
        __global algorithmFPType * cur_block_dist = &queue_dist[queue_pos * num_points];
        for (int j = group_id; j < num_points; j += group_num)
        {
            point_to_nbr_distance(points, point_id, j, power, local_id, subgroup_size, num_features, cur_block_dist);
        }
    }

    int get_nbr_status(__global const algorithmFPType * distances, int nbr_id, algorithmFPType eps) { return (distances[nbr_id] <= eps ? 1 : 0); }

    int get_undefined_nbr_status(__global const int * assignments, int point_id, int nbr_id, int is_nbr, int subgroup_offset) {
        int is_current_point = nbr_id + subgroup_offset == point_id;
        return (is_nbr > 0 && assignments[nbr_id] == _UNDEFINED_ && !is_current_point ? 1 : 0);
    }

    __kernel void count_neighbors_by_type(__global const int * assignments, __global const algorithmFPType * distances, int point_id,
                                          int first_chunk_offset, int chunk_size, int num_points, algorithmFPType eps, __global const int * queue,
                                          __global int * counters_all_nbrs, __global int * counters_undef_nbrs) {
        const int subgroup_index        = get_global_id(0) * get_num_sub_groups() + get_sub_group_id();
        const int subgroup_offset       = subgroup_index * chunk_size;
        const int subgroup_point_number = num_points - subgroup_offset < chunk_size ? num_points - subgroup_offset : chunk_size;
        point_id                        = first_chunk_offset < 0 ? point_id : queue[point_id];
        const int distance_offset       = subgroup_offset + (first_chunk_offset < 0 ? 0 : first_chunk_offset);
        if (subgroup_index >= num_points) return;

        const int subgroup_size                             = get_sub_group_size();
        const int local_id                                  = get_sub_group_local_id();
        __global const algorithmFPType * subgroup_distances = &distances[distance_offset];
        __global const int * subgroup_assignments           = &assignments[subgroup_offset];
        int counter_all                                     = 0;
        int counter_undefined                               = 0;
        for (int i = local_id; i < subgroup_point_number; i += subgroup_size)
        {
            int is_nbr           = get_nbr_status(subgroup_distances, i, eps);
            int is_undefined_nbr = get_undefined_nbr_status(subgroup_assignments, point_id, i, is_nbr, subgroup_offset);
            counter_all += is_nbr;
            counter_undefined += is_undefined_nbr;
        }
        int subgroup_counter_all       = sub_group_reduce_add(counter_all);
        int subgroup_counter_undefined = sub_group_reduce_add(counter_undefined);
        if (local_id == 0)
        {
            counters_all_nbrs[subgroup_index]   = subgroup_counter_all;
            counters_undef_nbrs[subgroup_index] = subgroup_counter_undefined;
        }
    }

    __kernel void set_buffer_value(int value_index, int value, __global int * buffer) {
        const int global_id = get_global_id(0);
        const int local_id  = get_local_id(1);
        if (local_id == 0 & global_id == 0) buffer[value_index] = value;
    }

    __kernel void set_buffer_value_by_queue_index(__global const int * queue, int queue_index, int value, __global int * buffer) {
        const int global_id = get_global_id(0);
        const int local_id  = get_local_id(1);
        if (local_id == 0 & global_id == 0) buffer[queue[queue_index]] = value;
    }

    __kernel void compute_chunk_offsets(__global const int * chunk_counters, int num_offsets, __global int * chunk_offsets) {
        if (get_sub_group_id() > 0) return;

        const int subgroup_size = get_sub_group_size();
        const int local_id      = get_sub_group_local_id();
        int subgroup_offset     = 0;
        for (int i = local_id; i < num_offsets; i += subgroup_size)
        {
            int cur_counter  = chunk_counters[i];
            int local_offset = subgroup_offset + sub_group_scan_exclusive_add(cur_counter);
            int total_offset = sub_group_reduce_add(cur_counter);
            chunk_offsets[i] = local_offset;
            subgroup_offset += total_offset;
        }
    }

    __kernel void push_points_to_queue(__global const algorithmFPType * distances, __global const int * chunk_offsets, int queue_last_element,
                                       int point_id, int cluster_id, int first_chunk_offset, int chunk_size, algorithmFPType eps, int num_points,
                                       __global int * assignments, __global int * queue) {
        const int subgroup_index  = get_global_id(0) * get_num_sub_groups() + get_sub_group_id();
        const int subgroup_offset = subgroup_index * chunk_size;
        point_id                  = first_chunk_offset < 0 ? point_id : queue[point_id];
        const int dist_offset     = subgroup_offset + (first_chunk_offset < 0 ? 0 : first_chunk_offset);
        if (subgroup_offset >= num_points) return;

        const int subgroup_point_number                     = num_points - subgroup_offset < chunk_size ? num_points - subgroup_offset : chunk_size;
        const int subgroup_size                             = get_sub_group_size();
        const int local_id                                  = get_sub_group_local_id();
        __global const algorithmFPType * subgroup_distances = &distances[dist_offset];
        __global int * subgroup_assignments                 = &assignments[subgroup_offset];
        const int subgroup_chunk_offset                     = chunk_offsets[subgroup_index];
        int local_offset                                    = 0;
        for (int i = local_id; i < subgroup_point_number; i += subgroup_size)
        {
            int is_nbr           = get_nbr_status(subgroup_distances, i, eps);
            int is_undefined_nbr = get_undefined_nbr_status(subgroup_assignments, point_id, i, is_nbr, subgroup_offset);
            int local_pos        = sub_group_scan_exclusive_add(is_undefined_nbr);
            if (is_undefined_nbr)
            {
                subgroup_assignments[i]                                                      = cluster_id;
                queue[queue_last_element + subgroup_chunk_offset + local_offset + local_pos] = subgroup_offset + i;
            }
            if (is_nbr && (subgroup_assignments[i] == _NOISE_))
            {
                subgroup_assignments[i] = cluster_id;
            }
            local_offset += sub_group_reduce_add(is_undefined_nbr);
        }
        if (subgroup_index == 0 && local_id == 0)
        {
            assignments[point_id] = cluster_id;
        }
    }

    __kernel void compute_cores_ex(int num_points, int num_features, int num_nbrs, algorithmFPType eps, 
                                   __global const algorithmFPType * points, __global int * cores) 
    {
        __local algorithmFPType buffer[8000];
        const int num = 8000 / num_features;
        const int global_id = get_global_id(0);
        const int local_size = get_local_size(1);
        const int index = global_id * local_size + get_local_id(1);
/*        const int subgroup_id = get_sub_group_id();
        const int subgroup_size = get_sub_group_size();
        const int local_id      = get_sub_group_local_id();
        const int num_sub_groups = get_num_sub_groups();
        if(local_id == 0)
            printf("item: %d %d %d %d\n", global_id, subgroup_id, num_sub_groups, subgroup_size);
        */
        int total = 0;
        int count = 0;
        while(total < num_points) {
            int cur_size = num_points - total;
            cur_size = cur_size > num ? num : cur_size;
            event_t e = async_work_group_copy(buffer, points + total * num_features, cur_size * num_features, 0);
            wait_group_events(1, &e);
            total += cur_size;
            if(index < num_points) 
            {
                for(int j = 0; j < cur_size; j++) 
                {
                    algorithmFPType sum = 0.0;
                    for (int i = 0; i < num_features; i++)
                    {
                        algorithmFPType val = points[index * num_features + i] - buffer[j * num_features + i];
                        val *= val;
                        sum += val;
                    }
                    int incr = (int)(sum <= eps);
                    count += incr;
                }
            }
        }
        cores[index] = (int)(count >= num_nbrs);
    }

    __kernel void compute_cores(int num_points, int num_features, int num_nbrs, algorithmFPType eps, 
                                __global const algorithmFPType * points, __global int * cores) 
    {
        const int global_id = get_global_id(0);
        if (get_sub_group_id() > 0) return;

        const int subgroup_size = get_sub_group_size();
        const int local_id      = get_sub_group_local_id();
//        if(local_id == 0 && global_id == 0)
//            printf("Input: %d %d %d %d\n", num_points, num_features, num_nbrs, subgroup_size);
        int count = 0;
        for(int j = 0; j < num_points; j++) 
        {
            algorithmFPType sum = 0.0;
            for (int i = local_id; i < num_features; i += subgroup_size)
            {
                algorithmFPType val = points[global_id * num_features + i] - points[j * num_features + i];
//                if(global_id == 0 && local_id == 0)
//                    printf("OrgVal: %d %15.8f\n", j, val);
                val *= val;
                sum += val;
//                if(global_id == 0 && local_id == 0)
//                    printf("Val: %d %15.13f %15.13f\n", j, val, sum);
            }
//            if(global_id == 0 && local_id == 0)
//                    printf("Final sum: %d %15.13f\n", j, sum);
            algorithmFPType cur_nbr_distance = sub_group_reduce_add(sum);
            int incr = (int)(cur_nbr_distance <= eps);
            count += incr;
//            if(global_id == 0 && local_id == 0) 
//                printf("Increase: %d %d %d %15.13f %15.13f %d\n", global_id, local_id, j, cur_nbr_distance, eps, incr);
        }
        if (local_id == 0)
        {
//            printf("Count: %d %d\n", global_id, count);
            cores[global_id] = (int)(count >= num_nbrs);
        }
    }

    __kernel void startNextCluster(int cluster_id, int num_points, int queue_end, const __global int * cores, 
                                   __global int * clusters, __global int * last_point, __global int * queue) 
    {
        if(get_sub_group_id() > 0)
            return;

        const int subgroup_size = get_sub_group_size();
        const int local_id      = get_sub_group_local_id();
        int start = last_point[0];
        for(int i = start + local_id; i < num_points; i++)
        {
            bool found = cores[i] == 1 && clusters[i] == -2;
            int index = sub_group_reduce_min(found ? i : num_points);
            if(index < num_points) 
            {
                if (local_id == 0) 
                {
                    clusters[index] = cluster_id;
                    last_point[0] = index + 1;
                    queue[queue_end] = index;
                }
                break;
            }
        }
    }

    __kernel void update_queue(int cluster_id, int num_points, int num_features, algorithmFPType eps, int queue_start, int queue_end, 
                                const __global algorithmFPType * points, __global int * cores, 
                                __global int * clusters, __global int * queue, 
                                __global int * queue_front) 
    {
        const int subgroup_index = get_global_id(0);
        if (get_sub_group_id() > 0) return;
        const int local_id      = get_sub_group_local_id();
//        if(local_id == 0)
//            printf("Cluster: %d %d \n", subgroup_index, clusters[subgroup_index]);
        if(clusters[subgroup_index] > -1) return;
        const int subgroup_size = get_sub_group_size();
        volatile __global int* counterPtr = queue_front;
        for(int j = queue_start; j < queue_end; j++) 
        {
            int index = queue[j];
//            if(local_id == 0)
//                printf("queue index: %d\n", index);
            algorithmFPType sum = 0.0;
            for (int i = local_id; i < num_features; i += subgroup_size)
            {
                algorithmFPType val = points[subgroup_index * num_features + i] - points[index * num_features + i];
                sum += val * val;
            }
            algorithmFPType distance = sub_group_reduce_add(sum);
            if(distance > eps)
                continue;
//            if(local_id == 0)
//                printf("Neghbor found\n");
            if(local_id == 0) 
            {
                clusters[subgroup_index] = cluster_id;
            }
            if(cores[subgroup_index] == 0)
                continue;
//            if(local_id == 0)
//                printf("Neghbor is core\n");
            if(local_id == 0) 
            {
                int queue_index = atomic_inc(counterPtr);
//                printf("queue front: %d %d %d %d\n", queue_index, subgroup_index, clusters[subgroup_index], cores[subgroup_index]);
                queue[queue_index] = subgroup_index;
            }
            break;
        }
    }


);

#endif
