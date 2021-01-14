/* file: dbscan_cl_kernels_slm.cl */
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

#ifndef __DBSCAN_CL_KERNELS_SLM_CL__
#define __DBSCAN_CL_KERNELS_SLM_CL__

#define DECLARE_SOURCE(name, src) static const char * name = #src;

DECLARE_SOURCE(
    dbscanClKernelsSlm,

    __kernel void computeCoresSlm(int numPoints, int numFeatures, int numNbrs, algorithmFPType eps, const __global algorithmFPType * points,
                               __global int * cores) {
        __local algorithmFPType xBuffer[__X_SIZE__];
        __local algorithmFPType yBuffer[__Y_SIZE__];
//        __local int counts[32];
/*        
        if(get_global_id(0) == 0 && get_local_id(1) == 0) 
        {
            printf("global size %d x %d; local size %d x %d\n", get_global_size(0), get_global_size(1),
                   get_local_size(0), get_local_size(1));
            printf(" %d; %d\n", get_global_size(0), get_global_size(1),
                   get_local_size(0), get_local_size(1));
        }
*/
        const int subgroupSize = get_sub_group_size();
        const int subgroupId   = get_sub_group_id();
        const int nSubgroups   = get_num_sub_groups();
        const int localId      = get_sub_group_local_id();

        const int batchMaxRows = __X_SIZE__ / numFeatures;
        const int yBlockMaxSize = __Y_SIZE__ / numFeatures;
        const int globalId = get_global_id(0);
        const int firstRow = globalId * batchMaxRows;
        const int lastRow = firstRow + batchMaxRows > numPoints ? numPoints : firstRow + batchMaxRows;
        const int nRows = (lastRow - firstRow) > nSubgroups ? nSubgroups : (lastRow - firstRow);
/*        
        if(get_global_id(0) == 0 && get_local_id(1) == 0) 
        {
            printf("batch %d block %d\n", batchMaxRows, yBlockMaxSize);
        }
*/
        event_t ev1 = async_work_group_copy(xBuffer, &points[firstRow * numFeatures], nRows*numFeatures, 0);
        wait_group_events(1,&ev1);

        const int numBlocks = numPoints / yBlockMaxSize + (int)((bool)(numPoints % yBlockMaxSize));
/*
        if(get_global_id(0) == 0 && get_local_id(1) == 0) 
        {
            printf("subgroups %d subgroup size %d numBlocks %d nRows %d\n", nSubgroups, subgroupSize, numBlocks, nRows);
            for(int i = 0; i < nRows; i++)
            {
                printf("%d ", i);
                for(int j = 0; j < numFeatures; j++)
                {
                    printf("%f ", xBuffer[i * numFeatures + j]);
                }
                printf("\n");
            }
        }*/
        int count = 0;
        for(int i = 0; i < numBlocks; i++)
        {
            const int blockBegin = i * yBlockMaxSize;
            const int blockEnd   = blockBegin + yBlockMaxSize > numPoints ? numPoints : blockBegin + yBlockMaxSize;
            const int blockSize  = blockEnd - blockBegin;
            event_t ev2 = async_work_group_copy(yBuffer, &points[blockBegin * numFeatures], blockSize * numFeatures, 0);
            wait_group_events(1,&ev2);
            const int offset = subgroupId * numFeatures;
            for(int k =0; k < blockSize; k++)
            {
                algorithmFPType sum = 0.0;
                for(int l = localId; l < numFeatures; l += subgroupSize)
                {
                    algorithmFPType val = xBuffer[subgroupId * numFeatures + l] - yBuffer[k * numFeatures + l];
                    sum += val * val;
                }
                algorithmFPType distance = sub_group_reduce_add(sum);
                count += localId == 0 ? (int)(distance <= eps) : 0;
            }
        }

        for(int j = subgroupId; j < nRows; j += nSubgroups)
        {
            if (localId == 0)
            {
                cores[firstRow + j] = (int)(count >= numNbrs);
/*                if(firstRow == 0)
                {
                    printf("Cores: %d %d %d \n", firstRow + j, cores[firstRow + j], count);
                }
*/                    
            }
        }
    }


);

#endif
