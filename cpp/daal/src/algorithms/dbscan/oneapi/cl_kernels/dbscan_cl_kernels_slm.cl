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

    __kernel void computeCoresSlm2(int numPoints, int numFeatures, int numNbrs, algorithmFPType eps, const __global algorithmFPType * points,
                               __global int * cores) 
    {
        __local algorithmFPType xBuffer[__X_SIZE__];
        const int subgroupSize = get_sub_group_size();
        const int localId      = get_sub_group_local_id();

        const int batchMaxRows = __X_SIZE__ / numFeatures;
        const int globalId = get_global_id(0);
        const int firstRow = globalId * batchMaxRows;
        const int lastRow = firstRow + batchMaxRows > numPoints ? numPoints : firstRow + batchMaxRows;
        const int nRows = (lastRow - firstRow);
        event_t ev1 = async_work_group_copy(xBuffer, &points[firstRow * numFeatures], nRows * numFeatures, 0);
        wait_group_events(1,&ev1);
        int count = 0;

        for(int i = 0; i < numPoints; i++)
        {
            algorithmFPType distance = 0;
            for(int j = localId; j < numFeatures; j += subgroupSize)
            {
                algorithmFPType y = points[i * numFeatures + j];
                for(int k = 0; k < nRows; k++)
                { 
                    algorithmFPType val = xBuffer[k * numFeatures + j];
                    val -= y;
                    algorithmFPType sum = sub_group_reduce_add(val * val);
                    distance += k == j ? sum : 0.0;
                }
            }
            count += (int)(distance <= eps);
        }
        if(localId < nRows)
        {
            cores[firstRow + localId] = (int)(count >= numNbrs);
        }
    }

    void load16(int row, int col, int firstRow, int tileOffset, int width, int rowOffset, int xRows,
                         const __global algorithmFPType * points, __local algorithmFPType * buffer)
    {
        const int offset = firstRow * rowOffset + tileOffset + rowOffset * row + col;
        buffer[row * 16 + col] = row < xRows && col < width ? points[offset] : 0.0;

    }

    algorithmFPType partialDistance16(int row, int col, int width, const __local algorithmFPType * xBuffer, 
                                    const __local algorithmFPType * yBuffer)
    {
        algorithmFPType distance = 0.0;
//        printf("Distance: row %d col %d xbuf %f ", row, col, xBuffer[row*16]);
        for(int i = 0; i < width; i ++)
        {
            algorithmFPType a =  xBuffer[i + row * 16];
            algorithmFPType b =  yBuffer[i + col * 16];
            distance += a * a - 2 * a * b + b * b;
        }
//        printf("Distance: row %d col %d xbuf[row * 16] %f ybuf[col * 16] %f dist %E\n", row, col, xBuffer[row * 16], yBuffer[col * 16], distance);
        return distance;
    }

    __kernel void computeCoresSlm16x16(int numPoints, int numFeatures, int numNbrs, algorithmFPType eps, const __global algorithmFPType * points,
                               __global int * cores) {
        // WG size 256
        __local algorithmFPType xBuffer[256];
        __local algorithmFPType yBuffer[256];
        const int localId = get_local_id(1);
        const int row = localId / 16;
        const int col = localId % 16;
        const int firstRow = get_global_id(0) * 16;
        const int xRows = 16 > numPoints - firstRow ? numPoints - firstRow : 16;
        int count = 0;
        for(int i = 0; i < numPoints; i += 16)
        {
            algorithmFPType distance = 0;
            const int yRows = 16 > numPoints - i ? numPoints - i : 16;
            if(firstRow == 0 && localId == 0)
            {
//                printf("xRows %d yRows %d\n", xRows, yRows);
            }
            for(int j = 0; j < numFeatures; j += 16)
            {
                const int width = 16 > numFeatures - j ? numFeatures -j : 16;
                load16(row, col, firstRow, j, width, numFeatures, xRows, points, xBuffer);
                load16(row, col, i, j, width, numFeatures, yRows, points, yBuffer);
                barrier(CLK_LOCAL_MEM_FENCE);
//                if(firstRow == 0)
//                {
//                    printf(" row %d col %d xBuf %f yBuf %f\n", row, col, xBuffer[localId], yBuffer[localId]);
//                }
                distance += partialDistance16(row, col, width, xBuffer, yBuffer);
            }
            count += row < xRows && col < yRows ? (int)(distance <= eps) : 0;
//            printf(" row %d col %d dist %f count %d \n", row, col, distance, count);
        }
        // SG size 16 is assumed
        int final = 0;
        for(int i = 0; i < 16; i++) 
        {
            const int sum = sub_group_reduce_add(row == i ? count : 0);
            final += row == i ? sum : 0;
        }
//        printf("localId %d row %d count %d final %d\n", localId, row, count, final);
        if(col == 0)
        {
            cores[firstRow + row] = (int)(final >= numNbrs);
//            printf("End: id %d core %d final %d final %d\n", firstRow + row, cores[firstRow + row], final, numNbrs);
        }
    }
/*
    void loadPacked16(int localId, int firstRow, int rowOffset, int rowWidth, int xRows,
                      const __global algorithmFPType * points, __local algorithmFPType * xbuffer)
    {
        const int maxId = 16 * xRows;
        const int row = localId / 16;
        const int col = localId % 16;
        const int offset = firstRow * rowOffset + row * rowWidth + col;
        xBuffer[localId] = localId < maxId && col < rowWidth ? points[offset] : 0.0;
    }

    algorithmFPType partialDistancePacked16(int localId, int pointInRow, int pointInCol, int pointWidth, 
                                            const __local algorithmFPType * xbuffer, 
                                            const __local algorithmFPType * ybuffer)
    {
        algorithmFPType distance = 0;
        const int row = localId / 16;
        const int col = localId % 16;
        
        for(int i = 0; i < width; i ++)
        {
            algorithmFPType a =  xBuffer[i + row * 16 + pointInRow * pointWidth];
            algorithmFPType b =  yBuffer[i + col * 16 + pointInCol * pointWidth];
            distance += a * a - 2 * a * b + b * b;
        }
        return distance;
    }


    __kernel void computeCoresSlmPacked16x16(int numPoints, int numFeatures, int numNbrs, algorithmFPType eps, const __global algorithmFPType * points,
                               __global int * cores) {
        // WG size 256
        __local algorithmFPType xBuffer[256];
        __local algorithmFPType yBuffer[256];
        const int localId = get_local_id(1);
        const int row = localId / 16;
        const int col = localId % 16;
        const int firstRow = get_global_id(0) * 16;
        const int xRows = packSize > numPoints - firstRow ? numPoints - firstRow : 16;
        loadPacked16(localId, firstRow, nFeatures, packSize, xRows, points, xBuffer);
        int count[16];
        for(int i = 0; i < 16; i++)
        {
            count[j] = 0;
        }
        for(int i = 0; i < numPoints; i += packSize)
        {
            const yRows = packSize > numPoints - i ? numPoints - i : packSize;
            algorithmFPType distance = 0;
            loadPacked16(localId, firstRow, nFeatures, packSize, yRows, points, yBuffer);
            for(int j = 0; j < numPacks; j++)
            {
                for(int k = 0; k < numPacks; k++)
                {
                    algorithmFPType distance = partialDistancePacked16(localId, j, k, nFeatures, xbuffer, ybuffer);
                    count[j] += col < yRows && row < xRows ? (int)(distance <= eps && ) : 0;
                }
            }
        }
        // SG size 16 is assumed
        int final = 0;
        for(int i = 0; i < 16; i++) {
            const int sum = sub_group_reduce_add(row == i ? count : 0);
            final += row == i ? sum : 0;
        }
        if(col < numPacks)
        {
            cores[firstRow + row] = (int)(count >= numNbrs);
        }
    }    
*/
);

#endif
