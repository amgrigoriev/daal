/* file: kmeans_dense_lloyd_kernel_base_ucapi_impl.i */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
//  Implementation of K-means Base Kernel for GPU.
//--
*/

#ifndef __KMEANS_DENSE_LLOYD_KERNEL_BASE_UCAPI_IMPL__
#define __KMEANS_DENSE_LLOYD_KERNEL_BASE_UCAPI_IMPL__

#include "kmeans_dense_lloyd_kernel_base_ucapi.h"
#include "env_detect.h"
#include "cl_kernels/kmeans_cl_kernels.cl"
#include "execution_context.h"
#include "oneapi/service_defines_oneapi.h"
#include "oneapi/internal/types.h"
#include "oneapi/blas_gpu.h"

#include "service_ittnotify.h"

using namespace daal::services;
using namespace daal::oneapi::internal;
using namespace daal::data_management;

inline char * utoa(uint32_t value, char * buffer, uint32_t buffer_size)
{
    uint32_t i = 0;
    while (value && i < buffer_size - 1)
    {
        size_t rem  = value % 10;
        buffer[i++] = 48 + rem;
        value /= 10;
    }

    for (uint32_t j = 0; j < i - j - 1; j++)
    {
        char tmp          = buffer[j];
        buffer[j]         = buffer[i - j - 1];
        buffer[i - j - 1] = tmp;
    }
    buffer[i] = 0;
    return buffer;
}

inline char * append(char * buffer, uint32_t & pos, uint32_t buffer_size, const char * append, uint32_t append_size)
{
    uint32_t i = 0;
    while (pos + i < buffer_size && i < append_size)
    {
        if (append[i] == 0) break;
        buffer[pos + i] = append[i];
        i++;
    }
    pos += i;
    return buffer;
}

inline uint32_t constStrLen(const char * s)
{
    uint32_t len = 0;
    while (s[len] != 0) len++;
    return len;
}

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace internal
{

template <typename algorithmFPType>
uint32_t KMeansDenseLloydKernelBaseUCAPI<algorithmFPType>::getCandidatePartNum(uint32_t nClusters)
{
    return _maxLocalBuffer / nClusters / sizeof(algorithmFPType);
}
template <typename algorithmFPType>
const char * KMeansDenseLloydKernelBaseUCAPI<algorithmFPType>::getBuildOptions(uint32_t nClusters)
{
    const uint32_t bufSize    = 1024;
    const uint32_t valBufSize = 16;
    static char buffer[bufSize];
    static char valBuffer[valBufSize];
    uint32_t numParts = getCandidatePartNum(nClusters);
    if (numParts > _preferableSubGroup) numParts = _preferableSubGroup;
    const char * s1 = "-cl-std=CL1.2 -D LOCAL_SUM_SIZE=";
    const char * s2 = " -D CND_PART_SIZE=";
    const char * s3 = " -D NUM_PARTS_CND=";
    uint32_t pos    = 0;
    append(buffer, pos, bufSize, s1, constStrLen(s1));
    append(buffer, pos, bufSize, utoa(_maxWorkItemsPerGroup, valBuffer, valBufSize), valBufSize);
    append(buffer, pos, bufSize, s2, constStrLen(s2));
    append(buffer, pos, bufSize, utoa(nClusters, valBuffer, valBufSize), valBufSize);
    append(buffer, pos, bufSize, s3, constStrLen(s3));
    append(buffer, pos, bufSize, utoa(numParts, valBuffer, valBufSize), valBufSize);
    buffer[pos] = 0;
    return buffer;
}

template <typename algorithmFPType>
uint32_t KMeansDenseLloydKernelBaseUCAPI<algorithmFPType>::getWorkgroupsCount(uint32_t rows)
{
    const uint32_t elementsPerGroup = _maxWorkItemsPerGroup;
    uint32_t workgroupsCount        = rows / elementsPerGroup;

    if (workgroupsCount * elementsPerGroup < rows) workgroupsCount++;

    return workgroupsCount;
}

template <typename algorithmFPType>
uint32_t KMeansDenseLloydKernelBaseUCAPI<algorithmFPType>::getComputeSquaresWorkgroupsCount(uint32_t nFeatures)
{
    size_t workItemsPerGroup = nFeatures < _maxWorkItemsPerGroup ? nFeatures : _maxWorkItemsPerGroup;
    while (workItemsPerGroup & (workItemsPerGroup - 1))
    {
        workItemsPerGroup++;
    }
    if (nFeatures <= 32)
    {
        workItemsPerGroup = nFeatures;
    }
    else if (nFeatures <= 64)
    {
        workItemsPerGroup = nFeatures / 2;
        if (nFeatures % 2 > 0) workItemsPerGroup++;
    }
    else if (nFeatures <= 128)
    {
        workItemsPerGroup = nFeatures / 4;
        if (nFeatures % 4 > 0) workItemsPerGroup++;
    }
    return workItemsPerGroup;
}

template <typename algorithmFPType>
const char * KMeansDenseLloydKernelBaseUCAPI<algorithmFPType>::getComputeSquaresKernelName(uint32_t nFeatures)
{
    if (nFeatures <= 32)
    {
        return "compute_squares_32";
    }
    else if (nFeatures <= 64)
    {
        return "compute_squares_64";
    }
    else if (nFeatures <= 128)
    {
        return "compute_squares_128";
    }
    return "compute_squares";
}

template <typename algorithmFPType>
void KMeansDenseLloydKernelBaseUCAPI<algorithmFPType>::computeSquares(ExecutionContextIface & context, const KernelPtr & kernel_compute_squares,
                                                                       const Buffer<algorithmFPType> & data, UniversalBuffer & dataSq, uint32_t nRows,
                                                                       uint32_t nFeatures, Status * st)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.computeSquares);

    KernelArguments args(4);
    args.set(0, data, AccessModeIds::read);
    args.set(1, dataSq, AccessModeIds::write);
    args.set(2, nRows);
    args.set(3, nFeatures);

    size_t workItemsPerGroup = getComputeSquaresWorkgroupsCount(nFeatures);

    KernelRange local_range(1, workItemsPerGroup);
    KernelRange global_range(nRows, workItemsPerGroup);

    KernelNDRange range(2);
    range.global(global_range, st);
    DAAL_CHECK_STATUS_PTR(st);
    range.local(local_range, st);
    DAAL_CHECK_STATUS_PTR(st);

    {
        DAAL_ITTNOTIFY_SCOPED_TASK(compute.computeSquares.run);
        context.run(range, kernel_compute_squares, args, st);
    }
}

template <typename algorithmFPType>
void KMeansDenseLloydKernelBaseUCAPI<algorithmFPType>::initDistances(ExecutionContextIface & context, const KernelPtr & kernel_init_distances,
                                                                      UniversalBuffer & centroidsSq, UniversalBuffer & distances, uint32_t blockSize,
                                                                      uint32_t nClusters, Status * st)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.initDistances);

    KernelArguments args(4);
    args.set(0, centroidsSq, AccessModeIds::read);
    args.set(1, distances, AccessModeIds::write);
    args.set(2, blockSize);
    args.set(3, nClusters);

    size_t workgroupsCount = getWorkgroupsCount(blockSize);

    KernelRange local_range(_maxWorkItemsPerGroup, 1);
    KernelRange global_range(workgroupsCount * _maxWorkItemsPerGroup, nClusters);

    KernelNDRange range(2);
    range.global(global_range, st);
    DAAL_CHECK_STATUS_PTR(st);
    range.local(local_range, st);
    DAAL_CHECK_STATUS_PTR(st);

    {
        DAAL_ITTNOTIFY_SCOPED_TASK(compute.initDistances.run);
        context.run(range, kernel_init_distances, args, st);
    }
}

template <typename algorithmFPType>
void KMeansDenseLloydKernelBaseUCAPI<algorithmFPType>::computeDistances(ExecutionContextIface & context, const Buffer<algorithmFPType> & data,
                                                                         const Buffer<algorithmFPType> & centroids, UniversalBuffer & distances,
                                                                         uint32_t blockSize, uint32_t nClusters, uint32_t nFeatures, Status * st)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.computeDistances);

    auto gemmStatus = BlasGpu<algorithmFPType>::xgemm(math::Layout::ColMajor, math::Transpose::Trans, math::Transpose::NoTrans, blockSize, nClusters,
                                                      nFeatures, algorithmFPType(-1.0), data, nFeatures, 0, centroids, nFeatures, 0,
                                                      algorithmFPType(1.0), distances.get<algorithmFPType>(), blockSize, 0);

    if (st != nullptr)
    {
        *st = gemmStatus;
    }
}

template <typename algorithmFPType>
void KMeansDenseLloydKernelBaseUCAPI<algorithmFPType>::computeAssignments(ExecutionContextIface & context,
                                                                           const KernelPtr & kernel_compute_assignments, UniversalBuffer & distances,
                                                                           const Buffer<int> & assignments, UniversalBuffer & mindistances,
                                                                           uint32_t blockSize, uint32_t nClusters, Status * st)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.computeAssignments);

    KernelArguments args(5);
    args.set(0, distances, AccessModeIds::read);
    args.set(1, assignments, AccessModeIds::write);
    args.set(2, mindistances, AccessModeIds::write);
    args.set(3, blockSize);
    args.set(4, nClusters);

    KernelRange local_range(1, _preferableSubGroup);
    KernelRange global_range(blockSize, _preferableSubGroup);

    KernelNDRange range(2);
    range.global(global_range, st);
    DAAL_CHECK_STATUS_PTR(st);
    range.local(local_range, st);
    DAAL_CHECK_STATUS_PTR(st);

    {
        DAAL_ITTNOTIFY_SCOPED_TASK(compute.computeAssignments.run);
        context.run(range, kernel_compute_assignments, args, st);
    }
}

template <typename algorithmFPType>
void KMeansDenseLloydKernelBaseUCAPI<algorithmFPType>::computePartialCandidates(
    ExecutionContextIface & context, const KernelPtr & kernel_partial_candidates, const Buffer<int> & assignments, UniversalBuffer & mindistances,
    UniversalBuffer & dataSq, UniversalBuffer & candidates, UniversalBuffer & candidateDistances, UniversalBuffer & partialCandidates,
    UniversalBuffer & partialCandidateDistances, uint32_t blockSize, uint32_t nClusters, uint32_t reset, Status * st)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.computePartialCandidates);

    KernelArguments args(10);
    args.set(0, assignments, AccessModeIds::read);
    args.set(1, mindistances, AccessModeIds::read);
    args.set(2, dataSq, AccessModeIds::read);
    args.set(3, candidates, AccessModeIds::read);
    args.set(4, candidateDistances, AccessModeIds::read);
    args.set(5, partialCandidates, AccessModeIds::write);
    args.set(6, partialCandidateDistances, AccessModeIds::write);
    args.set(7, blockSize);
    args.set(8, nClusters);
    args.set(9, reset);

    int num_parts = getCandidatePartNum(nClusters);
    if (num_parts > _preferableSubGroup) num_parts = _preferableSubGroup;
    KernelRange local_range(1, _preferableSubGroup);
    KernelRange global_range(num_parts, _preferableSubGroup);

    KernelNDRange range(2);
    range.global(global_range, st);
    DAAL_CHECK_STATUS_PTR(st);
    range.local(local_range, st);
    DAAL_CHECK_STATUS_PTR(st);

    {
        DAAL_ITTNOTIFY_SCOPED_TASK(compute.computePartialCandidates.run);
        context.run(range, kernel_partial_candidates, args, st);
    }
}

template <typename algorithmFPType>
void KMeansDenseLloydKernelBaseUCAPI<algorithmFPType>::mergePartialCandidates(
    ExecutionContextIface & context, const KernelPtr & kernel_merge_candidates, UniversalBuffer & candidates, UniversalBuffer & candidateDistances,
    UniversalBuffer & partialCandidates, UniversalBuffer & partialCandidateDistances, uint32_t nClusters, Status * st)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.mergePartialCandidates);

    KernelArguments args(5);
    args.set(0, candidates, AccessModeIds::write);
    args.set(1, candidateDistances, AccessModeIds::write);
    args.set(2, partialCandidates, AccessModeIds::read);
    args.set(3, partialCandidateDistances, AccessModeIds::read);
    args.set(4, (int)nClusters);

    int num_parts = getCandidatePartNum(nClusters);
    if (num_parts > _preferableSubGroup) num_parts = _preferableSubGroup;
    KernelRange local_range(1, num_parts);
    KernelRange global_range(1, num_parts);

    KernelNDRange range(2);
    range.global(global_range, st);
    DAAL_CHECK_STATUS_PTR(st);
    range.local(local_range, st);
    DAAL_CHECK_STATUS_PTR(st);

    {
        DAAL_ITTNOTIFY_SCOPED_TASK(compute.mergePartialCandidates.run);
        context.run(range, kernel_merge_candidates, args, st);
    }
    DAAL_CHECK_STATUS_PTR(st);
}

template <typename algorithmFPType>
void KMeansDenseLloydKernelBaseUCAPI<algorithmFPType>::partialReduceCentroids(
    ExecutionContextIface & context, const KernelPtr & kernel_partial_reduce_centroids, const Buffer<algorithmFPType> & data,
    UniversalBuffer & distances, const Buffer<int> & assignments, UniversalBuffer & partialCentroids, UniversalBuffer & partialCentroidsCounters,
    uint32_t blockSize, uint32_t nClusters, uint32_t nFeatures, uint32_t nPartialCentroids, uint32_t doReset, Status * st)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.partialReduceCentroids);

    KernelArguments args(9);
    args.set(0, data, AccessModeIds::read);
    args.set(1, distances, AccessModeIds::read);
    args.set(2, assignments, AccessModeIds::read);
    args.set(3, partialCentroids, AccessModeIds::write);
    args.set(4, partialCentroidsCounters, AccessModeIds::write);
    args.set(5, blockSize);
    args.set(6, nClusters);
    args.set(7, nFeatures);
    args.set(8, doReset);

    KernelRange global_range(nPartialCentroids * nFeatures);

    {
        DAAL_ITTNOTIFY_SCOPED_TASK(compute.partialReduce.run);
        context.run(global_range, kernel_partial_reduce_centroids, args, st);
    }
}

template <typename algorithmFPType>
void KMeansDenseLloydKernelBaseUCAPI<algorithmFPType>::mergeReduceCentroids(ExecutionContextIface & context,
                                                                             const KernelPtr & kernel_merge_reduce_centroids,
                                                                             UniversalBuffer & partialCentroids,
                                                                             UniversalBuffer & partialCentroidsCounters,
                                                                             const Buffer<algorithmFPType> & centroids, uint32_t nClusters,
                                                                             uint32_t nFeatures, uint32_t nPartialCentroids, Status * st)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.mergeReduceCentroids);

    KernelArguments args(6);
    args.set(0, partialCentroids, AccessModeIds::readwrite);
    args.set(1, partialCentroidsCounters, AccessModeIds::readwrite);
    args.set(2, centroids, AccessModeIds::write);
    args.set(3, nClusters);
    args.set(4, nFeatures);
    args.set(5, nPartialCentroids);

    KernelRange local_range(nPartialCentroids);
    KernelRange global_range(nPartialCentroids * nClusters);

    KernelNDRange range(1);
    range.global(global_range, st);
    DAAL_CHECK_STATUS_PTR(st);
    range.local(local_range, st);
    DAAL_CHECK_STATUS_PTR(st);

    {
        DAAL_ITTNOTIFY_SCOPED_TASK(compute.mergeReduceCentroids.run);
        context.run(range, kernel_merge_reduce_centroids, args, st);
    }
}

template <typename algorithmFPType>
void KMeansDenseLloydKernelBaseUCAPI<algorithmFPType>::updateObjectiveFunction(ExecutionContextIface & context,
                                                                                const KernelPtr & kernel_update_objective_function,
                                                                                UniversalBuffer & dataSq, UniversalBuffer & distances,
                                                                                const Buffer<int> & assignments,
                                                                                const Buffer<algorithmFPType> & objFunction, uint32_t blockSize,
                                                                                uint32_t nClusters, uint32_t doReset, Status * st)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.updateObjectiveFunction);

    if (doReset)
    {
        auto hostPtr = objFunction.toHost(data_management::writeOnly);
        *hostPtr     = 0.0f;
    }

    KernelArguments args(6);
    args.set(0, dataSq, AccessModeIds::read);
    args.set(1, distances, AccessModeIds::read);
    args.set(2, assignments, AccessModeIds::read);
    args.set(3, objFunction, AccessModeIds::write);
    args.set(4, (int)blockSize);
    args.set(5, (int)nClusters);

    KernelRange local_range(_maxWorkItemsPerGroup);
    KernelRange global_range(_maxWorkItemsPerGroup);

    KernelNDRange range(1);
    range.global(global_range, st);
    DAAL_CHECK_STATUS_PTR(st);
    range.local(local_range, st);
    DAAL_CHECK_STATUS_PTR(st);

    {
        DAAL_ITTNOTIFY_SCOPED_TASK(compute.updateObjectiveFunction.run);
        context.run(range, kernel_update_objective_function, args, st);
    }
}

} // namespace internal
} // namespace kmeans
} // namespace algorithms
} // namespace daal

#endif
