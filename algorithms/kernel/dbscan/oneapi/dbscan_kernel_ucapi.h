/* file: dbscan_kernel_ucapi.h */
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
//  Declaration of template function that computes DBSCAN for GPU.
//--
*/

#ifndef __DBSCAN_KERNEL_UCAPI_H
#define __DBSCAN_KERNEL_UCAPI_H

#include "dbscan_types.h"
#include "kernel.h"
#include "numeric_table.h"
#include "dbscan_utils_ucapi.h"
#include "oneapi/internal/execution_context.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace dbscan
{
namespace internal
{

template <typename algorithmFPType>
class DBSCANBatchKernelUCAPI : public Kernel
{
public:
    services::Status compute(const NumericTable * ntData, const NumericTable * ntWeights, NumericTable * ntAssignments,
                                    NumericTable * ntNClusters, NumericTable * ntCoreIndices, NumericTable * ntCoreObservations,
                                    const Parameter * par);

private:
    services::Status processNeighborhood(size_t clusterId, int * assignments, const NeighborhoodUCAPI<algorithmFPType> & neigh,
                                         QueueUCAPI<size_t> & qu);

    services::Status processResultsToCompute(DAAL_UINT64 resultsToCompute, int * const isCore, NumericTable * ntData,
                                             NumericTable * ntCoreIndices, NumericTable * ntCoreObservations);
    services::Status processRowNbrs(
        const oneapi::internal::UniversalBuffer& rowDistances,
        const oneapi::internal::UniversalBuffer& offsets,
        uint32_t rowId,
        uint32_t clusterId, 
        uint32_t chunkOffset,
        uint32_t numberOfChunks,
        uint32_t nRows,
        uint32_t qEnd,
        algorithmFPType eps,
        oneapi::internal::UniversalBuffer& assignments,
        oneapi::internal::UniversalBuffer& queue);

    services::Status countOffsets(
        const oneapi::internal::UniversalBuffer& counters,
        uint32_t numberOfChunks,
        oneapi::internal::UniversalBuffer& offsets);

    services::Status setBufferValue(
        oneapi::internal::UniversalBuffer& buffer,
        uint32_t index,
        int value); 
 
    services::Status setBufferValueByQueueIndex(
        oneapi::internal::UniversalBuffer& buffer,
        const oneapi::internal::UniversalBuffer& queue,
        uint32_t posInQueue,
        int value); 

    services::Status queryRow(
        const oneapi::internal::UniversalBuffer& data,
        uint32_t nRows, 
        uint32_t rowId,
        uint32_t dim, 
        uint32_t minkowskiPower,
        oneapi::internal::UniversalBuffer& rowDistances);

    services::Status queryQueueRows(
        const oneapi::internal::UniversalBuffer& data,
        uint32_t nRows, 
        const oneapi::internal::UniversalBuffer& queue,
        uint32_t queueBegin, 
        uint32_t queueBlockSize,
        uint32_t dim, 
        uint32_t minkowskiPower,
        oneapi::internal::UniversalBuffer& rowDistances);

    services::Status countNbrs(
        const oneapi::internal::UniversalBuffer& assignments,
        const oneapi::internal::UniversalBuffer& RowDistances,
        uint32_t rowId,
        size_t chunkOffset, 
        size_t nRows,
        size_t numberOfChunks,
        algorithmFPType eps,
        oneapi::internal::UniversalBuffer& counters,
        oneapi::internal::UniversalBuffer& undefCounters);

    uint32_t sumCounters(
        const oneapi::internal::UniversalBuffer& counters,
        uint32_t numberOfChunks);


    services::Status buildProgram(
        oneapi::internal::ClKernelFactoryIface & kernel_factory);

    size_t _minSgSize = 16;
    size_t _maxWgSize = 256;
    size_t _chunkNumber = 64;
    size_t _queueBlockSize = 64;
};

} // namespace internal
} // namespace dbscan
} // namespace algorithms
} // namespace daal

#endif
