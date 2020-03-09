/* file: dbscan_dense_default_batch_ucapi_impl */
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
//  Implementation of default method for DBSCAN algorithm on GPU.
//--
*/

#include "algorithm.h"
#include "numeric_table.h"

#include "dbscan_types.h"
#include "dbscan_kernel_ucapi.h"

#include "oneapi/sum_reducer.h"

#include "oneapi/service_defines_oneapi.h"
#include "oneapi/blas_gpu.h"

#include "cl_kernels/dbscan_cl_kernels.cl"

#include "service_ittnotify.h"
#include <iostream>

using namespace daal::internal;
using namespace daal::services;
using namespace daal::services::internal;
using namespace daal::oneapi::internal;

namespace daal
{
namespace algorithms
{
namespace dbscan
{
namespace internal
{

#define __DBSCAN_PREFETCHED_UCAPI_NEIGHBORHOODS_COUNT 256

template <typename algorithmFPType>
Status DBSCANBatchKernelUCAPI<algorithmFPType>::processNeighborhood(size_t clusterId, int * const assignments,
                                                                            const NeighborhoodUCAPI<algorithmFPType> & neigh, QueueUCAPI<size_t> & qu)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.processNeighborhood);    
    for (size_t j = 0; j < neigh.size(); j++)
    {
        const size_t nextObs = neigh.get(j);
        if (assignments[nextObs] == noise)
        {
            assignments[nextObs] = clusterId;
        }
        else if (assignments[nextObs] == undefined)
        {
            assignments[nextObs] = clusterId;
            DAAL_CHECK_STATUS_VAR(qu.push(nextObs));
        }
    }

    return services::Status();
}

template <typename algorithmFPType>
Status DBSCANBatchKernelUCAPI<algorithmFPType>::processResultsToCompute(DAAL_UINT64 resultsToCompute, int * const isCore,
                                                                                NumericTable * ntData, NumericTable * ntCoreIndices,
                                                                                NumericTable * ntCoreObservations)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.processResultsToCompute);
    auto& context = Environment::getInstance()->getDefaultExecutionContext();
    const size_t nRows     = ntData->getNumberOfRows();
    const size_t nFeatures = ntData->getNumberOfColumns();

    size_t nCoreObservations = 0;

    for (size_t i = 0; i < nRows; i++)
    {
        if (!isCore[i])
        {
            continue;
        }
        nCoreObservations++;
    }

    if (nCoreObservations == 0)
    {
        return Status();
    }

    if (resultsToCompute & computeCoreIndices)
    {
        DAAL_CHECK_STATUS_VAR(ntCoreIndices->resize(nCoreObservations));
        BlockDescriptor<int> indexRows;
        DAAL_CHECK_STATUS_VAR(ntCoreIndices->getBlockOfRows(0, nCoreObservations, writeOnly, indexRows));
        auto coreIndices = indexRows.getBuffer().toHost(ReadWriteMode::writeOnly).get();

        size_t pos = 0;
        for (size_t i = 0; i < nRows; i++)
        {
            if (!isCore[i])
            {
                continue;
            }
            coreIndices[pos] = i;
            pos++;
        }
    }

    if (resultsToCompute & computeCoreObservations)
    {
        DAAL_CHECK_STATUS_VAR(ntCoreObservations->resize(nCoreObservations));
        BlockDescriptor<algorithmFPType> coreObservationsRows;
        DAAL_CHECK_STATUS_VAR(ntCoreObservations->getBlockOfRows(0, nCoreObservations, writeOnly, coreObservationsRows));
        auto coreObservations = coreObservationsRows.getBuffer();

        size_t pos = 0;
        int result = 0;
        for (size_t i = 0; i < nRows; i++)
        {
            if (!isCore[i])
            {
                continue;
            }
            BlockDescriptor<algorithmFPType> dataRows;
            DAAL_CHECK_STATUS_VAR(ntData->getBlockOfRows(i, 1, readOnly, dataRows));
            auto data = dataRows.getBuffer();

            Status st;
            context.copy(UniversalBuffer(coreObservations), pos * nFeatures, UniversalBuffer(data), 0, nFeatures, &st);
            DAAL_CHECK_STATUS_VAR(st);
            pos++;
            DAAL_CHECK_STATUS_VAR(ntData->releaseBlockOfRows(dataRows));
        }
        if (result)
        {
            return Status(services::ErrorMemoryCopyFailedInternal);
        }
    }

    return Status();
}

template <typename algorithmFPType>
Status DBSCANBatchKernelUCAPI<algorithmFPType>::compute(const NumericTable * x, const NumericTable * ntWeights,
                                                                       NumericTable * ntAssignments, NumericTable * ntNClusters,
                                                                       NumericTable * ntCoreIndices, NumericTable * ntCoreObservations,
                                                                       const Parameter * par)
{

    Status s;
    auto& context = Environment::getInstance()->getDefaultExecutionContext();
    auto& kernel_factory = context.getClKernelFactory();    

    const algorithmFPType epsilon         = par->epsilon;
    const algorithmFPType minObservations = par->minObservations;
    const algorithmFPType minkowskiPower  = (algorithmFPType)2.0;

    NumericTable *ntData = const_cast<NumericTable *>( x );
    BlockDescriptor<algorithmFPType> dataRows;
    ntData->getBlockOfRows(0, nRows, readOnly, dataRows);
    auto data = dataRows.getBuffer();
    const size_t nRows = ntData->getNumberOfRows();
    const size_t dim = ntData->getNumberOfColumns();

    BlockDescriptor<int> assignRows;
    DAAL_CHECK_STATUS_VAR(ntAssignments->getBlockOfRows(0, nRows, readOnly, assignRows));
    {
        auto assignments = assignRows.getBuffer();
        {
            auto writeAssignments = assignments.toHost(ReadWriteMode::writeOnly);
            for(uint32_t i = 0; i < nRows; i++)
            {
                writeAssignments.get()[i] = undefined;
            }
        }
    }

    auto rowDistances = context.allocate(TypeIds::id<algorithmFPType>(), _queueBlockSize * nRows, &s);
    DAAL_CHECK_STATUS_VAR(s);
    auto singleRowDistances = context.allocate(TypeIds::id<algorithmFPType>(), nRows, &s);
    DAAL_CHECK_STATUS_VAR(s);
    auto isCore = context.allocate(TypeIds::id<int>(), nRows, &s);
    DAAL_CHECK_STATUS_VAR(s);
    auto queueEnd = context.allocate(TypeIds::id<int>(), 1, &s);
    DAAL_CHECK_STATUS_VAR(s);
    auto counters = context.allocate(TypeIds::id<int>(), _chunkNumber, &s);
    DAAL_CHECK_STATUS_VAR(s);
    auto undefCounters = context.allocate(TypeIds::id<int>(), _chunkNumber, &s);
    DAAL_CHECK_STATUS_VAR(s);
    auto offsets = context.allocate(TypeIds::id<int>(), _chunkNumber, &s);
    DAAL_CHECK_STATUS_VAR(s);

    {
        auto isCorePtr = isCore.toHost(ReadWriteMode::writeOnly).get();
        for(uint32_t i = 0; i < nRows; i++)
        {
            isCorePtr[i] = 0;
        }
    }

    size_t nClusters = 0;
    uint32_t qBegin = 0;
    uint32_t qEnd   = 0;
    {
        queueEnd.template get<int>().toHost(ReadWriteMode::writeOnly).get()[0] = qEnd;
    }

    for(int = 0; i < nRows; i++)
    {
        if(assignments[i] != undefined)
            continue;
        queryRow(data, nRows, i, dim, minkowskiPower, singleRowDistances);
        countNbrs(assignments, singleRowDistances, 0, nRows, _chunkNumber, eps, counters, undefCounters);
        uint32_t totalNbrs = sumCounters(counters, _chunkNumber);
        if(totalNbrs < minObservations)
        {
            markBufferValue(assignments, i, noise);
            continue;
        }
        nClusters++;
        setBufferValue(isCore, i, 1);
        countOffsets(undefCounters, _chunkNumber, offsets);
        processRowNbrs(assignments, i, nClusters - 1, rowDistances, 0, nRows, _chunkNumber, eps, queueEnd, offsets, queue);
        {
            qEnd = queueEnd.template get<int>().toHost(ReadWriteMode::readOnly).get()[0];
        }
        const uint32_t queueBlockSize = 64;
        while(qBegin < qEnd)
        {
            uint32_t curQueueBlockSize = qEnd - qBegin;
            if( curQueueBlockSize > queueBlockSize)
            {
                curQueueBlockSize = queueBlockSize;
            }
            queryQueueRows(data, nRows, queue, qBegin, queueBlockSize, dim, minkowskiPower, rowDistances);
            for(int i = 0; i < queueBlockSize)
            {
               countNbrs(assignments, rowDistances, nRows * i, nRows, _chunkNumber, eps, counters, undefCounters);
               uint32_t curNbrs = sumCounters(counters, _chunkNumber);
               if(totalNbrs < minObservations)
               {
                   setBufferValueByQueueIndex(assignments, queue, queueBegin + i, noise);
                   continue;
               }
               setBufferValueByQueueIndex(isCore, queue, queueBegin + i, 1);
               countOffsets(undefCounters, _chunkNumber, offsets);
               processNbrs(assignments, nClusters - 1, rowDistances, nRows * i, nRows, _chunkNumber, eps, queueEnd, offsets, queue, queueEnd);
            }
            qBegin += curQueueBlockSize;
        }
    }
    ntData->releaseBlockOfRows(dataRows);
    /*
    BlockDescriptor<int> nClustersRows;
    DAAL_CHECK_STATUS_VAR(ntNClusters->getBlockOfRows(0, 1, writeOnly, nClustersRows));
    nClustersRows.getBuffer().toHost(ReadWriteMode::writeOnly).get()[0] = nClusters;

    if (par->resultsToCompute & (computeCoreIndices | computeCoreObservations))
    {
        DAAL_CHECK_STATUS_VAR(processResultsToCompute(par->resultsToCompute, isCore, ntData, ntCoreIndices, ntCoreObservations));
    }*/
    std::cout << "Clusters: " << nClusters << std::endl;
    return s;

}

template <typename algorithmFPType>
services::Status DBSCANBatchKernelUCAPI<algorithmFPType>::processRowNbrs(
        const UniversalBuffer& rowDistances,
        const UniversalBuffer& offsets,
        uint32_t rowId,
        uint32_t clusterId, 
        uint32_t chunkOffset,
        uint32_t numberOfChunks,
        uint32_t nRows,
        algorithmFPType eps,
        UniversalBuffer& assignments,
        UniversalBuffer& queue,
        UniversalBuffer& queueEnd)
{
    services::Status st;
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.processRowNbrs);
    auto& context = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernel_factory = context.getClKernelFactory();
    buildProgram(kernel_factory, st);
    DAAL_CHECK_STATUS_PTR(st);
    auto kernel = kernel_factory.getKernel("process_neighbors", &st);
    DAAL_CHECK_STATUS_PTR(st);

    uint32_t chunkSize = nRows / numberOfChunks + uint32_t(bool(nRows % numberOfChunks));

    KernelArguments args(11);
    args.set(0, rowDistances, AccessModeIds::read);
    args.set(1, offset, AccessModeIds::read);
    args.set(2, assignmets, AccessModeIds::readWrite);
    args.set(3, queue, AccessModeIds::readWrite);
    args.set(4, queueEnd, AccessModeIds::readWrite);
    args.set(5, rowId);
    args.set(6, clusterId);
    args.set(7, chunkOffset);
    args.set(8, chunkSize);
    args.set(9, eps);
    args.set(10, nRows);

    KernelRange local_range(1, _maxWgSize);
    KernelRange global_range(nRows / _minSgSize + 1, _maxWgSize);

    KernelNDRange range(2);
    range.global(global_range, st); DAAL_CHECK_STATUS_PTR(st);
    range.local(local_range, st); DAAL_CHECK_STATUS_PTR(st);

    context.run(range, kernel, args, st);
    return st;
}

template <typename algorithmFPType>
services::Status DBSCANBatchKernelUCAPI<algorithmFPType>::countOffsets(
        const UniversalBuffer& counters,
        uint32_t numberOfChunks,
        const UniversalBuffer& offsets) 
{
    services::Status st;
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.countOffsets);
    auto& context = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernel_factory = context.getClKernelFactory();
    buildProgram(kernel_factory, st);
    DAAL_CHECK_STATUS_PTR(st);
    auto kernel = kernel_factory.getKernel("count_offsets", &st);
    DAAL_CHECK_STATUS_PTR(st);

    KernelArguments args(2);
    args.set(0, counters, AccessModeIds::readOnly);
    args.set(1, offsets, AccessModeIds::writeOnly);
    args.set(2, numberOfChunks);

    KernelRange local_range(1, _minSgSize);
    context.run(local_range, kernel, args, st);
    return st;
}

template <typename algorithmFPType>
services::Status DBSCANBatchKernelUCAPI<algorithmFPType>::setBufferValue(
        UniversalBuffer& buffer,
        uint32_t index,
        int value) 
{
    services::Status st;
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.setBufferValue);
    auto& context = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernel_factory = context.getClKernelFactory();
    buildProgram(kernel_factory, st);
    DAAL_CHECK_STATUS_PTR(st);
    auto kernel = kernel_factory.getKernel("set_buffer_value", &st);
    DAAL_CHECK_STATUS_PTR(st);

    KernelArguments args(2);
    args.set(0, buffer, AccessModeIds::readWrite);
    args.set(1, index);
    args.set(2, value);

    KernelRange global_range(1);
    context.run(global_range, kernel, args, st);
    return st;
}

template <typename algorithmFPType>
services::Status DBSCANBatchKernelUCAPI<algorithmFPType>::setBufferValueByQueueIndex(
        UniversalBuffer& buffer,
        const UniversalBuffer& queue,
        uint32_t posInQueue,
        int value) 
{
    services::Status st;
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.setBufferValueByIndirectIndex);
    auto& context = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernel_factory = context.getClKernelFactory();
    buildProgram(kernel_factory, st);
    DAAL_CHECK_STATUS_PTR(st);
    auto kernel = kernel_factory.getKernel("set_buffer_value_by_queue_index", &st);
    DAAL_CHECK_STATUS_PTR(st);

    KernelArguments args(4);
    args.set(0, queue, AccessModeIds::readOnly);
    args.set(1, buffer, AccessModeIds::readWrite);
    args.set(2, posInQueue);
    args.set(3, value);

    KernelRange global_range(1);
    context.run(global_range, kernel, args, st);
    return st;
}

template <typename algorithmFPType>
services::Status DBSCANBatchKernelUCAPI<algorithmFPType>::queryRow(
        const UniversalBuffer& data,
        uint32_t nRows, 
        uint32_t rowId,
        uint32_t dim, 
        uint32_t minkowskiPower,
        UniversalBuffer& rowDistances)
{
    services::Status st;
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.queryRow);
    auto& context = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernel_factory = context.getClKernelFactory();
    buildProgram(kernel_factory, st);
    DAAL_CHECK_STATUS_PTR(st);
    auto kernel = kernel_factory.getKernel("query_row", &st);
    DAAL_CHECK_STATUS_PTR(st);

    KernelArguments args(6);
    args.set(0, data, AccessModeIds::read);
    args.set(1, rowDistances, AccessModeIds::write);
    args.set(2, rowId);
    args.set(3, minkowskiPower);
    args.set(4, dim);
    args.set(5, nRows);

    KernelRange local_range(1, _maxWgSize);
    KernelRange global_range(nRows / _minSgSize + 1, _maxWgSize);

    KernelNDRange range(2);
    range.global(global_range, st); DAAL_CHECK_STATUS_PTR(st);
    range.local(local_range, st); DAAL_CHECK_STATUS_PTR(st);

    context.run(range, kernel, args, st);
    return st;
}

template <typename algorithmFPType>
Status DBSCANBatchKernelUCAPI<algorithmFPType>::queryQueueRows(
        const UniversalBuffer& data,
        uint32_t nRows, 
        const UniversalBuffer& queue,
        uint32_t queueBegin, 
        uint32_t queueBlockSize,
        uint32_t dim, 
        uint32_t minkowskiPower,
        UniversalBuffer& rowDistances)
{
    services::Status st;
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.queryRow);
    auto& context = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernel_factory = context.getClKernelFactory();
    buildProgram(kernel_factory, st);
    DAAL_CHECK_STATUS_PTR(st);
    auto kernel = kernel_factory.getKernel("query_queue", &st);
    DAAL_CHECK_STATUS_PTR(st);

    KernelArguments args(6);
    args.set(0, data, AccessModeIds::read);
    args.set(1, queue, AccessModeIds::read);
    args.set(2, rowDistances, AccessModeIds::write);
    args.set(2, queueBegin);
    args.set(2, queueBlockSize);
    args.set(3, minkowskiPower);
    args.set(4, dim);
    args.set(5, nRows);

    KernelRange local_range(1, _maxWgSize);
    KernelRange global_range(nRows / _minSgSize + 1, _maxWgSize);

    KernelNDRange range(2);
    range.global(global_range, st); DAAL_CHECK_STATUS_PTR(st);
    range.local(local_range, st); DAAL_CHECK_STATUS_PTR(st);

    context.run(range, kernel, args, st);
    return st;
}

template <typename algorithmFPType>
size_t DBSCANBatchKernelUCAPI<algorithmFPType>::countNbrs(
        const UniversalBuffer& assignments,
        const UniversalBuffer& RowDistances,
        size_t chunkOffset, 
        size_t nRows,
        size_t numberOfChunks,
        algorithmFPType eps,
        UniversalBuffer& counters,
        UniversalBuffer& undefCounters)
{
    services::Status st;
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.countNbrs);
    auto& context = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernel_factory = context.getClKernelFactory();
    buildProgram(kernel_factory, st);
    DAAL_CHECK_STATUS_PTR(st);
    auto kernel = kernel_factory.getKernel("count_neighbors", &st);
    DAAL_CHECK_STATUS_PTR(st);

    uint32_t chunkSize = nRows / numberOfChunks + uint32_t(bool(nRows % numberOfChunks));

    KernelArguments args(8);
    args.set(0, assignments, AccessModeIds::read);
    args.set(1, rowDistances, AccessModeIds::read);
    args.set(2, chunkOffset);
    args.set(3, chunkSize);
    args.set(4, nRows);
    args.set(5, eps);
    args.set(6, counters, AccessModeIds::write);
    args.set(7, counters, AccessModeIds::write);

    KernelRange local_range(1, _maxWgSize);
    KernelRange global_range(numberOfChunks, _maxWgSize);

    KernelNDRange range(2);
    range.global(global_range, st); DAAL_CHECK_STATUS_PTR(st);
    range.local(local_range, st); DAAL_CHECK_STATUS_PTR(st);

    context.run(range, kernel, args, st);
    return st;
}

template <typename algorithmFPType>
uin32_t DBSCANBatchKernelUCAPI<algorithmFPType>::sumCounters(
        const UniversalBuffer& counters,
        uint32_t numberOfChunks)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.sumCounters);
    auto cntPtr = counters.template get<int>().toHost(ReadWriteMode::writeOnly).get();
    uin32_t ret = 0;
    for(int i = 0; i < numberOfChunks; i++)
        ret += cntPtr[i];
    return ret;
}

template <typename algorithmFPType>
Status DBSCANBatchKernelUCAPI<algorithmFPType>::buildProgram(ClKernelFactoryIface & kernel_factory)
{
    Status st;
    auto fptype_name = oneapi::internal::getKeyFPType<algorithmFPType>();
    auto build_options = fptype_name;

    services::String cachekey("__daal_algorithms_dbscan_block_");
    cachekey.add(fptype_name);
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(compute.buildProgram);
        kernel_factory.build(ExecutionTargetIds::device, cachekey.c_str(), 
        dbscan_cl_kernels, build_options.c_str(), &st);
    }
    return st;
}


} // namespace internal
} // namespace dbscan
} // namespace algorithms
} // namespace daal
