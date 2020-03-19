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

#include "cl_kernels/dbscan_cl_kernels_2.cl"

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
    std::cout << "Compute" << std::endl;
    Status s;
    auto& context = Environment::getInstance()->getDefaultExecutionContext();
    auto& kernel_factory = context.getClKernelFactory();    

    const algorithmFPType epsilon         = par->epsilon;
    const algorithmFPType minObservations = par->minObservations;
    const uint32_t minkowskiPower  = (uint32_t)2.0;
    algorithmFPType eps = 1.0;
    for(uint32_t i = 0; i < minkowskiPower; i++)
        eps *= epsilon;
    std::cout << "EPS: " << eps << std::endl;

    NumericTable *ntData = const_cast<NumericTable *>( x );
    const size_t nRows = ntData->getNumberOfRows();
    std::cout << "Rows: " << nRows << std::endl;
    const size_t dim = ntData->getNumberOfColumns();

    BlockDescriptor<algorithmFPType> dataRows;
    ntData->getBlockOfRows(0, nRows, readOnly, dataRows);
    auto data = dataRows.getBuffer();

    BlockDescriptor<int> assignRows;
    DAAL_CHECK_STATUS_VAR(ntAssignments->getBlockOfRows(0, nRows, writeOnly, assignRows));
    auto assignments = daal::oneapi::internal::UniversalBuffer(assignRows.getBuffer());
    std::cout << "Init assignments: " << undefined << std::endl;
    {
        auto writeAssignments = assignments.template get<int>().toHost(ReadWriteMode::writeOnly);
        for(uint32_t i = 0; i < nRows; i++)
        {
            writeAssignments.get()[i] = undefined;
        }
    }
    {
        auto readAssignments = assignments.template get<int>().toHost(ReadWriteMode::readOnly);
        for(uint32_t i = 0; i < 20; i++)
        {
            std::cout << "Assigned: " << readAssignments.get()[i] << std::endl;
        }
    }
    auto rowDistances = context.allocate(TypeIds::id<algorithmFPType>(), _queueBlockSize * nRows, &s);
    DAAL_CHECK_STATUS_VAR(s);
    auto singleRowDistances = context.allocate(TypeIds::id<algorithmFPType>(), nRows, &s);
    DAAL_CHECK_STATUS_VAR(s);
    auto queue = context.allocate(TypeIds::id<int>(), nRows, &s);
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
    std::cout << "Chunk number: " << _chunkNumber << std::endl;
    std::cout << "Init isCore" << std::endl;
    {
        auto isCorePtr = isCore.template get<int>().toHost(ReadWriteMode::writeOnly).get();
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
    std::cout << "Main cycle: " << std::endl;
    for(int i = 0; i < nRows; i++)
    {
        {
            auto assignPtr = assignments.template get<int>().toHost(ReadWriteMode::writeOnly);
            if(assignPtr.get()[i] != undefined)
                continue;
        }
        std::cout << "New point: " << i << std::endl;
        DAAL_CHECK_STATUS_VAR(queryRow(data, nRows, i, dim, minkowskiPower, singleRowDistances));
        {
            auto dists = singleRowDistances.template get<algorithmFPType>().toHost(ReadWriteMode::readOnly);
            std::cout << "Distances" << std::endl;
            for(int j = 0; j < 20; j++)
                std::cout << dists.get()[j] << " ";
            std::cout << std::endl;
            int nbrs = 0;
            for(int j = 0; j < nRows; j++)
                if(dists.get()[j] < eps) 
                    nbrs++;
            for(int j = 0; j < nRows; j++)
                if(dists.get()[j] < eps) 
                    std::cout << j << " ";
            std::cout << std::endl;

            std::cout << "Real nbrs: " << nbrs << std::endl;
        }
        DAAL_CHECK_STATUS_VAR(countNbrs(assignments, singleRowDistances, i, -1, nRows, _chunkNumber, eps, queue, counters, undefCounters)); //done
        {
            auto cntr = counters.template get<int>().toHost(ReadWriteMode::readOnly);
            auto newcntr = undefCounters.template get<int>().toHost(ReadWriteMode::readOnly);
            for(int j = 0; j < _chunkNumber; j++)
                std::cout << "Cnt: " << cntr.get()[j] << " " << newcntr.get()[j] << std::endl;
         }
        uint32_t totalNbrs = sumCounters(counters, _chunkNumber); //done
        uint32_t newNbrs = sumCounters(undefCounters, _chunkNumber); //done
        std::cout << "Nbrs: " << totalNbrs << std::endl;
        if(totalNbrs < minObservations)
        {
            std::cout << "Let's skip" << std::endl;
            DAAL_CHECK_STATUS_VAR(setBufferValue(assignments, i, noise)); //Add
            std::cout << "Skipperd" << std::endl;
            continue;
        }
        nClusters++;
        std::cout << "New cluster" << std::endl;
        DAAL_CHECK_STATUS_VAR(setBufferValue(isCore, i, 1)); //Add
        std::cout << "Core is set" << std::endl;
        DAAL_CHECK_STATUS_VAR(countOffsets(undefCounters, _chunkNumber, offsets));
        {
            auto cntr = counters.template get<int>().toHost(ReadWriteMode::readOnly);
            auto offs = offsets.template get<int>().toHost(ReadWriteMode::readOnly);
            int off = 0;
            for(int j = 0; j < _chunkNumber; j++) {
                std::cout << "Offset: " << j << " " << off << " " << offs.get()[j] << std::endl;
                off += cntr.get()[j];
            }
        }
        
        DAAL_CHECK_STATUS_VAR(processRowNbrs(singleRowDistances, offsets, i, nClusters - 1, 0, 
                                            _chunkNumber, nRows, qEnd, eps, assignments, queue));
/*        {
            auto queueread = queue.template get<int>().toHost(ReadWriteMode::readOnly);
            for(int j = 0; j < newNbrs; j++)
                std::cout << "Nbr: " << queueread.get()[j] << std::endl;
        }*/
        qEnd += newNbrs;
        /*{
            uint32_t newNbrs = sumCounters(undefCounters, _chunkNumber);
            auto queueEndPtr = queueEnd.template get<int>().toHost(ReadWriteMode::readWrite);
            queueEndPtr.get()[0] += newNbrs;
            qEnd = queueEndPtr.get()[0];
        }*/

        while(qBegin < qEnd)
        {
            std::cout << "Queue: " << qBegin << " " << qEnd << std::endl;
            uint32_t curQueueBlockSize = qEnd - qBegin;
            if( curQueueBlockSize > _queueBlockSize)
            {
                curQueueBlockSize = _queueBlockSize;
            }
            std::cout << "curQueueBlockSize: " << curQueueBlockSize << std::endl;
            queryQueueRows(data, nRows, queue, qBegin, curQueueBlockSize, dim, minkowskiPower, rowDistances);
            {
                auto rd = rowDistances.template get<algorithmFPType>().toHost(ReadWriteMode::readOnly);
                /*for(int j = 0; j < curQueueBlockSize; j++)
                {
                    for(int k = 0; k < 20; k++)
                        std::cout << rd.get()[nRows * j + k] << " ";
                    std::cout << std::endl;
                } */
                auto hd = data.toHost(ReadWriteMode::readOnly);
                auto qu = queue.template get<int>().toHost(ReadWriteMode::readOnly);
                std::cout << "Queue: ";
                for(int j = 0; j < curQueueBlockSize; j++)
                    std::cout << qu.get()[j] << " ";
                std::cout << std::endl;
                for(int j = 0; j < curQueueBlockSize; j++)
                {
                    int quid = qu.get()[j];
                    std::cout << "quid: " << quid << std::endl;
                    int id = 937;
                    algorithmFPType trueDist = 0.0;
                    for(int k = 0; k < dim; k++) {
                        algorithmFPType val = 1.0;
                        algorithmFPType diff = fabs(hd.get()[quid * dim + k] - hd.get()[id * dim + k]);
                        for(int l = 0; l < minkowskiPower; l++)
                            val *= diff;
                        trueDist += val;
                    }
                    std::cout << "Comparison: " << j << " " << quid << " " << rd.get()[ j * nRows + id] << " " << trueDist << std::endl;


                }
            }
            
            for(int j = 0; j < curQueueBlockSize; j++)
            {
               countNbrs(assignments, rowDistances, qBegin + j, nRows * j, nRows, _chunkNumber, eps, queue, counters, undefCounters);
               uint32_t curNbrs = sumCounters(counters, _chunkNumber);
               uint32_t curNewNbrs = sumCounters(undefCounters, _chunkNumber); //done
               std::cout << "Nbrs: " << curNbrs << " " << curNewNbrs << std::endl;
               DAAL_CHECK_STATUS_VAR(queryRow(data, nRows, i, dim, minkowskiPower, singleRowDistances));
               /*{
                    auto dists = rowDistances.template get<algorithmFPType>().toHost(ReadWriteMode::readOnly);
                    std::cout << "Distances" << std::endl;
                    for(int j = 0; j < 20; j++)
                        std::cout << dists.get()[j] << " ";
                    std::cout << std::endl;
                    int nbrs = 0;
                    for(int j = 0; j < nRows; j++)
                        if(dists.get()[j] < eps) 
                            nbrs++;
                    for(int j = 0; j < nRows; j++)
                        if(dists.get()[j] < eps) 
                            std::cout << j << " ";
                    std::cout << std::endl;

                    std::cout << "Real nbrs: " << nbrs << std::endl;
                }*/
                if(totalNbrs < minObservations)
                {
                    setBufferValueByQueueIndex(assignments, queue, qBegin + j, noise); //Add
                    continue;
                }
                setBufferValueByQueueIndex(isCore, queue, qBegin + j, 1); //Add
                if(curNewNbrs > 0) {
                    std::cout << "Proessing queue" << std::endl;
                    countOffsets(undefCounters, _chunkNumber, offsets);
                    processRowNbrs(rowDistances, offsets, qBegin + j, nClusters - 1, nRows * j, _chunkNumber, nRows, qEnd, eps, assignments, queue);
                }
                else {
                    setBufferValueByQueueIndex(assignments, queue, qBegin + j, nClusters - 1); //Add
                }
                qEnd += curNewNbrs;
                std::cout << "Qend: " << qEnd << std::endl;
                {
                    auto cl = assignments. template get<int>().toHost(ReadWriteMode::readOnly);
                    auto cr = isCore. template get<int>().toHost(ReadWriteMode::readOnly);
                    for(int j = 0; j < 20; j++)
                        std::cout << "id " << j << " " << cl.get()[j] << " " << cr.get()[j] << " ";
                    std::cout << std::endl;
                }
            }
            qBegin += curQueueBlockSize;
            std::cout << "Queue iteration done: " << qBegin << " " << qEnd << std::endl;
            exit(0);
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
        uint32_t qEnd,
        algorithmFPType eps,
        UniversalBuffer& assignments,
        UniversalBuffer& queue)
{
    services::Status st;
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.processRowNbrs);
    auto& context = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernel_factory = context.getClKernelFactory();
    DAAL_CHECK_STATUS_VAR(buildProgram(kernel_factory));
    std::cout << "Process nbrs. chunkOffset = " << chunkOffset << std::endl;
    auto kernel = chunkOffset > 0 ? kernel_factory.getKernel("process_queue_neighbors", &st) : kernel_factory.getKernel("process_row_neighbors", &st);
    DAAL_CHECK_STATUS_VAR(st);

    uint32_t chunkSize = nRows / numberOfChunks + uint32_t(bool(nRows % numberOfChunks));
    /*{
        auto dist = rowDistances. template get<algorithmFPType>().toHost(ReadWriteMode::readOnly);
        for(int i = 0; i < 20; i++)
            std::cout << dist.get()[i] << " ";
        std::cout << std::endl;
    }*/

    KernelArguments args(11);
    args.set(0, rowDistances, AccessModeIds::read);
    args.set(1, offsets, AccessModeIds::read);
    args.set(2, assignments, AccessModeIds::readwrite);
    args.set(3, queue, AccessModeIds::readwrite);
    args.set(4, qEnd);
    args.set(5, rowId);
    args.set(6, clusterId);
    args.set(7, chunkOffset);
    args.set(8, chunkSize);
    args.set(9, eps);
    args.set(10, nRows);

    KernelRange local_range(1, _maxWgSize);
    KernelRange global_range(numberOfChunks * _minSgSize / _maxWgSize + 1, _maxWgSize);

    KernelNDRange range(2);
    range.global(global_range, &st); DAAL_CHECK_STATUS_VAR(st);
    range.local(local_range, &st); DAAL_CHECK_STATUS_VAR(st);

    context.run(range, kernel, args, &st);
    return st;
}



template <typename algorithmFPType>
services::Status DBSCANBatchKernelUCAPI<algorithmFPType>::countOffsets(
        const UniversalBuffer& counters,
        uint32_t numberOfChunks,
        UniversalBuffer& offsets) 
{
    services::Status st;
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.countOffsets);
    auto& context = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernel_factory = context.getClKernelFactory();
    DAAL_CHECK_STATUS_VAR(buildProgram(kernel_factory));
    auto kernel = kernel_factory.getKernel("count_offsets", &st);
    DAAL_CHECK_STATUS_VAR(st);

    KernelArguments args(3);
    args.set(0, counters, AccessModeIds::read);
    args.set(1, offsets, AccessModeIds::write);
    args.set(2, numberOfChunks);

    KernelRange local_range(1, _minSgSize);
    context.run(local_range, kernel, args, &st);
    return st;
}

template <typename algorithmFPType>
services::Status DBSCANBatchKernelUCAPI<algorithmFPType>::setBufferValue(
        UniversalBuffer& buffer,
        uint32_t index,
        int value) 
{
    std::cout << "Do set buffer value" << std::endl;
    services::Status st;
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.setBufferValue);
    auto& context = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernel_factory = context.getClKernelFactory();
    DAAL_CHECK_STATUS_VAR(buildProgram(kernel_factory));
    std::cout << "Taking kernel" << std::endl;
    auto kernel = kernel_factory.getKernel("set_buffer_value", &st);
    DAAL_CHECK_STATUS_VAR(st);
    std::cout << "Setting args" << std::endl;
    KernelArguments args(3);
    args.set(0, buffer, AccessModeIds::readwrite);
    args.set(1, index);
    args.set(2, value);

    KernelRange global_range(1);
    std::cout << "Run set buffer value: " << index << " " << value << std::endl;
    context.run(global_range, kernel, args, &st);
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
    DAAL_CHECK_STATUS_VAR(buildProgram(kernel_factory));
    auto kernel = kernel_factory.getKernel("set_buffer_value_by_queue_index", &st);
    DAAL_CHECK_STATUS_VAR(st);

    KernelArguments args(4);
    args.set(0, queue, AccessModeIds::read);
    args.set(1, buffer, AccessModeIds::readwrite);
    args.set(2, posInQueue);
    args.set(3, value);

    KernelRange global_range(1);
    context.run(global_range, kernel, args, &st);
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
    DAAL_CHECK_STATUS_VAR(buildProgram(kernel_factory));
    auto kernel = kernel_factory.getKernel("query_row", &st);
    DAAL_CHECK_STATUS_VAR(st);

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
    range.global(global_range, &st); DAAL_CHECK_STATUS_VAR(st);
    range.local(local_range, &st); DAAL_CHECK_STATUS_VAR(st);

    context.run(range, kernel, args, &st);
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
    DAAL_CHECK_STATUS_VAR(buildProgram(kernel_factory));
    auto kernel = kernel_factory.getKernel("query_queue", &st);
    DAAL_CHECK_STATUS_VAR(st);

    KernelArguments args(8);
    args.set(0, data, AccessModeIds::read);
    args.set(1, queue, AccessModeIds::read);
    args.set(2, rowDistances, AccessModeIds::write);
    args.set(3, queueBegin);
    args.set(4, queueBlockSize);
    args.set(5, minkowskiPower);
    args.set(6, dim);
    args.set(7, nRows);

    KernelRange local_range(1, _maxWgSize);
    KernelRange global_range(queueBlockSize, _maxWgSize);

    KernelNDRange range(2);
    range.global(global_range, &st); DAAL_CHECK_STATUS_VAR(st);
    range.local(local_range, &st); DAAL_CHECK_STATUS_VAR(st);

    context.run(range, kernel, args, &st);
    return st;
}

template <typename algorithmFPType>
Status DBSCANBatchKernelUCAPI<algorithmFPType>::countNbrs(
        const UniversalBuffer& assignments,
        const UniversalBuffer& rowDistances,
        uint32_t rowId, 
        int chunkOffset, 
        uint32_t nRows,
        uint32_t numberOfChunks,
        algorithmFPType eps,
        const UniversalBuffer& queue,
        UniversalBuffer& counters,
        UniversalBuffer& undefCounters)
{
    services::Status st;
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.countNbrs);
    auto& context = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernel_factory = context.getClKernelFactory();
    DAAL_CHECK_STATUS_VAR(buildProgram(kernel_factory));
    auto kernel = kernel_factory.getKernel("count_neighbors", &st);
    DAAL_CHECK_STATUS_VAR(st);

    uint32_t chunkSize = nRows / numberOfChunks + uint32_t(bool(nRows % numberOfChunks));
    std::cout << "Chunk size: " << chunkSize << std::endl;
    KernelArguments args(10);
    args.set(0, assignments, AccessModeIds::read);
    args.set(1, rowDistances, AccessModeIds::read);
    args.set(2, rowId);
    args.set(3, chunkOffset);
    args.set(4, chunkSize);
    args.set(5, nRows);
    args.set(6, eps);
    args.set(7, queue, AccessModeIds::read);
    args.set(8, counters, AccessModeIds::write);
    args.set(9, undefCounters, AccessModeIds::write);

    KernelRange local_range(1, _maxWgSize);
    KernelRange global_range(numberOfChunks * _minSgSize / _maxWgSize + 1, _maxWgSize);

    KernelNDRange range(2);
    range.global(global_range, &st); DAAL_CHECK_STATUS_VAR(st);
    range.local(local_range, &st); DAAL_CHECK_STATUS_VAR(st);

    context.run(range, kernel, args, &st);
    return st;
}

template <typename algorithmFPType>
uint32_t DBSCANBatchKernelUCAPI<algorithmFPType>::sumCounters(
        const UniversalBuffer& counters,
        uint32_t numberOfChunks)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.sumCounters);
    auto cntPtr = counters.template get<int>().toHost(ReadWriteMode::writeOnly).get();
    uint32_t ret = 0;
    for(uint32_t i = 0; i < numberOfChunks; i++)
        ret += cntPtr[i];
    return ret;
}

template <typename algorithmFPType>
Status DBSCANBatchKernelUCAPI<algorithmFPType>::buildProgram(ClKernelFactoryIface & kernel_factory)
{
    Status st;
    auto fptype_name = oneapi::internal::getKeyFPType<algorithmFPType>();
    auto build_options = fptype_name;
    build_options.add(" -D_NOISE_=-1 -D_UNDEFINED_=-2 ");

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
