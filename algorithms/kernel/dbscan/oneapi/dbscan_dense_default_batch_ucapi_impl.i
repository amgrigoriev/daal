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

#define __DBSCAN_PREFETCHED_NEIGHBORHOODS_COUNT 64

template <typename algorithmFPType>
Status DBSCANBatchKernelUCAPI<algorithmFPType>::processNeighborhood(size_t clusterId, int * const assignments,
                                                                            const NeighborhoodUCAPI<algorithmFPType> & neigh, QueueUCAPI<size_t> & qu)
{
//    std::cout << "Size: " << neigh.size() << std::endl;
    for (size_t j = 0; j < neigh.size(); j++)
    {
        const size_t nextObs = neigh.get(j);
//        if(nextObs == 13)
//            std::cout << "Why???????????????" << std::endl;
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
    std::cout << "Started: size = " << sizeof(algorithmFPType) << std::endl;
    auto& context = Environment::getInstance()->getDefaultExecutionContext();
    auto& kernel_factory = context.getClKernelFactory();    

    const algorithmFPType epsilon         = par->epsilon;
    const algorithmFPType minObservations = par->minObservations;
    const algorithmFPType minkowskiPower  = (algorithmFPType)2.0;

    std::cout << "Params taken" << std::endl;

    NumericTable *ntData = const_cast<NumericTable *>( x );
    const size_t nRows = ntData->getNumberOfRows();
    const size_t dim = ntData->getNumberOfColumns();

    std::cout << "Tables taken" << std::endl;

    /*
    BlockDescriptor<algorithmFPType> dataRows;
    DAAL_CHECK_STATUS_VAR(ntData->getBlockOfRows(0, nRows, readOnly, dataRows));
    auto data = dataRows.getBuffer();*/

    std::cout << "Data taken" << std::endl;

    BlockDescriptor<int> assignRows;
    DAAL_CHECK_STATUS_VAR(ntAssignments->getBlockOfRows(0, nRows, readOnly, assignRows));
    auto assignments = assignRows.getBuffer().toHost(ReadWriteMode::readWrite).get();

    service_memset<int, sse2>(assignments, undefined, nRows);
    std::cout << "Assignments initialized" << std::endl;

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nRows, sizeof(int));

    TArray<int, sse2> isCoreArray(nRows);
    DAAL_CHECK_MALLOC(isCoreArray.get());
    int * const isCore = isCoreArray.get();

    auto distances = context.allocate(TypeIds::id<algorithmFPType>(), _queryBlockSize * _dataBlockSize, &s);
    DAAL_CHECK_STATUS_VAR(s);
    auto positions = context.allocate(TypeIds::id<int>(), _queryBlockSize * _dataBlockSize, &s);
    DAAL_CHECK_STATUS_VAR(s);
    auto counters = context.allocate(TypeIds::id<int>(), _queryBlockSize, &s);
    DAAL_CHECK_STATUS_VAR(s);

    service_memset<int, sse2>(isCore, 0, nRows);

    std::cout << "Core initialized" << std::endl;

    const size_t prefetchBlockSize = __DBSCAN_PREFETCHED_NEIGHBORHOODS_COUNT;
    TArray<NeighborhoodUCAPI<algorithmFPType>, sse2> prefetchedNeighs(prefetchBlockSize);
    DAAL_CHECK_MALLOC(prefetchedNeighs.get());

    size_t nClusters = 0;
    size_t first = 0;
    QueueUCAPI<size_t> qu;
    const size_t inititalBlockSize = 1;
    NeighborhoodUCAPI<algorithmFPType> curNeighs[inititalBlockSize];
    VectorUCAPI<size_t> indices(_queryBlockSize);
    std::cout << indices.size() << std::endl;

    std::cout << "Containers reserved" << std::endl;


    while (first < nRows - 1)
    {
        std::cout << "Main cycle: " << first << " " << nRows << std::endl;
        indices.reset();
        std::cout << "Indices cleaned: "<< std::endl;
        getFreeIndices(first, inititalBlockSize, nRows, assignments, indices);
        if(indices.size() == 0)
        {
            break;
        }
        std::cout << "Indices obtained: " << indices.size() << std::endl;
        size_t pos = indices[indices.size() - 1] + 1;
        std::cout << "Pos:" << pos << "; nRows: " << nRows << std::endl;
        DAAL_CHECK_STATUS_VAR(query(ntData, distances, positions, counters, indices.ptr(), indices.size(), 
                           nRows, dim, epsilon, assignments, &curNeighs[0], true));
//        std::cout << "Let's process querried nbrs: " << std::endl;
        for(size_t iNbr = 0; iNbr < 1/*indices.size()*/; iNbr++)
        {
//            std::cout << "External cycle: " << iNbr << std::endl;
            if(!(assignments[indices[iNbr]] == undefined)) {
//                std::cout << "Visited" << std::endl;
                continue;
            }
            if (curNeighs[iNbr].weight() < minObservations)
            {
                assignments[indices[iNbr]] = noise;
//                std::cout << iNbr << " noise " << std::endl;
                continue;
            }
            nClusters++;
            isCore[indices[iNbr]]      = 1;
            assignments[indices[iNbr]] = (nClusters - 1);
//            std::cout << iNbr << " " << " cluster: " << (nClusters - 1) << std::endl;

            qu.reset();

            DAAL_CHECK_STATUS_VAR(processNeighborhood(assignments[indices[iNbr]], assignments, curNeighs[iNbr], qu));
//            for(int i = 0; i < 20; i++)
//                std::cout << "assignments: " << assignments[i] << std::endl;
//            std::cout << "Next neighbors: " << qu.tail() << " " << qu.head() << std::endl;
            size_t firstPrefetchedPos = 0;
            size_t lastPrefetchedPos  = 0;
            int count = 0;
            while (!qu.empty())
            {
//                std::cout << "!!!!!!Queue block: " << count++ << std::endl;
                VectorUCAPI<size_t> curNbrIndices = qu.popVector(prefetchBlockSize);
                for(int j = 0; j < curNbrIndices.size(); j++)
                    std::cout << curNbrIndices[j] << " ";
                std::cout << std::endl;
                DAAL_CHECK_STATUS_VAR(query(ntData, distances, positions, counters, curNbrIndices.ptr(), 
                                                    curNbrIndices.size(), nRows, dim, epsilon,
                                                    assignments, &(prefetchedNeighs[0]),  
                                                    true));
                for(size_t ind = 0; ind < curNbrIndices.size(); ind++) {
                    NeighborhoodUCAPI<algorithmFPType> & curNeigh = prefetchedNeighs[ind];

                    assignments[curNbrIndices[ind]] = nClusters - 1;
                    if (curNeigh.weight() < minObservations) continue;

                    isCore[curNbrIndices[ind]] = 1;

                    DAAL_CHECK_STATUS_VAR(processNeighborhood(nClusters - 1, assignments, curNeigh, qu));
                }
//                for(int i = 0; i < 20; i++)
//                    std::cout << "#assignments: " << assignments[i] << std::endl;
            }
        }
        first = pos;
    }

    BlockDescriptor<int> nClustersRows;
    DAAL_CHECK_STATUS_VAR(ntNClusters->getBlockOfRows(0, 1, writeOnly, nClustersRows));
    nClustersRows.getBuffer().toHost(ReadWriteMode::writeOnly).get()[0] = nClusters;

    if (par->resultsToCompute & (computeCoreIndices | computeCoreObservations))
    {
        DAAL_CHECK_STATUS_VAR(processResultsToCompute(par->resultsToCompute, isCore, ntData, ntCoreIndices, ntCoreObservations));
    }
    std::cout << "COmpute done" << std::endl;
    return s;

}

template <typename algorithmFPType>
services::Status DBSCANBatchKernelUCAPI<algorithmFPType>::query(
        NumericTable* ntData, 
        UniversalBuffer& distances,
        UniversalBuffer& positions,
        UniversalBuffer& counters,
        size_t * indices, size_t n, size_t nRows, size_t dim, 
        algorithmFPType eps,
        int * const assignments,
        NeighborhoodUCAPI<algorithmFPType> * neighs, 
        bool doReset)
{
    services::Status s;
    static int count = 0;
    count++;

    auto& context = Environment::getInstance()->getDefaultExecutionContext();

    if (doReset)
    {
        std::cout << "  Reseting indices..." << std::endl;
        for (size_t i = 0; i < n; i++)
        {
            neighs[i].reset();
        }
    }

    algorithmFPType epsSq = eps * eps;

    size_t nQuerryBlocks = n / _queryBlockSize + size_t (n % _queryBlockSize != 0);
    size_t nDataBlocks = nRows / _dataBlockSize + size_t (nRows % _dataBlockSize != 0);

//    std::cout << "  Blocks: " << nQuerryBlocks << " " << nDataBlocks << std::endl;

    DAAL_CHECK_STATUS_VAR(s);
    auto queryBlock = context.allocate(TypeIds::id<algorithmFPType>(), _queryBlockSize * dim, &s);
    auto dataBlock = context.allocate(TypeIds::id<algorithmFPType>(), _dataBlockSize * dim, &s);

//    std::cout << "  Blocks allocated" << std::endl;

    for(size_t qblock = 0; qblock < nQuerryBlocks; qblock++)
    {
        const size_t qfirst = qblock * _queryBlockSize;
        size_t qlast = qfirst + _queryBlockSize;
        qlast = qlast > n ? n : qlast;
        const size_t qsize = qlast -qfirst;
//        std::cout << "  qblock: " << qblock << "; qsize: " << qsize << std::endl;
        std::cout << "  Filling rows" << std::endl;
        fillRows(context, ntData, queryBlock, indices, qfirst, qsize, dim, &s);
        if(0/*count == 4*/)
        {
            auto data = queryBlock.template get<algorithmFPType>().toHost(ReadWriteMode::readOnly).get();
            std::cout << "QueryBlock" << std::endl;
            for(int i = 0; i < 10; i++) {
                double sq = 0.0;
                std::cout << indices[qfirst + i] << " ";
                for(int j = 0; j < dim; j++) {
                    std::cout << data[i * dim + j] << " ";
                    sq += data[i * dim + j] * data[i * dim + j];
                }
                std::cout << sq;
                std::cout << std::endl;
            }
        }
//        std::cout << "  Calling sum reducer" << std::endl;
        auto qSum = math::SumReducer::sum(math::Layout::RowMajor,
                                            queryBlock, qsize, dim, &s); DAAL_CHECK_STATUS_VAR(s);
/*        if(count == 4)
        {
            auto data = qSum.sumOfSquares.template get<algorithmFPType>().toHost(ReadWriteMode::readOnly).get();
            std::cout << "Querry squares" << std::endl;
            for(int i = 0; i < qsize; i++) {
                    std::cout << data[i] << " ";
                std::cout << std::endl;
            }
        }*/
        for(size_t dblock = 0; dblock < nDataBlocks; dblock++ )
        {
            const size_t dfirst = dblock * _dataBlockSize;
            size_t dlast = dfirst + _dataBlockSize;
            dlast = dlast > nRows ? nRows : dlast;
            const size_t dsize = dlast -dfirst;
//            std::cout << "  dblock: " << dblock << "; dsize: " << dsize << std::endl;
            BlockDescriptor<algorithmFPType> dataRows;
//            std::cout << "  Getting block of rows" << std::endl;
            DAAL_CHECK_STATUS_VAR(ntData->getBlockOfRows(dfirst, dsize, readOnly, dataRows));
            if(0/*count == 4*/)
            {
                auto data = dataRows.getBuffer().toHost(ReadWriteMode::readOnly).get();
                std::cout << "Data Block" << std::endl;
                for(int i = 0; i < 15; i++) {
                    std::cout << (dfirst + i) << " ";
                    double sq = 0.0;
                    for(int j = 0; j < dim; j++) {
                        std::cout << data[i * dim + j] << " ";
                        sq += data[i * dim + j] * data[i * dim + j];
                    }
                    std::cout << sq;
                    std::cout << std::endl;
                }
            }
//            context.copy(dataBlock, 0, dataRows.getBuffer(), dfirst * dim, dsize * dim, &s);
//            std::cout << "  Calling sum reducer" << std::endl;
            auto dSum = math::SumReducer::sum(math::Layout::RowMajor, dataRows.getBuffer(), dsize, dim, &s); 
            DAAL_CHECK_STATUS_VAR(s); 
/*            {
                auto data = dSum.sumOfSquares.template get<algorithmFPType>().toHost(ReadWriteMode::readOnly).get();
                std::cout << "Data squares" << std::endl;
                for(int i = 0; i < dsize; i++) {
                        std::cout << data[i] << " ";
                    std::cout << std::endl;
                }
            }*/
//            std::cout << "  Calling init_distances" << std::endl;                                                
            initDistances(context, qSum.sumOfSquares, dSum.sumOfSquares, distances, qsize, dsize, &s); DAAL_CHECK_STATUS_VAR(s);
//            std::cout << "  Calling GEMM" << std::endl;
            if(0/*count == 4*/)
            {
                auto data = distances.template get<algorithmFPType>().toHost(ReadWriteMode::readOnly).get();
                std::cout << "initial distances" << std::endl;
                for(int i = 0; i < 10; i++) {
                    for(int j = 0; j < 15; j++)
                        std::cout << data[i * dsize + j] << " ";
                    std::cout << std::endl;
                }
            } 
            computeDistances(context, dataRows.getBuffer(), 
                             queryBlock.template get<algorithmFPType>(),
                             distances, dsize, qsize, dim, &s); DAAL_CHECK_STATUS_VAR(s);
            DAAL_CHECK_STATUS_VAR(ntData->releaseBlockOfRows(dataRows));
//            std::cout << "  Calling partition" << std::endl;
            if(0/*count == 4*/)
            {
                auto data = distances.template get<algorithmFPType>().toHost(ReadWriteMode::readOnly).get();
                std::cout << "final distances" << std::endl;
                for(int i = 0; i < 10; i++) {
                    for(int j = 0; j < 15; j++)
                        std::cout << data[i * dsize + j] << " ";
                    std::cout << std::endl;
                }
            }
            // Exclude counting itself!!!!
            partition(context, distances, positions, counters, dsize, qsize, epsSq, &s); DAAL_CHECK_STATUS_VAR(s);
            {
//                std::cout << "  Filling next neighborhood" << std::endl;
                auto weights = counters.template get<int>().toHost(ReadWriteMode::readOnly).get();
                auto ids = positions.template get<int>().toHost(ReadWriteMode::readOnly).get();
                for (size_t i = 0; i < qsize; i++)
                {
                    size_t localSize = weights[i];
//                    std::cout << i << " Nbrs: " << localSize << std::endl;
                    for (size_t j = 0; j < localSize; j++)
                    {
                        /*std::cout << "  ind: " << j << "; pos: " << ids[j + i * N] << std::endl;
                        if(dfirst + ids[j] == 13) {
                            std::cout << "Unexpected glue!!!!!!!!! " << count << " " << dblock << " " << qblock << std::endl;
                            std::cout << "Nbr: " << indices[qfirst + i] << " " << qfirst << " " << i << std::endl;
                            std::cout << "epsSq: " << epsSq << std::endl;
                            exit(0);
                        }*/
                        neighs[qfirst + i].add(dfirst + ids[j + i * dsize], 0);
//                        std::cout << dfirst + ids[j] << " ";
                    }
//                    std::cout << std::endl;
                    neighs[qfirst + i].addWeight(localSize);
                }    
            }
        }
    }
    /*
    std::cout << "Current weights" << std::endl;
    for(int i = 0; i < n; i++)
        std::cout << i << " index: " << indices[i] << " weight: " << neighs[i].weight() << std::endl;*/
    return s;
}

template <typename algorithmFPType>
void DBSCANBatchKernelUCAPI<algorithmFPType>::fillRows(
        ExecutionContextIface& context,
        NumericTable* ntData,
        UniversalBuffer& nbrRows,
        size_t * indices, 
        size_t pos, 
        size_t n,
        size_t dim,
        Status * status)
{
    DAAL_CHECK_STATUS_PTR(status);
    const size_t nRows = ntData->getNumberOfRows();
    auto buffer = nbrRows.template get<algorithmFPType>();
    auto bufptr = buffer.toHost(ReadWriteMode::writeOnly);
    BlockDescriptor<algorithmFPType> row;
    Status s = ntData->getBlockOfRows(0, nRows, readOnly, row);
    if(!s.ok()) 
    {
        if(status) 
        {
            status->add(s);
        }
        std::cout << "  Bad status" << std::endl;
        return;
    }
    std::cout << "Getting buffer" << std::endl;
    auto rowBuffer = row.getBuffer();
    std::cout << "Row to host" << std::endl;
    auto rowptr = rowBuffer.toHost(ReadWriteMode::readOnly, status);
    DAAL_CHECK_STATUS_PTR(status);
    for(size_t i = 0; i < n; i++) 
    {
        std::cout << "fill row: " << indices[pos + i] << " from " << n << std::endl;
        /*
        BlockDescriptor<algorithmFPType> row;
        Status s = ntData->getBlockOfRows(indices[pos + i], 1, readOnly, row);
        if(!s.ok()) 
        {
            if(status) 
            {
                status->add(s);
            }
            std::cout << "  Bad status" << std::endl;
            return;
        }
        DAAL_CHECK_STATUS_PTR(status);
        std::cout << "Getting buffer" << std::endl;
        auto rowBuffer = row.getBuffer();
        std::cout << "Row to host" << std::endl;
        auto rowptr = rowBuffer.toHost(ReadWriteMode::readOnly, status);
        DAAL_CHECK_STATUS_PTR(status);*/
        std::cout << "Copy" << std::endl;
        for(size_t j = 0; j < dim; j++) 
        {
            bufptr.get()[i * dim + j] = rowptr.get()[indices[pos + i] * dim + j];
//            std::cout << bufptr.get()[i * dim + j] << " ";
        }
//        ntData->releaseBlockOfRows(row);
        std::cout << "Copying done" << std::endl;
//        std::cout << std::endl;
/*        std::cout << "  Copying context " << (i * dim) << " " << indices[pos + i] * dim << std::endl;
        context.copy(buffer, i * dim, row.getBuffer(), indices[pos + i] * dim, dim, &s);*/
    }
    ntData->releaseBlockOfRows(row);
    std::cout << "fillRows done" << std::endl;
}

template <typename algorithmFPType>
size_t DBSCANBatchKernelUCAPI<algorithmFPType>::getFreeIndices(
        size_t first, 
        size_t size, 
        size_t nRows,
        int * const assignments,
        VectorUCAPI<size_t>& data)
{
//    std::cout << "  initial size: " << data.size() << "; nRows: " << nRows << std::endl;
    size_t pos = first;
    while(data.size() < size && pos < nRows)
    {
        if(assignments[pos] == undefined) {
//            std::cout << " Pushing back: " << pos << std::endl;
            data.push_back(pos);
        }
        pos++;
        if(!(nRows > pos))
            break;
    }
//    std::cout << "  final pos: " << pos << std::endl;
    return pos;
}

template <typename algorithmFPType>
void DBSCANBatchKernelUCAPI<algorithmFPType>::initDistances(
        ExecutionContextIface& context,
        UniversalBuffer& querySq,
        UniversalBuffer& dataSq,
        UniversalBuffer& distances,
        uint32_t probesBlockSize,
        uint32_t dataBlockSize,
        Status* st)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.initDistances);

    auto & kernel_factory = context.getClKernelFactory();
    buildProgram(kernel_factory, st);
    DAAL_CHECK_STATUS_PTR(st);
    auto kernel = kernel_factory.getKernel("init_distances", st);
    DAAL_CHECK_STATUS_PTR(st);

    KernelArguments args(4);
    args.set(0, querySq, AccessModeIds::read);
    args.set(1, dataSq, AccessModeIds::read);
    args.set(2, distances, AccessModeIds::write);
    args.set(3, dataBlockSize);

    KernelRange global_range(dataBlockSize, probesBlockSize);
    context.run(global_range, kernel, args, st);
}

template <typename algorithmFPType>
void DBSCANBatchKernelUCAPI<algorithmFPType>::partition(ExecutionContextIface& context,
        UniversalBuffer& distances,
        UniversalBuffer& positions,
        UniversalBuffer& counters,
        uint32_t dsize,
        uint32_t qsize,
        algorithmFPType pivot,
        Status* st)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.initDistances);

    auto & kernel_factory = context.getClKernelFactory();
    buildProgram(kernel_factory, st);
    DAAL_CHECK_STATUS_PTR(st);
    auto kernel = kernel_factory.getKernel("partition", st);
    DAAL_CHECK_STATUS_PTR(st);

//    std::cout << "Pivot: " << pivot << std::endl;

    KernelArguments args(6);
    args.set(0, distances, AccessModeIds::read);
    args.set(1, positions, AccessModeIds::write);
    args.set(2, counters, AccessModeIds::write);
    args.set(3, dsize);
    args.set(4, qsize);
    args.set(5, pivot);

    KernelRange local_range(1, _maxLocalSize);
    KernelRange global_range(qsize, _maxLocalSize);

    KernelNDRange range(2);
    range.global(global_range, st); DAAL_CHECK_STATUS_PTR(st);
    range.local(local_range, st); DAAL_CHECK_STATUS_PTR(st);

    context.run(range, kernel, args, st);
}

template <typename algorithmFPType>
void DBSCANBatchKernelUCAPI<algorithmFPType>::computeDistances(
    ExecutionContextIface & context, const Buffer<algorithmFPType> & data,
    const Buffer<algorithmFPType> & probes, 
    UniversalBuffer & distances,
    uint32_t dataBlockSize, 
    uint32_t probeBlockSize, 
    uint32_t nFeatures,
    Status * st)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.GEMM);
    auto gemmStatus = BlasGpu<algorithmFPType>::xgemm(math::Layout::RowMajor, math::Transpose::NoTrans, math::Transpose::Trans, probeBlockSize,
                                                      dataBlockSize, nFeatures, algorithmFPType(-2.0), probes, nFeatures, 0, data, nFeatures, 0,
                                                      algorithmFPType(1.0), distances.get<algorithmFPType>(), dataBlockSize, 0);
    if (st != nullptr)
    {
        *st = gemmStatus;
    }
}

template <typename algorithmFPType>
void DBSCANBatchKernelUCAPI<algorithmFPType>::buildProgram(ClKernelFactoryIface & kernel_factory, Status * st)
{
    auto fptype_name = oneapi::internal::getKeyFPType<algorithmFPType>();
    auto build_options = fptype_name;

    services::String cachekey("__daal_algorithms_dbscan_block_");
    cachekey.add(fptype_name);
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(compute.buildProgram);
        kernel_factory.build(ExecutionTargetIds::device, cachekey.c_str(), 
        dbscan_cl_kernels, build_options.c_str(), st);
        DAAL_CHECK_STATUS_PTR(st);
    }

//    std::cout << "Kernels built" << std::endl;

}


} // namespace internal
} // namespace dbscan
} // namespace algorithms
} // namespace daal
