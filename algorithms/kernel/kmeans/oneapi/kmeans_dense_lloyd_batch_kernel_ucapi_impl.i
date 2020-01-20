/* file: kmeans_dense_lloyd_batch_kernel_ucapi_impl.i */
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
//  Implementation of K-means Batch Kernel for GPU.
//--
*/

#ifndef __KMEANS_DENSE_LLOYD_BATCH_KERNEL_UCAPI_IMPL__
#define __KMEANS_DENSE_LLOYD_BATCH_KERNEL_UCAPI_IMPL__

#include "env_detect.h"
#include "cl_kernels/kmeans_cl_kernels.cl"
#include "execution_context.h"
#include "oneapi/service_defines_oneapi.h"
#include "oneapi/internal/types.h"
#include "oneapi/blas_gpu.h"

#include "service_ittnotify.h"

DAAL_ITTNOTIFY_DOMAIN(kmeans.dense.lloyd.batch.oneapi);

using namespace daal::services;
using namespace daal::oneapi::internal;
using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace internal
{
template <typename algorithmFPType>
Status KMeansDenseLloydBatchKernelUCAPI<algorithmFPType>::compute(const NumericTable * const * a, const NumericTable * const * r,
                                                                  const Parameter * par)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute);

    Status st;

    auto & context        = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernel_factory = context.getClKernelFactory();

    NumericTable * ntData         = const_cast<NumericTable *>(a[0]);
    NumericTable * ntInCentroids  = const_cast<NumericTable *>(a[1]);
    NumericTable * ntOutCentroids = const_cast<NumericTable *>(r[0]);
    NumericTable * ntAssignments  = const_cast<NumericTable *>(r[1]);
    NumericTable * ntObjFunction  = const_cast<NumericTable *>(r[2]);
    NumericTable * ntNIterations  = const_cast<NumericTable *>(r[3]);

    const size_t nIter     = par->maxIterations;
    const size_t nRows     = ntData->getNumberOfRows();
    const size_t nFeatures = ntData->getNumberOfColumns();
    const size_t nClusters = par->nClusters;

    auto fptype_name   = oneapi::internal::getKeyFPType<algorithmFPType>();
    auto build_options = fptype_name;

    build_options.add(this->getBuildOptions(nClusters));

    services::String cachekey("__daal_algorithms_kmeans_lloyd_dense_batch_");
    cachekey.add(fptype_name);
    cachekey.add(build_options.c_str());

    {
        DAAL_ITTNOTIFY_SCOPED_TASK(compute.buildProgram);
        kernel_factory.build(ExecutionTargetIds::device, cachekey.c_str(), kmeans_cl_kernels, build_options.c_str());
    }

    const size_t nPartialCentroids = 128;
    const size_t nValuesInBlock    = 1024 * 1024 * 1024 / sizeof(algorithmFPType);
    const size_t nMinRows          = 1;
    size_t gemmBlockSize           = nValuesInBlock;

    while (gemmBlockSize > nValuesInBlock / nClusters)
    {
        gemmBlockSize >>= 1;
    }

    if (gemmBlockSize < nMinRows)
    {
        return Status(ErrorKMeansNumberOfClustersIsTooLarge);
    }

    size_t datasetBlockSize = nValuesInBlock;
    while (datasetBlockSize > nValuesInBlock / nFeatures)
    {
        datasetBlockSize >>= 1;
    }

    if (datasetBlockSize < nMinRows)
    {
        return Status(ErrorIncorrectNumberOfFeatures);
    }

    size_t blockSize = datasetBlockSize > gemmBlockSize ? gemmBlockSize : datasetBlockSize;
    if (blockSize > nRows)
    {
        blockSize = nRows;
    }

    size_t nPartNum = this->getCandidatePartNum(nClusters);

    auto dataSq                    = context.allocate(TypeIds::id<algorithmFPType>(), blockSize, &st);
    auto centroidsSq               = context.allocate(TypeIds::id<algorithmFPType>(), nClusters, &st);
    auto distances                 = context.allocate(TypeIds::id<algorithmFPType>(), blockSize * nClusters, &st);
    auto mindistances              = context.allocate(TypeIds::id<algorithmFPType>(), blockSize, &st);
    auto candidates                = context.allocate(TypeIds::id<int>(), nClusters, &st);
    auto candidateDistances        = context.allocate(TypeIds::id<algorithmFPType>(), nClusters, &st);
    auto partialCandidates         = context.allocate(TypeIds::id<int>(), nClusters * nPartNum, &st);
    auto partialCandidateDistances = context.allocate(TypeIds::id<algorithmFPType>(), nClusters * nPartNum, &st);
    auto partialCentroids          = context.allocate(TypeIds::id<algorithmFPType>(), nPartialCentroids * nClusters * nFeatures, &st);
    auto partialCentroidsCounters  = context.allocate(TypeIds::id<int>(), nPartialCentroids * nClusters, &st);
    DAAL_CHECK_STATUS_VAR(st);

    auto compute_squares = kernel_factory.getKernel(this->getComputeSquaresKernelName(nFeatures), &st);
    DAAL_CHECK_STATUS_VAR(st);

    auto init_distances             = kernel_factory.getKernel("init_distances", &st);
    auto compute_assignments        = kernel_factory.getKernel("reduce_assignments", &st);
    auto partial_reduce_centroids   = kernel_factory.getKernel("partial_reduce_centroids", &st);
    auto merge_reduce_centroids     = kernel_factory.getKernel("merge_reduce_centroids", &st);
    auto update_objective_function  = kernel_factory.getKernel("update_objective_function", &st);
    auto compute_partial_candidates = kernel_factory.getKernel("partial_candidates", &st);
    auto merge_partial_candidates   = kernel_factory.getKernel("merge_candidates", &st);

    BlockDescriptor<algorithmFPType> inCentroidsRows;
    ntInCentroids->getBlockOfRows(0, nClusters, readOnly, inCentroidsRows);
    auto inCentroids = inCentroidsRows.getBuffer();

    BlockDescriptor<algorithmFPType> outCentroidsRows;
    ntOutCentroids->getBlockOfRows(0, nClusters, readWrite, outCentroidsRows);
    auto outCentroids = outCentroidsRows.getBuffer();

    BlockDescriptor<algorithmFPType> objFunctionRows;
    ntObjFunction->getBlockOfRows(0, nClusters, readWrite, objFunctionRows);
    auto objFunction = objFunctionRows.getBuffer();

    algorithmFPType prevObjFunction = (algorithmFPType)0.0;

    size_t iter    = 0;
    size_t nBlocks = nRows / blockSize + int(nRows % blockSize != 0);

    for (; iter < nIter; iter++)
    {
        for (size_t block = 0; block < nBlocks; block++)
        {
            size_t first = block * blockSize;
            size_t last  = first + blockSize;

            if (last > nRows)
            {
                last = nRows;
            }

            size_t curBlockSize = last - first;

            BlockDescriptor<algorithmFPType> dataRows;
            ntData->getBlockOfRows(first, curBlockSize, readOnly, dataRows);
            auto data = dataRows.getBuffer();

            BlockDescriptor<int> assignmentsRows;
            ntAssignments->getBlockOfRows(first, curBlockSize, writeOnly, assignmentsRows);
            auto assignments = assignmentsRows.getBuffer();

            this->computeSquares(context, compute_squares, inCentroids, centroidsSq, nClusters, nFeatures, &st);
            DAAL_CHECK_STATUS_VAR(st);
            this->initDistances(context, init_distances, centroidsSq, distances, curBlockSize, nClusters, &st);
            DAAL_CHECK_STATUS_VAR(st);
            this->computeDistances(context, data, inCentroids, distances, blockSize, nClusters, nFeatures, &st);
            DAAL_CHECK_STATUS_VAR(st);
            this->computeAssignments(context, compute_assignments, distances, assignments, mindistances, curBlockSize, nClusters, &st);
            DAAL_CHECK_STATUS_VAR(st);
            this->computeSquares(context, compute_squares, data, dataSq, curBlockSize, nFeatures, &st);
            DAAL_CHECK_STATUS_VAR(st);
            this->computePartialCandidates(context, compute_partial_candidates, assignments, mindistances, dataSq, candidates, candidateDistances,
                                     partialCandidates, partialCandidateDistances, curBlockSize, nClusters, int(block == 0), &st);
            DAAL_CHECK_STATUS_VAR(st);
            this->mergePartialCandidates(context, merge_partial_candidates, candidates, candidateDistances, partialCandidates, partialCandidateDistances,
                                   nClusters, &st);
            DAAL_CHECK_STATUS_VAR(st);
            this->partialReduceCentroids(context, partial_reduce_centroids, data, distances, assignments, partialCentroids, partialCentroidsCounters,
                                   curBlockSize, nClusters, nFeatures, nPartialCentroids, int(block == 0), &st);
            DAAL_CHECK_STATUS_VAR(st);
            this->updateObjectiveFunction(context, update_objective_function, dataSq, distances, assignments, objFunction, curBlockSize, nClusters,
                                    int(block == 0), &st);
            DAAL_CHECK_STATUS_VAR(st);

            ntData->releaseBlockOfRows(dataRows);
            ntAssignments->releaseBlockOfRows(assignmentsRows);
        }

        mergeReduceCentroids(context, merge_reduce_centroids, partialCentroids, partialCentroidsCounters, outCentroids, nClusters, nFeatures,
                             nPartialCentroids, &st);
        DAAL_CHECK_STATUS_VAR(st);
        auto counters                     = partialCentroidsCounters.template get<int>().toHost(ReadWriteMode::readOnly);
        auto candidatesIds                = candidates.get<int>().toHost(ReadWriteMode::readOnly);
        auto candidatesDists              = candidateDistances.template get<algorithmFPType>().toHost(ReadWriteMode::readOnly);
        auto clusterFeatures              = outCentroids.toHost(ReadWriteMode::readWrite);
        algorithmFPType objFuncCorrection = 0.0;
        int cPos                          = 0;
        for (int iCl = 0; iCl < nClusters; iCl++)
            if (counters.get()[iCl] == 0)
            {
                if (cPos >= nClusters) continue;
                int id = candidatesIds.get()[cPos];
                if (id < 0 || id >= nRows)
                {
                    continue;
                }
                objFuncCorrection += candidatesDists.get()[cPos];
                BlockDescriptor<algorithmFPType> singleRow;
                ntData->getBlockOfRows(0, blockSize, readOnly, singleRow);
                auto row_data = singleRow.getBlockPtr();
                for (int iFeature = 0; iFeature < nFeatures; iFeature++)
                    clusterFeatures.get()[iCl * nFeatures + iFeature] = row_data[id * nFeatures + iFeature];
                cPos++;
                ntData->releaseBlockOfRows(singleRow);
            }
        algorithmFPType curObjFunction = (algorithmFPType)0.0;
        {
            auto hostPtr   = objFunction.toHost(data_management::readOnly);
            curObjFunction = *hostPtr;
            curObjFunction -= objFuncCorrection;
        }

        if (par->accuracyThreshold > (algorithmFPType)0.0)
        {
            algorithmFPType objFuncDiff =
                curObjFunction - prevObjFunction > 0 ? curObjFunction - prevObjFunction : -(curObjFunction - prevObjFunction);
            if (objFuncDiff < par->accuracyThreshold)
            {
                iter++;
                break;
            }
        }
        prevObjFunction = curObjFunction;

        inCentroids = outCentroids;
    }
    for (size_t block = 0; block < nBlocks; block++)
    {
        size_t first = block * blockSize;
        size_t last  = first + blockSize;

        if (last > nRows)
        {
            last = nRows;
        }

        size_t curBlockSize = last - first;

        BlockDescriptor<algorithmFPType> dataRows;
        ntData->getBlockOfRows(first, curBlockSize, readOnly, dataRows);
        auto data = dataRows.getBuffer();

        BlockDescriptor<int> assignmentsRows;
        ntAssignments->getBlockOfRows(first, curBlockSize, writeOnly, assignmentsRows);
        auto assignments = assignmentsRows.getBuffer();

        this->computeSquares(context, compute_squares, inCentroids, centroidsSq, nClusters, nFeatures, &st);
        DAAL_CHECK_STATUS_VAR(st);
        this->initDistances(context, init_distances, centroidsSq, distances, curBlockSize, nClusters, &st);
        DAAL_CHECK_STATUS_VAR(st);
        this->computeDistances(context, data, inCentroids, distances, blockSize, nClusters, nFeatures, &st);
        DAAL_CHECK_STATUS_VAR(st);
        this->computeAssignments(context, compute_assignments, distances, assignments, mindistances, curBlockSize, nClusters, &st);
        DAAL_CHECK_STATUS_VAR(st);
        this->computeSquares(context, compute_squares, data, dataSq, curBlockSize, nFeatures, &st);
        DAAL_CHECK_STATUS_VAR(st);
        this->updateObjectiveFunction(context, update_objective_function, dataSq, distances, assignments, objFunction, curBlockSize, nClusters,
                                int(block == 0), &st);
        DAAL_CHECK_STATUS_VAR(st);
        DAAL_CHECK_STATUS_VAR(st);
    }

    ntInCentroids->releaseBlockOfRows(inCentroidsRows);
    ntOutCentroids->releaseBlockOfRows(outCentroidsRows);
    ntObjFunction->releaseBlockOfRows(objFunctionRows);

    {
        BlockDescriptor<int> nIterationsRows;
        ntNIterations->getBlockOfRows(0, 1, writeOnly, nIterationsRows);
        auto nIterationsHostPtr = nIterationsRows.getBlockSharedPtr();
        int * nIterations       = nIterationsHostPtr.get();
        nIterations[0]          = iter;
        ntNIterations->releaseBlockOfRows(nIterationsRows);
    }

    return st;
}


} // namespace internal
} // namespace kmeans
} // namespace algorithms
} // namespace daal

#endif
