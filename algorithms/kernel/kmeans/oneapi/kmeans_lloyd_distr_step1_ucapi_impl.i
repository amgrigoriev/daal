/* file: kmeans_lloyd_distr_step1_impl.i */
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
//  Implementation of Lloyd method for K-means algorithm.
//--
*/

#include "env_detect.h"
#include "cl_kernels/kmeans_cl_kernels.cl"
#include "execution_context.h"
#include "oneapi/service_defines_oneapi.h"
#include "oneapi/internal/types.h"
#include "oneapi/blas_gpu.h"

#include "service_ittnotify.h"
#include "service_numeric_table.h"

#include "oneapi/kmeans_lloyd_distr_step1_kernel_ucapi.h"

DAAL_ITTNOTIFY_DOMAIN(kmeans.dense.lloyd.distr.step1.oneapi);

using namespace daal::internal;
using namespace daal::services::internal;
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
#define __DAAL_FABS(a) (((a) > (algorithmFPType)0.0) ? (a) : (-(a)))

template <typename algorithmFPType>
Status KMeansDistributedStep1KernelUCAPI<algorithmFPType>::compute(size_t na, const NumericTable * const * a, size_t nr,
                                                                const NumericTable * const * r, const Parameter * par)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute);

    Status st;

    auto & context        = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernel_factory = context.getClKernelFactory();

    NumericTable * ntData         = const_cast<NumericTable *>(a[0]);
    NumericTable * ntInCentroids  = const_cast<NumericTable *>(a[1]);
    NumericTable * ntClusterS0    = const_cast<NumericTable *>(r[0]);
    NumericTable * ntClusterS1    = const_cast<NumericTable *>(r[1]);
    NumericTable * ntObjFunction  = const_cast<NumericTable *>(r[2]);
    NumericTable * ntCValues      = const_cast<NumericTable *>(r[3]);
    NumericTable * ntCCentroids   = const_cast<NumericTable *>(r[4]);
    NumericTable * ntAssignments  = const_cast<NumericTable *>(r[5]);

    const size_t nIter     = par->maxIterations;
    const size_t nRows     = ntData->getNumberOfRows();
    const size_t nFeatures = ntData->getNumberOfColumns();
    const size_t nClusters = par->nClusters;

    int result             = 0;

    BlockDescriptor<algorithmFPType> inCentroidsRows;
    ntInCentroids->getBlockOfRows(0, nClusters, readOnly, inCentroidsRows);
    auto inCentroids = inCentroidsRows.getBuffer();

    BlockDescriptor<int> ntClusterS0Rows;
    ntClusterS0->getBlockOfRows(0, nClusters, writeOnly, ntClusterS0Rows);
    auto outCCounters = ntClusterS0Rows.getBuffer();

    BlockDescriptor<algorithmFPType> ntClusterS1Rows;
    ntClusterS1->getBlockOfRows(0, nClusters, writeOnly, ntClusterS1Rows);
    auto outCentroids = ntClusterS1Rows.getBuffer();

    BlockDescriptor<algorithmFPType> ntObjFunctionRows;
    ntObjFunction->getBlockOfRows(0, nClusters, writeOnly, ntObjFunctionRows);
    auto outObjFunction = ntObjFunctionRows.getBuffer();

    BlockDescriptor<algorithmFPType> ntCValuesRows;
    ntCValues->getBlockOfRows(0, nClusters, writeOnly, ntCValuesRows);
    auto outCValues = UniversalBuffer(ntCValuesRows.getBuffer());

    BlockDescriptor<algorithmFPType> ntCCentroidsRows;
    ntCCentroids->getBlockOfRows(0, nClusters, writeOnly, ntCCentroidsRows);
    auto outCCentroids = UniversalBuffer(ntCCentroidsRows.getBuffer());

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

    auto assignments               = context.allocate(TypeIds::id<int>(), blockSize, &st);
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

    size_t nBlocks = nRows / blockSize + int(nRows % blockSize != 0);
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
        
        this->computeSquares(context, compute_squares, inCentroids, centroidsSq, nClusters, nFeatures, &st);
        DAAL_CHECK_STATUS_VAR(st);
        this->initDistances(context, init_distances, centroidsSq, distances, curBlockSize, nClusters, &st);
        DAAL_CHECK_STATUS_VAR(st);
        this->computeDistances(context, data, inCentroids, distances, blockSize, nClusters, nFeatures, &st);
        DAAL_CHECK_STATUS_VAR(st);
        this->computeAssignments(context, compute_assignments, distances, assignments.template get<int>(), mindistances, curBlockSize, nClusters, &st);
        DAAL_CHECK_STATUS_VAR(st);
        this->computeSquares(context, compute_squares, data, dataSq, curBlockSize, nFeatures, &st);
        DAAL_CHECK_STATUS_VAR(st);
        this->computePartialCandidates(context, compute_partial_candidates, assignments.template get<int>(), mindistances, dataSq, candidates, candidateDistances,
                                    partialCandidates, partialCandidateDistances, curBlockSize, nClusters, int(block == 0), &st);
        DAAL_CHECK_STATUS_VAR(st);
        this->mergePartialCandidates(context, merge_partial_candidates, outCCentroids, outCValues, partialCandidates, partialCandidateDistances,
                                nClusters, &st);
        DAAL_CHECK_STATUS_VAR(st);
        this->partialReduceCentroids(context, partial_reduce_centroids, data, distances, assignments.template get<int>(), partialCentroids, partialCentroidsCounters,
                                curBlockSize, nClusters, nFeatures, nPartialCentroids, int(block == 0), &st);
        DAAL_CHECK_STATUS_VAR(st);

        this->updateObjectiveFunction(context, update_objective_function, dataSq, distances, assignments.template get<int>(), outObjFunction, curBlockSize, nClusters,
                                int(block == 0), &st);
        DAAL_CHECK_STATUS_VAR(st);

        ntData->releaseBlockOfRows(dataRows);
        if (par->assignFlag) {
            BlockDescriptor<int> assignmentsRows;
            st = ntAssignments->getBlockOfRows(0, nRows, writeOnly, assignmentsRows);
            DAAL_CHECK_STATUS_VAR(st);
            auto fassignments = assignmentsRows.getBuffer();
            context.copy(fassignments, first, assignments, 0, blockSize, &st);
            ntAssignments->releaseBlockOfRows(assignmentsRows);
        }
    }
    this->mergeReduceCentroids(context, merge_reduce_centroids, partialCentroids, partialCentroidsCounters, outCentroids, nClusters, nFeatures,
                            nPartialCentroids, &st);
    {
        auto finalCounters = partialCentroidsCounters.template get<int>().toHost(ReadWriteMode::readOnly);
        auto outCCountersHost = outCCounters.toHost(ReadWriteMode::readOnly);
        for(int i = 0; i < nClusters; i++) {
            outCCountersHost.get()[i] = finalCounters.get()[i];
        }
    }
    ntInCentroids->releaseBlockOfRows(inCentroidsRows);
    ntClusterS0->releaseBlockOfRows(ntClusterS0Rows);
    ntClusterS1->releaseBlockOfRows(ntClusterS1Rows);
    ntObjFunction->releaseBlockOfRows(ntObjFunctionRows);
    ntCValues->releaseBlockOfRows(ntCValuesRows);
    ntCCentroids->releaseBlockOfRows(ntCCentroidsRows);
    ntCCentroids->getBlockOfRows(0, nClusters, writeOnly, ntCCentroidsRows);
    return st;
}

template <typename algorithmFPType>
Status KMeansDistributedStep1KernelUCAPI<algorithmFPType>::finalizeCompute(size_t na, const NumericTable * const * a, size_t nr,
                                                                                   const NumericTable * const * r, const Parameter * par)
{
    if (!par->assignFlag) return Status();

    NumericTable * ntPartialAssignments = const_cast<NumericTable *>(a[0]);
    NumericTable * ntAssignments        = const_cast<NumericTable *>(r[0]);
    const size_t n                      = ntPartialAssignments->getNumberOfRows();

    ReadRows<int, sse2> inBlock(*ntPartialAssignments, 0, n);
    DAAL_CHECK_BLOCK_STATUS(inBlock);
    const int * inAssignments = inBlock.get();

    WriteOnlyRows<int, sse2> outBlock(*ntAssignments, 0, n);
    DAAL_CHECK_BLOCK_STATUS(outBlock);
    int * outAssignments = outBlock.get();

    PRAGMA_IVDEP
    for (size_t i = 0; i < n; i++)
    {
        outAssignments[i] = inAssignments[i];
    }
    return Status();
}

} // namespace internal
} // namespace kmeans
} // namespace algorithms
} // namespace daal
