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
    services::Status query(
                        NumericTable * ntData,
                        oneapi::internal::UniversalBuffer& distances,
                        oneapi::internal::UniversalBuffer& positions,
                        oneapi::internal::UniversalBuffer& counters,
                        size_t * indices, size_t n, size_t nRows, size_t dim, 
                        algorithmFPType eps,
                        int * const assignments,
                        NeighborhoodUCAPI<algorithmFPType> * neighs, 
                        bool doReset = false);
    void fillRows(
                        oneapi::internal::ExecutionContextIface& context,
                        NumericTable * ntData,
                        oneapi::internal::UniversalBuffer& nbrRows,
                        size_t * indices, 
                        size_t pos, 
                        size_t number,
                        size_t dim, 
                        services::Status* status);
    size_t getFreeIndices(
                        size_t first, 
                        size_t size, 
                        size_t nRows,
                        int * const assignments,
                        VectorUCAPI<size_t>& data);
    void initDistances(
                        oneapi::internal::ExecutionContextIface& context,
                        oneapi::internal::UniversalBuffer& querySq,
                        oneapi::internal::UniversalBuffer& dataSq,
                        oneapi::internal::UniversalBuffer& distances,
                        uint32_t dataBlockSize,
                        uint32_t probesBlockSize,
                        services::Status* st);
    void computeDistances(
                        oneapi::internal::ExecutionContextIface& context,
                        const services::Buffer<algorithmFPType>& data,
                        const services::Buffer<algorithmFPType>& query,
                        oneapi::internal::UniversalBuffer& distances,
                        uint32_t dsize,
                        uint32_t qsize,
                        uint32_t nFeatures,
                        services::Status* st);
    void partition(
                        oneapi::internal::ExecutionContextIface& context,
                        oneapi::internal::UniversalBuffer& distances,
                        oneapi::internal::UniversalBuffer& positions,
                        oneapi::internal::UniversalBuffer& counters,
                        uint32_t dsize,
                        uint32_t qsize,
                        algorithmFPType pivot,
                        services::Status* st);

    void buildProgram(
                        oneapi::internal::ClKernelFactoryIface & kernel_factory, 
                        services::Status * st);

    size_t _queryBlockSize = 256;
    size_t _dataBlockSize = 2048;
    size_t _maxLocalSize = 16;
};

} // namespace internal
} // namespace dbscan
} // namespace algorithms
} // namespace daal

#endif
