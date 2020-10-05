/* file: select_indexed.cpp */
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

#include "services/daal_defines.h"
#include "src/sycl/select_indexed.h"
#include "src/sycl/cl_kernels/select_indexed.cl"
#include "services/internal/execution_context.h"
#include "src/externals/service_rng.h"
#include "src/algorithms/engines/engine_batch_impl.h"
#include "services/daal_string.h"
#include "src/services/service_data_utils.h"
#include "src/externals/service_ittnotify.h"

using namespace daal::data_management;
using namespace daal::services::internal;

const uint32_t maxInt32 = static_cast<uint32_t>(daal::services::internal::MaxVal<int32_t>::get());

namespace daal
{
namespace services
{
namespace internal
{
namespace sycl
{
namespace selection
{
DAAL_ITTNOTIFY_DOMAIN(daal.oneapi.internal.select.select_indexed);

void run_quick_select_simd(ExecutionContextIface & context, ClKernelFactoryIface & kernelFactory, const UniversalBuffer & dataVectors,
                           const UniversalBuffer & indexVectors, const UniversalBuffer & rndSeq, uint32_t nRndSeq, uint32_t nK, uint32_t nVectors,
                           uint32_t vectorSize, uint32_t lastVectorSize, uint32_t vectorOffset, QuickSelectIndexed::Result & result,
                           services::Status * status)
{
    if (nRndSeq > maxInt32 || nRndSeq == 0 || vectorSize > maxInt32 || lastVectorSize > vectorSize || nK > vectorSize || nK > lastVectorSize
        || vectorOffset > maxInt32)
    {
        services::internal::tryAssignStatus(status, Status(services::ErrorBufferSizeIntegerOverflow));
        return;
    }

    auto func_kernel = kernelFactory.getKernel("quick_select_group", status);
    DAAL_CHECK_STATUS_PTR(status);

    const uint32_t maxWorkItemsPerGroup = 16;
    KernelRange localRange(1, maxWorkItemsPerGroup);
    KernelRange globalRange(nVectors, maxWorkItemsPerGroup);

    KernelNDRange range(2);
    range.global(globalRange, status);
    DAAL_CHECK_STATUS_PTR(status);
    range.local(localRange, status);
    DAAL_CHECK_STATUS_PTR(status);

    KernelArguments args(10);
    args.set(0, dataVectors, AccessModeIds::read);
    args.set(1, indexVectors, AccessModeIds::read);
    args.set(2, result.values, AccessModeIds::write);
    args.set(3, result.indices, AccessModeIds::write);
    args.set(4, rndSeq, AccessModeIds::read);
    args.set(5, static_cast<int32_t>(nRndSeq));
    args.set(6, static_cast<int32_t>(vectorSize));
    args.set(7, static_cast<int32_t>(lastVectorSize));
    args.set(8, static_cast<int32_t>(nK));
    args.set(9, static_cast<int32_t>(vectorOffset));

    context.run(range, func_kernel, args, status);
    DAAL_CHECK_STATUS_PTR(status);
}

void run_direct_select_simd(ExecutionContextIface & context, ClKernelFactoryIface & kernelFactory, const UniversalBuffer & dataVectors, uint32_t nK,
                            uint32_t nVectors, uint32_t vectorSize, uint32_t lastVectorSize, uint32_t vectorOffset,
                            QuickSelectIndexed::Result & result, services::Status * status)
{
    if (vectorSize > maxInt32 || lastVectorSize > vectorSize || vectorOffset > maxInt32)
    {
        services::internal::tryAssignStatus(status, Status(services::ErrorBufferSizeIntegerOverflow));
        return;
    }
    auto func_kernel = kernelFactory.getKernel("direct_select_group", status);
    DAAL_CHECK_STATUS_PTR(status);

    const uint32_t maxWorkItemsPerGroup = 16;
    KernelRange localRange(1, maxWorkItemsPerGroup);
    KernelRange globalRange(nVectors, maxWorkItemsPerGroup);

    KernelNDRange range(2);
    range.global(globalRange, status);
    DAAL_CHECK_STATUS_PTR(status);
    range.local(localRange, status);
    DAAL_CHECK_STATUS_PTR(status);

    KernelArguments args(7);
    args.set(0, dataVectors, AccessModeIds::read);
    args.set(1, result.values, AccessModeIds::write);
    args.set(2, result.indices, AccessModeIds::write);
    args.set(3, static_cast<int32_t>(vectorSize));
    args.set(4, static_cast<int32_t>(lastVectorSize));
    args.set(5, static_cast<int32_t>(vectorOffset));
    if (dataVectors.type() == TypeIds::float32)
    {
        args.set(6, FLT_MAX);
    }
    else
    {
        args.set(6, DBL_MAX);
    }

    context.run(range, func_kernel, args, status);
    DAAL_CHECK_STATUS_PTR(status);
}

void SelectIndexed::convertIndicesToLabels(const UniversalBuffer & indices, const UniversalBuffer & labels, uint32_t nVectors, uint32_t vectorSize,
                                           uint32_t vectorOffset, services::Status * status)
{
    Status st;
    auto index2labels = labels.template get<int>().toHost(ReadWriteMode::readOnly, &st);
    services::internal::tryAssignStatus(status, st);
    if (!st.ok())
    {
        return;
    }
    auto index2labelsPtr = index2labels.get();
    if (!index2labelsPtr)
    {
        services::internal::tryAssignStatus(status, Status(ErrorNullPtr));
        return;
    }
    auto outIndex = indices.template get<int>().toHost(ReadWriteMode::readWrite, &st);
    services::internal::tryAssignStatus(status, st);
    if (!st.ok())
    {
        return;
    }
    auto outIndexPtr = outIndex.get();
    if (!outIndexPtr)
    {
        services::internal::tryAssignStatus(status, Status(ErrorNullPtr));
        return;
    }
    for (size_t vec = 0; vec < nVectors; vec++)
        for (size_t k = 0; k < vectorSize; k++)
        {
            int index                         = outIndexPtr[vec * vectorSize + k];
            outIndexPtr[vec * vectorSize + k] = index2labelsPtr[vec * vectorOffset + index];
        }
}

void QuickSelectIndexed::buildProgram(ClKernelFactoryIface & kernelFactory, const TypeId & vectorTypeId, services::Status * status)
{
    services::String fptype_name = getKeyFPType(vectorTypeId);
    auto build_options           = fptype_name;
    build_options.add("-cl-std=CL1.2 ");

    services::String cachekey("__daal_oneapi_internal_qselect_indexed_");
    cachekey.add(fptype_name);
    kernelFactory.build(ExecutionTargetIds::device, cachekey.c_str(), quick_select_simd, build_options.c_str(), status);
    DAAL_CHECK_STATUS_PTR(status);
}

SelectIndexed::Result & QuickSelectIndexed::selectIndices(const UniversalBuffer & dataVectors, const UniversalBuffer & tempIndices,
                                                          const UniversalBuffer & rndSeq, uint32_t nRndSeq, uint32_t nK, uint32_t nVectors,
                                                          uint32_t vectorSize, uint32_t lastVectorSize, uint32_t vectorOffset, Result & result,
                                                          services::Status * status)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(QuickSelectIndexed.select);

    auto & context       = services::internal::getDefaultContext();
    auto & kernelFactory = context.getClKernelFactory();

    buildProgram(kernelFactory, dataVectors.type(), status);
    DAAL_CHECK_STATUS_RETURN_IF_FAIL(status, result);

    run_quick_select_simd(context, kernelFactory, dataVectors, tempIndices, rndSeq, nRndSeq, nK, nVectors, vectorSize, lastVectorSize, vectorOffset,
                          result, status);
    DAAL_CHECK_STATUS_RETURN_IF_FAIL(status, result);
    return result;
}

Status QuickSelectIndexed::adjustIndexBuffer(uint32_t number, uint32_t size)
{
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(uint32_t, size, number);
    uint32_t newSize = size * number;
    Status st;
    if (_indexSize < newSize)
    {
        auto & context = Environment::getInstance()->getDefaultExecutionContext();
        _indices       = context.allocate(TypeIds::id<int>(), newSize, &st);
        _indexSize     = newSize;
    }
    return st;
}

Status QuickSelectIndexed::init(Params & par)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.RNG);
    Status st;
    _nRndSeq        = (par.dataSize > _maxSeqLength || par.dataSize < 2) ? _maxSeqLength : par.dataSize;
    auto engineImpl = dynamic_cast<daal::algorithms::engines::internal::BatchBaseImpl *>(&(*par.engine));
    if (!engineImpl)
    {
        return Status(ErrorIncorrectEngineParameter);
    }
    size_t numbers[_maxSeqLength];
    daal::internal::RNGs<size_t, sse2> rng;
    rng.uniform(_nRndSeq, &numbers[0], engineImpl->getState(), 0, (size_t)(_nRndSeq - 1));
    float values[_maxSeqLength];
    for (uint32_t i = 0; i < _nRndSeq; i++)
    {
        values[i] = static_cast<float>(numbers[i]) / (_nRndSeq - 1);
    }
    auto & context = Environment::getInstance()->getDefaultExecutionContext();
    _rndSeq        = context.allocate(par.type, _nRndSeq, &st);
    DAAL_CHECK_STATUS_VAR(st);
    context.copy(_rndSeq, 0, (void *)&values[0], 0, _nRndSeq, &st);
    DAAL_CHECK_STATUS_VAR(st);
    return st;
}

SelectIndexed * QuickSelectIndexed::create(Params & par, daal::services::Status * st)
{
    QuickSelectIndexed * ret = new QuickSelectIndexed();
    daal::services::Status status;
    if (!ret)
    {
        services::internal::tryAssignStatus(st, Status(ErrorMemoryAllocationFailed));
        return nullptr;
    }
    status = ret->init(par);
    services::internal::tryAssignStatus(st, status);
    if (!status.ok())
    {
        delete ret;
        return nullptr;
    }
    return ret;
}

void DirectSelectIndexed::buildProgram(ClKernelFactoryIface & kernelFactory, const TypeId & vectorTypeId, uint32_t nK,
                                       daal::services::Status * status)
{
    services::String fptype_name = getKeyFPType(vectorTypeId);
    auto build_options           = fptype_name;
    build_options.add("-cl-std=CL1.2 -D __K__=");
    char buffer[DAAL_MAX_STRING_SIZE];
    daal::services::daal_int_to_string(buffer, DAAL_MAX_STRING_SIZE, nK);
    build_options.add(buffer);

    services::String cachekey("__daal_oneapi_internal_dselect_indexed_");
    cachekey.add(fptype_name);
    kernelFactory.build(ExecutionTargetIds::device, cachekey.c_str(), direct_select_simd, build_options.c_str(), status);
    DAAL_CHECK_STATUS_PTR(status);
}

SelectIndexed::Result & DirectSelectIndexed::selectIndices(const UniversalBuffer & dataVectors, uint32_t nK, uint32_t nVectors, uint32_t vectorSize,
                                                           uint32_t lastVectorSize, uint32_t vectorOffset, Result & result, services::Status * status)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(QuickSelectIndexed.select);

    auto & context       = services::internal::getDefaultContext();
    auto & kernelFactory = context.getClKernelFactory();

    buildProgram(kernelFactory, dataVectors.type(), nK, status);
    DAAL_CHECK_STATUS_RETURN_IF_FAIL(status, result);

    run_direct_select_simd(context, kernelFactory, dataVectors, nK, nVectors, vectorSize, lastVectorSize, vectorOffset, result, status);
    DAAL_CHECK_STATUS_RETURN_IF_FAIL(status, result);
    return result;
}

SelectIndexed * DirectSelectIndexed::create(Params & par, daal::services::Status * st)
{
    DirectSelectIndexed * ret = new DirectSelectIndexed(par.nK);
    if (!ret)
    {
        services::internal::tryAssignStatus(st, Status(ErrorMemoryAllocationFailed));
        return nullptr;
    }
    return ret;
}

SelectIndexedFactory::SelectIndexedFactory()
{
    _entries << makeEntry<DirectSelectIndexed>();
    _entries << makeEntry<QuickSelectIndexed>();
}

SelectIndexed * SelectIndexedFactory::create(int nK, SelectIndexed::Params & par, Status * st)
{
    for (size_t i = 0; i < _entries.size(); i++)
        if (_entries[i].inRange(nK))
        {
            return _entries[i].createMethod(par, st);
        }
    services::internal::tryAssignStatus(st, Status(ErrorMethodNotImplemented));
    return nullptr;
}

} // namespace selection
} // namespace sycl
} // namespace internal
} // namespace services
} // namespace daal
