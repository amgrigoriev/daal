/* file: communicator_oneccl_sycl.h */
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

#ifndef __DAAL_ONEAPI_INTERNAL_COMMUNICATOR_ONECCL_SYCL_H__
#define __DAAL_ONEAPI_INTERNAL_COMMUNICATOR_ONECCL_SYCL_H__

#include <CL/cl.h>
#include <CL/sycl.hpp>
#include "ccl.h"
#include "oneapi/internal/communicator.h"
#include <iostream>

namespace daal
{
namespace preview
{
namespace comm
{
namespace internal
{
namespace interface1
{

class CommunicatorOneCclImpl : public CommunicatorIface
{
public:
    CommunicatorOneCclImpl(cl::sycl::queue & deviceQueue)
        : _deviceQueue(deviceQueue)
    {
        std::cout << "Construct comm" << std::endl;
        ccl_init();
        ccl_get_comm_rank(NULL, &_rank);
        ccl_get_comm_size(NULL, &_size);
        ccl_stream_create(ccl_stream_sycl, &deviceQueue, &_stream);
//        _rank = 0; 
//        _size = 1;
        std::cout << "Comm construction done" << std::endl;
    }


    void allReduceSum(UniversalBuffer src, UniversalBuffer dest, size_t count,
              services::Status * status = nullptr) DAAL_C11_OVERRIDE
    {
        DAAL_ASSERT(dest.type() == src.type());
        // TODO: Thread safe?
        try
        {
//            BufferCopier::copy(_deviceQueue, dest, 0, src, 0, count);
            BufferAllReducer::allReduceSum(_deviceQueue, _stream, dest, src, count);
        }
        catch (cl::sycl::exception const & e)
        {
            convertSyclExceptionToStatus(e, status);
        }
    }
    void allGatherV(UniversalBuffer src, srcCount, UniversalBuffer dest, size_t destCount,
              services::Status * status = nullptr) DAAL_C11_OVERRIDE
    {
        DAAL_ASSERT(dest.type() == src.type());
        // TODO: Thread safe?
        try
        {
//            BufferCopier::copy(_deviceQueue, dest, 0, src, 0, count);
            BufferAllReducer::allReduceSum(_deviceQueue, _stream, dest, destCount, src, srcCount);
        }
        catch (cl::sycl::exception const & e)
        {
            convertSyclExceptionToStatus(e, status);
        }
    }
    size_t size() DAAL_C11_OVERRIDE { return _size; }
    size_t rank() DAAL_C11_OVERRIDE { return _rank; }


private:
    cl::sycl::queue _deviceQueue;
    ccl_stream_t _stream;
    size_t _rank;
    size_t _size;
};

/** } */
} // namespace interface1

using interface1::CommunicatorOneCclImpl;

} // namespace internal
} // namespace preview
} // namespace preview
} // namespace daal

#endif // DAAL_SYCL_INTERFACE
