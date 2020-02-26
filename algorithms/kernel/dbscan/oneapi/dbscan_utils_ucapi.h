/* file: dbscan_utils_ucapi.h */
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
//  Common functions for DBSCAN on GPU
//--
*/

#ifndef __DBSCAN_UTILS_UCAPI_H_
#define __DBSCAN_UTILS_UCAPI_H_

#include "dbscan_types.h"

#include "threading.h"
#include "daal_defines.h"
#include "error_handling.h"
#include "service_memory.h"
#include "service_numeric_table.h"
#include "service_math.h"
#include "service_kernel_math.h"
#include "service_error_handling.h"

#include <iostream>

using namespace daal::internal;
using namespace daal::services::internal;
using namespace daal::algorithms::internal;

namespace daal
{
namespace algorithms
{
namespace dbscan
{
namespace internal
{
#define __DBSCAN_DEFAULT_QUEUE_SIZE        8
#define __DBSCAN_DEFAULT_VECTOR_SIZE       8
#define __DBSCAN_DEFAULT_NEIGHBORHOOD_SIZE 8

template <typename T>
class VectorUCAPI
{
    static const size_t defaultSize = __DBSCAN_DEFAULT_VECTOR_SIZE;

public:
    DAAL_NEW_DELETE();

    VectorUCAPI() : _data(nullptr), _size(0), _capacity(0) {
        std::cout << "Default constructor" << std::endl;
    }

    VectorUCAPI(size_t capacity) : _size(0), _capacity(capacity)
    {
        _data = static_cast<T *>(services::internal::service_calloc<T, sse2>(_capacity * sizeof(T)));
/*        if(_data == nullptr)
            std::cout << "      allocatin failed" << std::endl;
        else 
            std::cout << "      allocatin passed" << std::endl;
        *_data = 0;
        _data[_size] = 0;
        std::cout << "Check passed " << (long)_data << std::endl;*/
    }

    ~VectorUCAPI() { clear(); }

    VectorUCAPI(const VectorUCAPI &) = delete;
    VectorUCAPI(VectorUCAPI&& other) :  _data(other._data), _size(other._size), _capacity(other._capacity) 
    {
        other._data = nullptr;
        other._size = 0;
        other._capacity = 0;
    }
    VectorUCAPI & operator=(const VectorUCAPI &) = delete;

    void clear()
    {
//        std::cout << "Clear!!!!!!!!!!!!!!!!!!" << std::endl;
        if (_data)
        {
            services::daal_free(_data);
            _data = nullptr;
        }
    }

    void reset() { 
//        std::cout << "Reset " <<  (long)_data << std::endl;
        _size = 0; }

    services::Status push_back(const T & value)
    {
//        std::cout << "      push_back : " << _size << " " << _capacity << " " << (long)_data << std::endl;
        if (_size >= _capacity)
        {
//            std::cout << "      will grow " << std::endl;
            services::Status status = grow();
            DAAL_CHECK_STATUS_VAR(status);
        }
/*        _data[0] = value;
        std::cout << "      pre-added" << std::endl;*/
        _data[_size] = value;
//        std::cout << "      added" << std::endl;
        _size++;


        return services::Status();
    }

    inline T & operator[](size_t index) { return _data[index]; }

    size_t size() const {  /*std::cout << "\n       size " <<  (long)_data << std::endl;*/ return _size; }

    T * ptr() { return _data; }

private:
    services::Status grow()
    {
//        std::cout << "Grow" << std::endl;
        int result        = 0;
        _capacity         = (_capacity == 0 ? defaultSize : _capacity * 2);
        T * const newData = static_cast<T *>(services::internal::service_calloc<T, sse2>(_capacity * sizeof(T)));
        DAAL_CHECK_MALLOC(newData);

        if (_data != nullptr)
        {
            result = services::internal::daal_memcpy_s(newData, _size * sizeof(T), _data, _size * sizeof(T));
            services::daal_free(_data);
        }

        _data = newData;
        return (!result) ? services::Status() : services::Status(services::ErrorMemoryCopyFailedInternal);
    }

    T * _data;
    size_t _size;
    size_t _capacity;
};

template <typename T>
class QueueUCAPI
{
    static const size_t defaultSize = __DBSCAN_DEFAULT_QUEUE_SIZE;

public:
    DAAL_NEW_DELETE();

    QueueUCAPI() : _data(nullptr), _head(0), _tail(0), _capacity(0) {}

    ~QueueUCAPI() { clear(); }

    QueueUCAPI(const QueueUCAPI &) = delete;
    QueueUCAPI & operator=(const QueueUCAPI &) = delete;

    void clear()
    {
        if (_data)
        {
            services::daal_free(_data);
            _data = nullptr;
        }
    }

    void reset() { _head = _tail = 0; }

    services::Status push(const T & value)
    {
        if (_tail >= _capacity)
        {
            services::Status status = grow();
            DAAL_CHECK_STATUS_VAR(status)
        }

        _data[_tail] = value;
        _tail++;

        return services::Status();
    }

    T pop()
    {
        if (_head < _tail)
        {
            const T value = _data[_head];
            _head++;
            return value;
        }

        return (T)0;
    }
    VectorUCAPI<T> popVector(size_t blockSize)
    {
        VectorUCAPI<T> ret;
        for(size_t i = 0; i < blockSize; i++) 
        {
            if(empty())
            {
                break;
            }
            ret.push_back(pop());
        }
        return ret;
    } 

    bool empty() const { return (_head == _tail); }
    size_t head() const { return _head; }
    size_t tail() const { return _tail; }

    T * getInternalPtr(size_t ind)
    {
        if (ind >= _tail)
        {
            return nullptr;
        }

        return &_data[ind];
    }

private:
    services::Status grow()
    {
        int result        = 0;
        _capacity         = (_capacity == 0 ? defaultSize : _capacity * 2);
        T * const newData = static_cast<T *>(services::internal::service_calloc<T, sse2>(_capacity * sizeof(T)));
        DAAL_CHECK_MALLOC(newData);

        if (_data != nullptr)
        {
            result = services::internal::daal_memcpy_s(newData, _tail * sizeof(T), _data, _tail * sizeof(T));
            services::daal_free(_data);
            _data = nullptr;
        }

        _data = newData;
        return (!result) ? services::Status() : services::Status(services::ErrorMemoryCopyFailedInternal);
    }

    T * _data;
    size_t _head;
    size_t _tail;
    size_t _capacity;
};


template <typename FPType>
class NeighborhoodUCAPI
{
    static const size_t defaultSize = __DBSCAN_DEFAULT_NEIGHBORHOOD_SIZE;

public:
    DAAL_NEW_DELETE();

    NeighborhoodUCAPI() : _values(nullptr), _capacity(0), _size(0), _weight(0) {}

    NeighborhoodUCAPI(const NeighborhoodUCAPI &) = delete;
    NeighborhoodUCAPI & operator=(const NeighborhoodUCAPI &) = delete;

    ~NeighborhoodUCAPI() { clear(); }

    void clear()
    {
        if (_values)
        {
            services::daal_free(_values);
            _values = nullptr;
        }
        _capacity = _size = 0;
        _weight           = 0;
    }

    void reset()
    {
        _size   = 0;
        _weight = 0;
    }

    services::Status add(const size_t & value, FPType w)
    {
        if (_size >= _capacity)
        {
            services::Status status = grow();
            DAAL_CHECK_STATUS_VAR(status)
        }

        _values[_size] = value;
        _size++;
        _weight += w;

        return services::Status();
    }

    void addWeight(FPType w) { _weight += w; }

    size_t get(size_t id) const { return _values[id]; }

    size_t size() const { return _size; }

    FPType weight() const { return _weight; }

private:
    services::Status grow()
    {
        int result               = 0;
        _capacity                = (_capacity == 0 ? defaultSize : _capacity * 2);
        void * ptr               = services::daal_calloc(_capacity * sizeof(size_t));
        size_t * const newValues = static_cast<size_t *>(ptr);
        DAAL_CHECK_MALLOC(newValues);

        if (_values != nullptr)
        {
            result = services::internal::daal_memcpy_s(newValues, _size * sizeof(size_t), _values, _size * sizeof(size_t));
            services::daal_free(_values);
            _values = nullptr;
        }

        _values = newValues;

        return (!result) ? services::Status() : services::Status(services::ErrorMemoryCopyFailedInternal);
    }

    size_t * _values;

    size_t _capacity;
    size_t _size;

    FPType _weight;
};

} // namespace internal
} // namespace dbscan
} // namespace algorithms
} // namespace daal

#endif
