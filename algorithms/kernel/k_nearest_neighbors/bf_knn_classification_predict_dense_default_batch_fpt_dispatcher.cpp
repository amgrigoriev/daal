/* file: bf_knn_classification_predict_dense_default_batch_fpt_dispatcher.cpp */
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

#include "bf_knn_classification_predict.h"
#include "bf_knn_classification_predict_dense_default_batch_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER_SYCL(bf_knn_classification::prediction::BatchContainer, batch, DAAL_FPTYPE, bf_knn_classification::prediction::defaultDense)

namespace bf_knn_classification
{
namespace prediction
{
namespace interface1
{
template<typename algorithmFPType, bf_knn_classification::prediction::Method method>
Batch<algorithmFPType, method>::Batch() : classifier::prediction::Batch()
{
    _par = new ParameterType();
    initialize();
}

template<typename algorithmFPType, bf_knn_classification::prediction::Method method>
Batch<algorithmFPType, method>::Batch(const Batch &other): classifier::prediction::Batch(other)
{
    _param.k = other.parameter().k;
    _param.dataUseInModel = other.parameter().dataUseInModel;
    _param.engine = other.parameter().engine->clone();
    initialize();
}

template class Batch<DAAL_FPTYPE, defaultDense>;

} // namespace interface3
} // namespace prediction
} // namespace namespace bf_knn_classification
} // namespace algorithms
} // namespace daal