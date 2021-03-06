/* file: df_regression_train_hist_batch_fpt_cpu_v1.cpp */
/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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
//  Implementation of decision forest regression training functions for the hist method
//--
*/

#include "src/algorithms/dtrees/forest/regression/inner/df_regression_train_container_v1.h"
#include "src/algorithms/dtrees/forest/regression/df_regression_train_kernel.h"

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace regression
{
namespace training
{
namespace interface1
{
template class BatchContainer<DAAL_FPTYPE, hist, DAAL_CPU>;
} // namespace interface1
} // namespace training
} // namespace regression
} // namespace decision_forest
} // namespace algorithms
} // namespace daal
