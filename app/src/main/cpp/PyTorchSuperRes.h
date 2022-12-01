/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 ~ Copyright 2022 Adobe
 ~
 ~ Licensed under the Apache License, Version 2.0 (the "License");
 ~ you may not use this file except in compliance with the License.
 ~ You may obtain a copy of the License at
 ~
 ~     http://www.apache.org/licenses/LICENSE-2.0
 ~
 ~ Unless required by applicable law or agreed to in writing, software
 ~ distributed under the License is distributed on an "AS IS" BASIS,
 ~ WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 ~ See the License for the specific language governing permissions and
 ~ limitations under the License.
 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

#ifndef SUPERRESMLStats_PYTORCHSUPERRES_H
#define SUPERRESMLStats_PYTORCHSUPERRES_H

#include "SuperRes.h"
#include "torch/script.h"
#include "opencv2/opencv.hpp"
#include "MobileCallGuard.h"

namespace MLStats
{
    class PyTorchSuperRes : public SuperRes {
    public:
        PyTorchSuperRes(Device cDevice, DataType cType) : SuperRes(cDevice, cType) {
        }
        bool loadModel();
        std::vector<ResultSet> doTestRun(std::string & externalPath);
    private:
        mutable torch::jit::script::Module mModule;
        std::string FRAMEWORK="pytorch";
        bool isNHWC=false;
    };
}





#endif //SUPERRESMLStats_PYTORCHSUPERRES_H
