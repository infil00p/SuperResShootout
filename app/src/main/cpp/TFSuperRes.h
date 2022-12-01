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

#ifndef SUPERRESMLStats_TFSUPERRES_H
#define SUPERRESMLStats_TFSUPERRES_H

#include "SuperRes.h"
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate_c_api.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"

namespace MLStats
{


    class TFSuperRes : public SuperRes {
    public:
        TFSuperRes(Device cDevice, DataType cType) : SuperRes(cDevice, cType)
        {

        }
        bool loadModel();
        std::vector<ResultSet> doTestRun(std::string & externalPath);
        ~TFSuperRes() {
            TfLiteInterpreterDelete(interpreter);
            TfLiteInterpreterOptionsDelete(mOptions);
            if(delegate != nullptr)
            {
                if(getDevice() == MLStats::Device::NNAPI)
                {
                    TfLiteNnapiDelegateDelete(delegate);
                }
                else
                {
                    TfLiteGpuDelegateV2Delete(delegate);
                }
            }

            TfLiteModelDelete(model);
        }

    private:
        TfLiteModel * model;
        TfLiteInterpreterOptions * mOptions;
        TfLiteInterpreter* interpreter;
        TfLiteDelegate* delegate = nullptr;
        std::string FRAMEWORK = "tflite";
        std::string const TF_PATH = "/data/data/org.infil00p.superresstats/files/tflite/";

    };
}




#endif //SUPERRESMLStats_TFSUPERRES_H
