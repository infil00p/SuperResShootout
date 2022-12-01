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

#include <jni.h>
#include <string>
#include "ORTSuperRes.h"
#include "TFSuperRes.h"
#include "PyTorchSuperRes.h"

extern "C"
JNIEXPORT jboolean JNICALL
Java_org_infil00p_superresstats_MainActivity_doTest(JNIEnv *env, jobject thiz,
                                                           jstring external_file_path,
                                                           jint device, jint datatype) {

    auto cDevice = static_cast<MLStats::Device>(device);
    auto cType = static_cast<MLStats::DataType>(datatype);

    std::string externalPath = std::string(env->GetStringUTFChars(external_file_path, nullptr));

    {
        std::unique_ptr<MLStats::TFSuperRes> tflite = std::make_unique<MLStats::TFSuperRes>(cDevice, cType);
        tflite->loadModel();
        auto results = tflite->doTestRun(externalPath);
        MLStats::Model::createReport(results, externalPath);
    }

    // We only run ORT on CPU right now, there is no GPU execution provider for this
    if(cDevice == MLStats::Device::CPU)
    {
        std::unique_ptr<MLStats::ORTSuperRes> ort = std::make_unique<MLStats::ORTSuperRes>(cDevice, cType);
        ort->loadModel();
        auto results = ort->doTestRun(externalPath);
        MLStats::Model::createReport(results, externalPath);
    }

    // PyTorch
    {
        std::unique_ptr<MLStats::PyTorchSuperRes> pytorch = std::make_unique<MLStats::PyTorchSuperRes>(cDevice, cType);
        pytorch->loadModel();
        auto results = pytorch->doTestRun(externalPath);
        MLStats::Model::createReport(results, externalPath);
    }

    return true;

}