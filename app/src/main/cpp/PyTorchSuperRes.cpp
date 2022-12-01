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

#include "PyTorchSuperRes.h"

bool MLStats::PyTorchSuperRes::loadModel() {
    auto qengines = at::globalContext().supportedQEngines();
    if (std::find(qengines.begin(), qengines.end(), at::QEngine::QNNPACK) !=
        qengines.end())
    {
        at::globalContext().setQEngine(at::QEngine::QNNPACK);
    }

    MobileCallGuard guard;
    if(getDevice() == MLStats::Device::GPU)
    {
        mModule = torch::jit::load(PYTORCH_PATH + "superres_vulkan.pt");
    }
    else
    {
        mModule = torch::jit::load(PYTORCH_PATH + "superres.pt");
    }
    mModule.eval();
    return true;
}

std::vector <MLStats::ResultSet> MLStats::PyTorchSuperRes::doTestRun(std::string & externalPath) {
    std::vector <ResultSet> output;
    int runLength = 100;
    // Allocate the tensors to warm up the model
    std::string path;
    for(int i = 0; i < filePaths.size(); ++i) {
        // Do the Open CV thing
        std::vector<cv::Mat> yCrCb;
        cv::Mat greyscale;
        ResultSet record;
        record.framework = FRAMEWORK;
        std::tie(yCrCb, greyscale) = preProcessImage(filePaths[i]);
        const auto sizes = std::vector<int64_t>{1, 1, 224, 224};

        std::vector<torch::jit::IValue> pytorchInputs;
        at::Tensor input  = torch::from_blob(
                (float * )(greyscale.data),
                torch::IntArrayRef(sizes),
                at::TensorOptions(at::kFloat));
        if(getDevice() == MLStats::Device::GPU && at::is_vulkan_available()) {
            auto gpuInput = input.vulkan();
            pytorchInputs.emplace_back(gpuInput);
        }
        else
        {
            pytorchInputs.emplace_back(input);
        }

        auto start = std::chrono::steady_clock::now();
        auto pyTorchOutput = [&]() {
            MobileCallGuard guard;
            return mModule.forward(pytorchInputs);
        }();
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        record.duration = elapsed_seconds.count();

        if (pyTorchOutput.tagKind() == "Tensor")
        {
            at::Tensor outTensor;
            auto rawOutTensor = pyTorchOutput.toTensor();
            if(getDevice() == MLStats::Device::GPU && at::is_vulkan_available()) {
                outTensor = rawOutTensor.cpu();
                record.device="GPU (Vulkan)";
            }
            else {
                outTensor = rawOutTensor;
                record.device="CPU";
            }

            // Wrap the Tensor
            cv::Mat outputGreyscale(cv::Size(672,672), CV_32F, outTensor.data_ptr());
            record.imageUri = this->postProcessImage(yCrCb, outputGreyscale, i, FRAMEWORK, externalPath);
            // Create ResultSet record and push it on the results
            output.push_back(record);
        }
    }
    return output;
}
