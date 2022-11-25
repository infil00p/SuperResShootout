//
// Created by Joe Bowser on 2022-11-03.
//

#include "PyTorchSuperRes.h"

bool MLStats::PyTorchSuperRes::loadModel() {
    auto qengines = at::globalContext().supportedQEngines();
    if (std::find(qengines.begin(), qengines.end(), at::QEngine::QNNPACK) !=
        qengines.end())
    {
        at::globalContext().setQEngine(at::QEngine::QNNPACK);
    }

    MobileCallGuard guard;
    if(isNHWC)
    {
        mModule = torch::jit::load(PYTORCH_PATH + "superres_nhwc.pt");
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

        auto input = torch::from_blob(
                (float * )(greyscale.data),
                torch::IntArrayRef(sizes),
                at::TensorOptions(at::kFloat));
        std::vector<torch::jit::IValue> pytorchInputs;
        pytorchInputs.push_back(input);

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
            auto outTensor = pyTorchOutput.toTensor();
            // Wrap the Tensor
            cv::Mat outputGreyscale(cv::Size(672,672), CV_32F, outTensor.data_ptr());
            record.imageUri = this->postProcessImage(yCrCb, outputGreyscale, i, FRAMEWORK, externalPath);
            // Create ResultSet record and push it on the results
            output.push_back(record);
        }
    }
    return output;
}
