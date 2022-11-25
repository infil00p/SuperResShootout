//
// Created by Joe Bowser on 2022-11-03.
//

#include "ORTSuperRes.h"

bool MLStats::ORTSuperRes::loadModel() {
    std::string full_model_path = ORT_PATH + "super_resolution.with_runtime_opt.ort";
    session = std::make_unique<Ort::Session>(env, full_model_path.c_str(), session_options);
    return true;
}

std::vector <MLStats::ResultSet> MLStats::ORTSuperRes::doTestRun(std::string & externalPath) {
    std::vector <ResultSet> output;
    int runLength = 100;
    // Allocate the tensors to warm up the model
    std::string path;
    for(int i = 0; i < filePaths.size(); ++i)
        {
        // Do the Open CV thing
        std::vector<cv::Mat> yCrCb;
        cv::Mat greyscale;
        ResultSet record;
        record.framework = "ORT";
        std::tie(yCrCb, greyscale) = preProcessImage(filePaths[i]);

        // Allocate the things
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        constexpr size_t input_tensor_size = 224 * 224;
        std::vector<int64_t> input_node_dims = {1, 1, 224, 224};
        auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, (float *)greyscale.data,
                                                            input_tensor_size,
                                                            input_node_dims.data(), 4);
        const char* input_names[] = {"input"};
        const char* output_names[] = {"output"};

        auto start = std::chrono::steady_clock::now();
        auto output_result = session->Run(Ort::RunOptions{nullptr}, input_names,
                                         &input_tensor, 1,
                                         output_names, 1);
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        record.duration = elapsed_seconds.count();

        // Copy the Data Out
        float* outputPtr = output_result.front().GetTensorMutableData<float>();
        cv::Mat outputGreyscale(cv::Size(672,672), CV_32F, outputPtr);

        record.imageUri = this->postProcessImage(yCrCb, outputGreyscale, i, FRAMEWORK, externalPath);
        // Create ResultSet record and push it on the results

        output.push_back(record);
    }
    return output;
}
