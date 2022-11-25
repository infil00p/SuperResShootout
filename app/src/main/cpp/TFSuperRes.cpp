//
// Created by Joe Bowser on 2022-11-03.
//

#include <tensorflow/lite/c/common.h>
#include "TFSuperRes.h"
#include <chrono>
#include <android/log.h>

#define PRE_TAG "SuperRes_TFLite"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR,    PRE_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,     PRE_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,     PRE_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG,    PRE_TAG, __VA_ARGS__)

bool MLStats::TFSuperRes::loadModel() {
    std::string fullPath = TF_PATH + "model_float32.tflite";
    model = TfLiteModelCreateFromFile(fullPath.c_str());
    options = TfLiteInterpreterOptionsCreate();
    return model != nullptr;
}

std::vector <MLStats::ResultSet> MLStats::TFSuperRes::doTestRun(std::string & externalPath) {
    std::vector <ResultSet> output;
    // Create the interpreter
    interpreter = TfLiteInterpreterCreate(model, options);
    // Allocate the tensors to warm up the model
    TfLiteInterpreterAllocateTensors(interpreter);
    for(size_t i = 0; i < filePaths.size(); ++i) {
        // Do the Open CV thing
        std::vector<cv::Mat> yCrCb;
        cv::Mat greyscale;
        ResultSet record;
        record.framework = "TFLite";
        std::tie(yCrCb, greyscale) = preProcessImage(filePaths[i]);


        TfLiteTensor * input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
        std::string name_of_tensor = std::string(TfLiteTensorName(input_tensor));
        // Copy the data over (THIS IS WHERE TF IS BAD)
        size_t imageSize = greyscale.rows * greyscale.cols * sizeof(float);
        if(greyscale.data == nullptr) {
            LOGE("We don't have data");
        }
        TfLiteTensorCopyFromBuffer(input_tensor, greyscale.data, imageSize);

        auto start = std::chrono::steady_clock::now();
        TfLiteInterpreterInvoke(interpreter);
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        record.duration = elapsed_seconds.count();

        // Create an output buffer for this
        std::vector<float> outBuffer;
        size_t outBufferSize = 672 * 672;
        outBuffer.resize(outBufferSize);
        const TfLiteTensor * output_tensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);
        TfLiteTensorCopyToBuffer(output_tensor, outBuffer.data(), outBufferSize * sizeof(float));
        // Write to an output file on internal storage
        cv::Mat outputGreyscale = cv::Mat(672, 672, CV_32F, outBuffer.data());
        record.imageUri = this->postProcessImage(yCrCb, outputGreyscale, i, FRAMEWORK, externalPath);
        // Create ResultSet record and push it on the results

        output.push_back(record);
    }
    return output;
}
