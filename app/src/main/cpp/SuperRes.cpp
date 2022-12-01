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
 
#include <fstream>
#include "SuperRes.h"
#include <android/log.h>

#define PRE_TAG "SuperRes_PrePostCode"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR,    PRE_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,     PRE_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,     PRE_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG,    PRE_TAG, __VA_ARGS__)



std::pair<std::vector<cv::Mat>, cv::Mat> MLStats::SuperRes::preProcessImage(std::string path) {

    cv::Mat inputImage = cv::imread(path);
    cv::Mat inputYcc;
    try {
        cv::cvtColor(inputImage, inputYcc, cv::COLOR_BGR2YCrCb);
    }
    catch(cv::Exception exception)
    {
        LOGE("Error %s", exception.err.c_str());
    }

    std::vector<cv::Mat> yCbCr;
    cv::split(inputYcc, yCbCr);
    cv::Mat greyscale;
    yCbCr[0].convertTo(greyscale, CV_32F);
    // Divide by 255
    greyscale = greyscale/255;
    std::pair<std::vector<cv::Mat>, cv::Mat> out;
    out.first = yCbCr;
    out.second = greyscale;
    return out;
}

std::string MLStats::SuperRes::postProcessImage(std::vector<cv::Mat> & yCrCb, cv::Mat & outputY,
                                                 int index, std::string & framework,
                                                 std::string & externalPath) {
    cv::Mat temp_cy, out_cy, out_cr, out_cb;
    cv::resize(yCrCb[1], out_cr, cv::Size(outputY.rows, outputY.cols));
    cv::resize(yCrCb[2], out_cb, cv::Size(outputY.rows, outputY.cols));
    // This is why it gets blown out
    temp_cy = outputY * 255;
    temp_cy.convertTo(out_cy, CV_8UC1);
    std::vector<cv::Mat> resizedPic{ out_cy, out_cr, out_cb };
    cv::Mat colorImage;
    try {
        cv::merge(resizedPic, colorImage);
    }
    catch (cv::Exception exception){
        LOGE("Error %s", exception.err.c_str());
    }
    cv::Mat out_image_bgr;
    cv::cvtColor(colorImage, out_image_bgr, cv::COLOR_YCrCb2BGR);
    // Write to the file system
    std::string outUri = externalPath + "/out_" + std::to_string(index) + "_" + framework + ".png";
    // Let's just check the output of the net
    cv::imwrite(outUri, out_image_bgr);
    return outUri;
}

MLStats::SuperRes::SuperRes(MLStats::Device cDevice, MLStats::DataType cType) : Model(cDevice, cType) {
    for(int i = 1; i < 26; ++i) {
        std::string current_file;
        if(i < 10) {
            current_file = IMAGE_PATH + "0" + std::to_string(i) + ".png";
        }
        else {
            current_file = IMAGE_PATH + std::to_string(i) + ".png";
        }
        filePaths.push_back(current_file);
    }
}


