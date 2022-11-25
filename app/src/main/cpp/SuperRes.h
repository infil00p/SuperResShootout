//
// Created by Joe Bowser on 2022-11-03.
//

#ifndef SUPERRESMLStats_SUPERRES_H
#define SUPERRESMLStats_SUPERRES_H

#include "opencv2/opencv.hpp"
#include <vector>
#include <string>
#include "rapidjson/document.h"
#include "rapidjson/prettywriter.h"

namespace MLStats {

    // We support all the quantizations
    enum DataType {
        Float32 = 0,
        Float16 = 1,
        Int8 = 2
    };

    // This is on Android and we intend to support all the accelerators
    enum Device {
        CPU = 0,
        GPU = 1,
        NNAPI = 2,
        SNPE = 3,
    };

    // This needs to contain the
    struct ResultSet {
        std::string imageUri;
        double duration;
        std::string framework;
        std::string device;
        std::string dataType;
    };


    class Model {
    public:
        static void createReport(std::vector<MLStats::ResultSet> & results, std::string & externalPath);
    };


    class SuperRes : Model {
    public:
        SuperRes();
        virtual bool loadModel() = 0;
        virtual std::vector<ResultSet> doTestRun(std::string & externalPath) = 0;
        std::pair<std::vector<cv::Mat>, cv::Mat> preProcessImage(std::string path);
        std::string postProcessImage(std::vector<cv::Mat> & yCrCb, cv::Mat & outputY,
                                     int index, std::string & framework,
                                     std::string & externalPath);
        std::vector<std::string> filePaths;

        std::string IMAGE_PATH = "/data/data/org.infil00p.superresstats/files/image_set/";

    };

}
#endif //SUPERRESMLStats_SUPERRES_H
