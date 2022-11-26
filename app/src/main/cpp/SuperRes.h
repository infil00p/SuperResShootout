//
// Created by Joe Bowser on 2022-11-03.
//

#ifndef SUPERRESMLStats_SUPERRES_H
#define SUPERRESMLStats_SUPERRES_H

#include "opencv2/opencv.hpp"
#include <vector>
#include <string>
#include "Model.h"

namespace MLStats {

    class SuperRes : public Model {
    public:
        SuperRes(Device cDevice, DataType cType);
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
