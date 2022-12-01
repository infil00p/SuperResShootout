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
