//
// Created by Joe Bowser on 2022-11-03.
//

#ifndef SUPERRESMLStats_PYTORCHSUPERRES_H
#define SUPERRESMLStats_PYTORCHSUPERRES_H

#include "SuperRes.h"
#include "torch/script.h"
#include "opencv2/opencv.hpp"
#include "MobileCallGuard.h"

namespace MLStats
{
    class PyTorchSuperRes : public SuperRes {
    public:
        PyTorchSuperRes(Device cDevice, DataType cType) : SuperRes(cDevice, cType) {
        }
        bool loadModel();
        std::vector<ResultSet> doTestRun(std::string & externalPath);
    private:
        mutable torch::jit::script::Module mModule;
        std::string FRAMEWORK="pytorch";
        bool isNHWC=false;
    };
}





#endif //SUPERRESMLStats_PYTORCHSUPERRES_H
