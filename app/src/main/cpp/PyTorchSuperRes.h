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
    class PyTorchSuperRes : SuperRes {
    public:
        PyTorchSuperRes(bool nhwc = false) {
            isNHWC = nhwc;
            if(isNHWC)
            {
                // LOLWTF?!?!  SO MUCH FOR THIS BEING CONSTANT
                FRAMEWORK="pytorch_nhwc";
            }
            else
            {
                FRAMEWORK="pytorch";
            }
        }
        bool loadModel();
        std::vector<ResultSet> doTestRun(std::string & externalPath);
    private:
        mutable torch::jit::script::Module mModule;
        std::string FRAMEWORK;
        bool isNHWC;
    };
}





#endif //SUPERRESMLStats_PYTORCHSUPERRES_H
