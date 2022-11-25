//
// Created by Joe Bowser on 2022-11-03.
//

#ifndef SUPERRESMLStats_TFSUPERRES_H
#define SUPERRESMLStats_TFSUPERRES_H

#include "SuperRes.h"
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate_c_api.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"

namespace MLStats
{


    class TFSuperRes : SuperRes {
    public:
        bool loadModel();
        std::vector<ResultSet> doTestRun(std::string & externalPath);
        ~TFSuperRes() {
            TfLiteInterpreterDelete(interpreter);
            TfLiteInterpreterOptionsDelete(options);

            TfLiteModelDelete(model);
        }

    private:
        TfLiteModel * model;
        TfLiteInterpreterOptions * options;
        TfLiteInterpreter* interpreter;
        TfLiteDelegate* delegate = nullptr;
        std::string FRAMEWORK = "tflite";
        std::string const TF_PATH = "/data/data/org.infil00p.superresstats/files/tflite/";

    };
}




#endif //SUPERRESMLStats_TFSUPERRES_H
