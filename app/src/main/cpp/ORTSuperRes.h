//
// Created by Joe Bowser on 2022-11-03.
//

#ifndef SUPERRESMLStats_ORTSUPERRES_H
#define SUPERRESMLStats_ORTSUPERRES_H
#include "SuperRes.h"
#include "onnxruntime_cxx_api.h"

namespace MLStats
{

    class ORTSuperRes : public SuperRes {
    public:
        ORTSuperRes(Device cDevice, DataType cType) : SuperRes(cDevice, cType)
        {

        }
        ~ORTSuperRes() {
            if(session != nullptr)
            {
                // Reset the session before closing
                session.reset();
            }
        }
        bool loadModel();
        std::vector<ResultSet> doTestRun(std::string & externalPath);
    private:
        Ort::Env env;
        std::unique_ptr<Ort::Session> session;
        Ort::SessionOptions session_options;
        std::string FRAMEWORK = "ort";
        std::string ORT_PATH = "/data/data/org.infil00p.superresstats/files/ort/";
    };
}

#endif //SUPERRESMLStats_ORTSUPERRES_H
