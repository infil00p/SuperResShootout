#include <jni.h>
#include <string>
#include "ORTSuperRes.h"
#include "TFSuperRes.h"
#include "PyTorchSuperRes.h"

extern "C"
JNIEXPORT jboolean JNICALL
Java_org_infil00p_superresstats_MainActivity_doTest(JNIEnv *env, jobject thiz,
                                                           jstring external_file_path) {

    std::string externalPath = std::string(env->GetStringUTFChars(external_file_path, nullptr));
    {
        std::unique_ptr<MLStats::TFSuperRes> tflite = std::make_unique<MLStats::TFSuperRes>();
        tflite->loadModel();
        auto results = tflite->doTestRun(externalPath);
        MLStats::Model::createReport(results, externalPath);
    }

    {
        std::unique_ptr<MLStats::ORTSuperRes> ort = std::make_unique<MLStats::ORTSuperRes>();
        ort->loadModel();
        auto results = ort->doTestRun(externalPath);
        MLStats::Model::createReport(results, externalPath);
    }

    // PyTorch NCHW
    {
        std::unique_ptr<MLStats::PyTorchSuperRes> pytorch = std::make_unique<MLStats::PyTorchSuperRes>(false);
        pytorch->loadModel();
        auto results = pytorch->doTestRun(externalPath);
        MLStats::Model::createReport(results, externalPath);
    }

    // PyTorch NHWC
    {
        std::unique_ptr<MLStats::PyTorchSuperRes> pytorchNHWC = std::make_unique<MLStats::PyTorchSuperRes>(true);
        pytorchNHWC->loadModel();
        auto results = pytorchNHWC->doTestRun(externalPath);
        MLStats::Model::createReport(results, externalPath);
    }

    return true;

}