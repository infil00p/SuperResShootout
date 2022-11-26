#include <jni.h>
#include <string>
#include "ORTSuperRes.h"
#include "TFSuperRes.h"
#include "PyTorchSuperRes.h"

extern "C"
JNIEXPORT jboolean JNICALL
Java_org_infil00p_superresstats_MainActivity_doTest(JNIEnv *env, jobject thiz,
                                                           jstring external_file_path,
                                                           jint device, jint datatype) {

    auto cDevice = static_cast<MLStats::Device>(device);
    auto cType = static_cast<MLStats::DataType>(datatype);

    std::string externalPath = std::string(env->GetStringUTFChars(external_file_path, nullptr));

    {
        std::unique_ptr<MLStats::TFSuperRes> tflite = std::make_unique<MLStats::TFSuperRes>(cDevice, cType);
        tflite->loadModel();
        auto results = tflite->doTestRun(externalPath);
        MLStats::Model::createReport(results, externalPath);
    }

    // We only run ORT on CPU right now, there is no GPU execution provider for this
    if(cDevice == MLStats::Device::CPU)
    {
        std::unique_ptr<MLStats::ORTSuperRes> ort = std::make_unique<MLStats::ORTSuperRes>(cDevice, cType);
        ort->loadModel();
        auto results = ort->doTestRun(externalPath);
        MLStats::Model::createReport(results, externalPath);
    }

    // PyTorch
    {
        std::unique_ptr<MLStats::PyTorchSuperRes> pytorch = std::make_unique<MLStats::PyTorchSuperRes>(cDevice, cType);
        pytorch->loadModel();
        auto results = pytorch->doTestRun(externalPath);
        MLStats::Model::createReport(results, externalPath);
    }

    return true;

}