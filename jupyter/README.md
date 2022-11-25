# Model Conversion

The Jupyter Notebook is based on this code sample from the PyTorch Tutorials and takes one of the simplest PyTorch models and converts it to all three platforms.  This test MIGHT be pretty meaningless since we're only testing 
the implementations of three ops, however this needs models for it to run.

The Jupyter Notebook produces PyTorch and ONNX Runtime models.  The next step in the conversion to Tensorflow and TFLite is as follows:

1. Install the OpenVINO Development Tools for ONNX
2. Follow the steps to convert to OpenVINO IR Format (https://docs.openvino.ai/latest/openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_ONNX.html)
3. Use this project (https://github.com/PINTO0309/openvino2tensorflow) to convert the OpenVINO to TF format.  I tried onnx2tf, but I wasn't able to get it to work

We only do Float32, but the converter can also produce quantized models as well, as with all TF Conversions, YMMV, but for this to be an honest comparison, we base our benchmarks on 
models that originate in PyTorch.
