## Super Resolution Performance Stat Profiler.

This application is a test of the three main inference libraries on Android, TFLite, PyTorch and ONNX Runtime and it uses one of the the simplest model with the simplest Pre/Post processing, the Super Resolution model from the PyTorch Tutorials.

This is only on CPU for now, because it's the only delegate that can be used across all three inference frameworks.  SuperResoultion on PyTorch can't run on NNAPI due to PixelShuffle, and ONNX Runtime does not have a GPU backend on Android.

This code is licenced under the Apache Licence.
