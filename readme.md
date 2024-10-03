<div align="center">

# ComfyUI Rife TensorRT ⚡

[![python](https://img.shields.io/badge/python-3.10.12-green)](https://www.python.org/downloads/release/python-31012/)
[![cuda](https://img.shields.io/badge/cuda-12.4-green)](https://developer.nvidia.com/cuda-downloads)
[![trt](https://img.shields.io/badge/TRT-10.4.0-green)](https://developer.nvidia.com/tensorrt)
[![by-nc-sa/4.0](https://img.shields.io/badge/license-CC--BY--NC--SA--4.0-lightgrey)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en)

</div>

This project provides a [TensorRT](https://github.com/NVIDIA/TensorRT) implementation of [RIFE](https://github.com/hzwer/ECCV2022-RIFE) for ultra fast frame interpolation inside ComfyUI

This project is licensed under [CC BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/), everyone is FREE to access, use, modify and redistribute with the same license.

If you like the project, please give me a star! ⭐

---

<!-- ## ⏱️ Performance

_Note: The following results were benchmarked on FP16 engines inside ComfyUI, using 1000 similar frames_

| Device | FPS |
| :----: | :-: |
|  -  | -  | -->

## 🚀 Installation

Navigate to the ComfyUI `/custom_nodes` directory

```bash
git clone https://github.com/yuvraj108c/ComfyUI-Rife-Tensorrt
cd ./ComfyUI-Rife-Tensorrt
pip install -r requirements.txt
```

## 🛠️ Building Tensorrt Engine

1. Download one of the following onnx models:
   - [rife49_ensemble_True_scale_1_sim.onnx](https://huggingface.co/yuvraj108c/rife-onnx/resolve/main/rife49_ensemble_True_scale_1_sim.onnx)
   - [rife48_ensemble_True_scale_1_sim.onnx](https://huggingface.co/yuvraj108c/rife-onnx/resolve/main/rife48_ensemble_True_scale_1_sim.onnx)
   - [rife47_ensemble_True_scale_1_sim.onnx](https://huggingface.co/yuvraj108c/rife-onnx/resolve/main/rife47_ensemble_True_scale_1_sim.onnx)
2. Edit paths inside [export_trt.py](./export_trt.py) and build tensorrt engine by running:
   - `python export_trt.py`

3. Place the exported engine inside ComfyUI `/models/tensorrt/rife` directory

## ☀️ Usage

- Insert node by `Right Click -> tensorrt -> Rife Tensorrt`

## 🤖 Environment tested

- Ubuntu 22.04 LTS, Cuda 12.4, Tensorrt 10.4.0, Python 3.10, RTX 3070 GPU
- Windows (Not tested, but should work)

## 👏 Credits

- https://github.com/styler00dollar/VSGAN-tensorrt-docker
- https://github.com/Fannovel16/ComfyUI-Frame-Interpolation
- https://github.com/hzwer/ECCV2022-RIFE

## License

[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
