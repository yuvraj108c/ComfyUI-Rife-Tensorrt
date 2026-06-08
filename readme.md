<div align="center">

# ComfyUI Rife TensorRT ⚡

[![python](https://img.shields.io/badge/python-3.12.3-green)](https://www.python.org/downloads/release/python-3123//)
[![cuda](https://img.shields.io/badge/cuda-13.0-green)](https://developer.nvidia.com/cuda-downloads)
[![trt](https://img.shields.io/badge/TRT-10.14.1.48-green)](https://developer.nvidia.com/tensorrt)
[![by-nc-sa/4.0](https://img.shields.io/badge/license-CC--BY--NC--SA--4.0-lightgrey)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en)


This project provides a [TensorRT](https://github.com/NVIDIA/TensorRT) implementation of [RIFE](https://github.com/hzwer/ECCV2022-RIFE) for ultra fast frame interpolation inside ComfyUI

**Last tested**: 08 June 2026 (ComfyUI v0.23.0 | Torch 2.12.0 | Python 3.12.3 | L40S | CUDA 13.0 | Ubuntu 24.04)

<img width="938" height="236" alt="Screenshot 2026-06-08 at 09 21 51" src="https://github.com/user-attachments/assets/8228bd5f-7683-4b66-a476-220d5f667808" />

</div>

## ⭐ Support
If you like my projects and wish to see updates and new features, please consider supporting me. It helps a lot! 

[![ComfyUI-Depth-Anything-Tensorrt](https://img.shields.io/badge/ComfyUI--Depth--Anything--Tensorrt-blue?style=flat-square)](https://github.com/yuvraj108c/ComfyUI-Depth-Anything-Tensorrt)
[![ComfyUI-Upscaler-Tensorrt](https://img.shields.io/badge/ComfyUI--Upscaler--Tensorrt-blue?style=flat-square)](https://github.com/yuvraj108c/ComfyUI-Upscaler-Tensorrt)
[![ComfyUI-Dwpose-Tensorrt](https://img.shields.io/badge/ComfyUI--Dwpose--Tensorrt-blue?style=flat-square)](https://github.com/yuvraj108c/ComfyUI-Dwpose-Tensorrt)
[![ComfyUI-Rife-Tensorrt](https://img.shields.io/badge/ComfyUI--Rife--Tensorrt-blue?style=flat-square)](https://github.com/yuvraj108c/ComfyUI-Rife-Tensorrt)

[![ComfyUI-Whisper](https://img.shields.io/badge/ComfyUI--Whisper-gray?style=flat-square)](https://github.com/yuvraj108c/ComfyUI-Whisper)
[![ComfyUI_InvSR](https://img.shields.io/badge/ComfyUI__InvSR-gray?style=flat-square)](https://github.com/yuvraj108c/ComfyUI_InvSR)
[![ComfyUI-Thera](https://img.shields.io/badge/ComfyUI--Thera-gray?style=flat-square)](https://github.com/yuvraj108c/ComfyUI-Thera)
[![ComfyUI-Video-Depth-Anything](https://img.shields.io/badge/ComfyUI--Video--Depth--Anything-gray?style=flat-square)](https://github.com/yuvraj108c/ComfyUI-Video-Depth-Anything)
[![ComfyUI-PiperTTS](https://img.shields.io/badge/ComfyUI--PiperTTS-gray?style=flat-square)](https://github.com/yuvraj108c/ComfyUI-PiperTTS)

[![buy-me-coffees](https://i.imgur.com/3MDbAtw.png)](https://www.buymeacoffee.com/yuvraj108cZ)
[![paypal-donation](https://i.imgur.com/w5jjubk.png)](https://paypal.me/yuvraj108c)

---

## ⏱️ Performance

_Note: The following results were benchmarked on FP16 engines inside ComfyUI, using 2000 frames consisting of 2 alternating similar frames, averaged 2-3 times_

| Device | Rife Engine | Resolution| Multiplier | FPS |
| :----: | :-: | :-: | :-: | :-: |
|  H100  | rife49_ensemble_True_scale_1_sim | 512 x 512  | 2 | 45 |
|  H100  | rife49_ensemble_True_scale_1_sim | 512 x 512  | 4 | 57 |
|  H100  | rife49_ensemble_True_scale_1_sim | 1280 x 1280  | 2 | 21 |

## 🚀 Installation

Navigate to the ComfyUI `/custom_nodes` directory

```bash
git clone https://github.com/yuvraj108c/ComfyUI-Rife-Tensorrt
cd ./ComfyUI-Rife-Tensorrt
pip install -r requirements.txt
```

## 🛠️ Supported Models

The following RIFE models are supported and will be automatically downloaded and built:
   - **rife49_ensemble_True_scale_1_sim** (default) - Latest and most accurate
   - **rife48_ensemble_True_scale_1_sim** - Good balance of speed and quality
   - **rife47_ensemble_True_scale_1_sim** - Fastest option

Models are automatically downloaded from [HuggingFace](https://huggingface.co/yuvraj108c/rife-onnx) and TensorRT engines are built on first use.

## ☀️ Usage

1. **Load Model**: Insert `Right Click -> Add Node -> tensorrt -> Load Rife Tensorrt Model`
   - Choose your preferred RIFE model (rife47, rife48, or rife49)
   - Select precision (fp16 recommended for speed, fp32 for maximum accuracy)
   - The model will be automatically downloaded and TensorRT engine built on first use

2. **Process Frames**: Insert `Right Click -> Add Node -> tensorrt -> Rife Tensorrt`
   - Connect the loaded model from step 1
   - Input your video frames
   - Configure interpolation settings (multiplier, etc.)
   - Image resolutions between `256x256` and `3840x3840` are supported 


## 🚨 Updates

### 08 June 2026
- **Automatic Model Management**: No more manual downloads! Models are automatically downloaded from HuggingFace and TensorRT engines are built on demand. [PR#14](https://github.com/yuvraj108c/ComfyUI-Rife-Tensorrt/pull/14) by [@reaperhammer](https://github.com/reaperhammer)
- **Improved Workflow + Codebase**: New two-node system with `Load Rife Tensorrt Model` + `Rife Tensorrt` for better organization
- **Remove cuda-python**: No more cuda installation issues on windows

## 👏 Credits

- https://github.com/styler00dollar/VSGAN-tensorrt-docker
- https://github.com/Fannovel16/ComfyUI-Frame-Interpolation
- https://github.com/hzwer/ECCV2022-RIFE

## License

[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
