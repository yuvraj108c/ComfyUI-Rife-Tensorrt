import torch
import time
from trt_utilities import Engine
import tensorrt

print("Tensorrt version: ", tensorrt.__version__)

def export_trt(trt_path=None, onnx_path=None, use_fp16=True):
    if trt_path is None:
        trt_path = input(
            "Enter the path to save the TensorRT engine (e.g ./realesrgan.engine): ")
    if onnx_path is None:
        onnx_path = input(
            "Enter the path to the ONNX model (e.g ./realesrgan.onnx): ")

    engine = Engine(trt_path)

    torch.cuda.empty_cache()

    s = time.time()
    ret = engine.build(
        onnx_path,
        use_fp16,
        enable_preview=True,
        input_profile=[
            # any sizes from 256x256 to 3840x3840, batch size 1
            {
                "img0": [(1, 3, 256, 256), (1, 3, 512, 512), (1, 3, 3840, 3840)],
                "img1": [(1, 3, 256, 256), (1, 3, 512, 512), (1, 3, 3840, 3840)],
            },
        ],
    )
    e = time.time()
    print(f"Time taken to build: {(e-s)} seconds")
    print(f"Tensorrt engine saved at: {trt_path}")

    return ret


export_trt(trt_path="./models/rife49_ensemble_True_scale_1_sim.engine",
           onnx_path="./models/rife49_ensemble_True_scale_1_sim.onnx", use_fp16=True)
