import torch
import os
from comfy.model_management import get_torch_device
from ..vfi_utilities import preprocess_frames, postprocess_frames, generate_frames_rife
from ..trt_utilities import Engine
import folder_paths
import time
import comfy.model_management as mm

class RifeTensorrt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frames": ("IMAGE", {"tooltip": "Input frames for video frame interpolation"}),
                "rife_trt_model": ("RIFE_TRT_MODEL", {"tooltip": "Tensorrt model built and loaded"}),
                "clear_cache_after_n_frames": ("INT", {"default": 100, "min": 1, "max": 1000, "tooltip": "Clear CUDA cache after processing this many frames"}),
                "multiplier": ("INT", {"default": 2, "min": 1, "tooltip": "Frame interpolation multiplier"}),
            },
        }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "vfi"
    CATEGORY = "tensorrt"

    def vfi(
        self,
        frames,
        rife_trt_model,
        clear_cache_after_n_frames=100,
        multiplier=2,
    ):
        B, H, W, C = frames.shape
        shape_dict = {
            "img0": {"shape": (1, 3, H, W)},
            "img1": {"shape": (1, 3, H, W)},
            "output": {"shape": (1, 3, H, W)},
        }

        cudaStream = torch.cuda.current_stream().cuda_stream
        engine = rife_trt_model
        engine.activate()
        engine.allocate_buffers(shape_dict=shape_dict)

        frames = preprocess_frames(frames)

        def return_middle_frame(frame_0, frame_1, timestep):
            timestep_t = torch.tensor([timestep], dtype=torch.float32).to(get_torch_device())
            output = engine.infer({"img0": frame_0, "img1": frame_1, "timestep": timestep_t}, cudaStream)
            result = output['output']
            return result

        result = generate_frames_rife(frames, clear_cache_after_n_frames, multiplier, return_middle_frame)
        out = postprocess_frames(result)

        engine.reset()

        return (out,)