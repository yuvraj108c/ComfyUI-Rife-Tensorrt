import torch
import os
from comfy.model_management import get_torch_device
from .vfi_utilities import preprocess_frames, postprocess_frames, generate_frames_rife, logger
from .trt_utilities import Engine
import folder_paths
import time
from polygraphy import cuda

ENGINE_DIR = os.path.join(folder_paths.models_dir, "tensorrt", "rife")

class RifeTensorrt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frames": ("IMAGE", ),
                "engine": (os.listdir(ENGINE_DIR),),
                "clear_cache_after_n_frames": ("INT", {"default": 50, "min": 1, "max": 1000}),
                "multiplier": ("INT", {"default": 2, "min": 1}),
                "use_cuda_graph": ("BOOLEAN", {"default": True}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "vfi"
    CATEGORY = "tensorrt"
    OUTPUT_NODE=True

    def vfi(
        self,
        frames,
        engine,
        clear_cache_after_n_frames=50,
        multiplier=2,
        use_cuda_graph=True,
        keep_model_loaded=False,
    ):
        B, H, W, C = frames.shape
        shape_dict = {
            "img0": {"shape": (1, 3, H, W)},
            "img1": {"shape": (1, 3, H, W)},
            "output": {"shape": (1, 3, H, W)},
        }

        cudaStream = cuda.Stream()
        engine_path = os.path.join(ENGINE_DIR, engine)
        if (not hasattr(self, 'engine') or self.engine_label != engine):
            self.engine = Engine(engine_path)
            logger(f"Loading TensorRT engine: {engine_path}")
            self.engine.load()
            self.engine.activate()
            self.engine_label = engine
        else:
            logger(f"Using cached TensorRT engine: {engine_path}")

        self.engine.allocate_buffers(shape_dict=shape_dict)

        frames = preprocess_frames(frames)

        def return_middle_frame(frame_0, frame_1, timestep):
            timestep_t = torch.tensor([timestep], dtype=torch.float32).to(get_torch_device())
            # s = time.time()
            output = self.engine.infer({"img0": frame_0, "img1": frame_1, "timestep": timestep_t}, cudaStream, use_cuda_graph)
            # e = time.time()
            # print(f"Time taken to infer: {(e-s)*1000} ms")

            result = output['output']
            return result

        result = generate_frames_rife(frames, clear_cache_after_n_frames, multiplier, return_middle_frame)
        out = postprocess_frames(result)
        
        if not keep_model_loaded:
            del self.engine, self.engine_label

        return (out,)


NODE_CLASS_MAPPINGS = {
    "RifeTensorrt": RifeTensorrt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RifeTensorrt": "⚡ Rife Tensorrt",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
