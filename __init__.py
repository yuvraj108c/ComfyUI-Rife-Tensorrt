import torch
import os
from comfy.model_management import get_torch_device
from comfy.utils import ProgressBar
from .vfi_utilities import preprocess_frames, postprocess_frames, generate_frames_rife, logger
from .trt_utilities import Engine, MultiStreamEngine
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
                "clear_cache_after_n_frames": ("INT", {"default": 100, "min": 1, "max": 1000}),
                "multiplier": ("INT", {"default": 2, "min": 1}),
                "cuda_streams": ("INT", {"default": 1, "min": 24}),
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
        clear_cache_after_n_frames=100,
        multiplier=2,
        cuda_streams=1,
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
            self.engine = MultiStreamEngine(engine_path)
            logger(f"Loading TensorRT engine: {engine_path}")
            self.engine.load()
            self.engine_label = engine
        else:
            logger(f"Using cached TensorRT engine: {engine_path}")
        
        self.engine.set_num_streams(cuda_streams)
        self.engine.activate()
        logger(f"Cuda streams: {cuda_streams}")
        self.engine.allocate_buffers(shape_dict=shape_dict)
        logger("allocation done")

        frames = preprocess_frames(frames)
        logger("preprocessing done")
        def return_middle_frame(data_batch):
            # s = time.time()
            results = self.engine.infer(data_batch)
            # e = time.time()
            # print(f"Time taken to infer: {(e-s)*1000} ms")

            return results

        result = generate_frames_rife(frames, clear_cache_after_n_frames, multiplier, return_middle_frame, cuda_streams)
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
