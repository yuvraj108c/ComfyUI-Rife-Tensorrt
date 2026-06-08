from ..trt_utilities import Engine
from ..utilities import download_file, load_node_config, rife_logger
import folder_paths
import time
import comfy.model_management as mm
import tensorrt
import os

# Image dimensions for TensorRT engine building
IMAGE_DIM_MIN = 256
IMAGE_DIM_OPT = 512
IMAGE_DIM_MAX = 3840

LOAD_RIFE_NODE_CONFIG = load_node_config()

class LoadRifeTensorrtModel:
    @classmethod
    def INPUT_TYPES(cls):
        # Use the pre-loaded configuration
        model_config = LOAD_RIFE_NODE_CONFIG.get("model", {})
        precision_config = LOAD_RIFE_NODE_CONFIG.get("precision", {})

        # Provide sensible defaults if keys are missing in the config
        model_options = model_config.get("options", ["rife49_ensemble_True_scale_1_sim"])
        model_default = model_config.get("default", "rife49_ensemble_True_scale_1_sim")
        model_tooltip = model_config.get("tooltip", "Select a RIFE model.")

        precision_options = precision_config.get("options", ["fp16", "fp32"])
        precision_default = precision_config.get("default", "fp16")
        precision_tooltip = precision_config.get("tooltip", "Select precision.")

        return {
            "required": {
                "model": (model_options, {"default": model_default, "tooltip": model_tooltip}),
                "precision": (precision_options, {"default": precision_default, "tooltip": precision_tooltip}),
            }
        }

    RETURN_NAMES = ("rife_trt_model",)
    RETURN_TYPES = ("RIFE_TRT_MODEL",)
    CATEGORY = "tensorrt"
    DESCRIPTION = "Load RIFE tensorrt models, they will be built automatically if not found."
    FUNCTION = "load_rife_tensorrt_model"

    def load_rife_tensorrt_model(self, model, precision):
        tensorrt_models_dir = os.path.join(folder_paths.models_dir, "tensorrt", "rife")
        onnx_models_dir = os.path.join(folder_paths.models_dir, "onnx")

        os.makedirs(tensorrt_models_dir, exist_ok=True)
        os.makedirs(onnx_models_dir, exist_ok=True)

        onnx_model_path = os.path.join(onnx_models_dir, f"{model}.onnx")

        # Build tensorrt model path with detailed naming
        engine_channel = 3
        engine_min_batch, engine_opt_batch, engine_max_batch = 1, 1, 1
        engine_min_h, engine_opt_h, engine_max_h = IMAGE_DIM_MIN, IMAGE_DIM_OPT, IMAGE_DIM_MAX
        engine_min_w, engine_opt_w, engine_max_w = IMAGE_DIM_MIN, IMAGE_DIM_OPT, IMAGE_DIM_MAX
        tensorrt_model_path = os.path.join(tensorrt_models_dir, f"{model}_{precision}_{engine_min_batch}x{engine_channel}x{engine_min_h}x{engine_min_w}_{engine_opt_batch}x{engine_channel}x{engine_opt_h}x{engine_opt_w}_{engine_max_batch}x{engine_channel}x{engine_max_h}x{engine_max_w}_{tensorrt.__version__}.trt")

        if not os.path.exists(tensorrt_model_path):
            if not os.path.exists(onnx_model_path):
                onnx_model_download_url = f"https://huggingface.co/yuvraj108c/rife-onnx/resolve/main/{model}.onnx"
                rife_logger.info(f"Downloading {onnx_model_download_url}")
                download_file(url=onnx_model_download_url, save_path=onnx_model_path)
            else:
                rife_logger.info(f"ONNX model found at: {onnx_model_path}")

            rife_logger.info(f"Building TensorRT engine for {onnx_model_path}: {tensorrt_model_path}")
            mm.soft_empty_cache()
            s = time.time()
            engine = Engine(tensorrt_model_path)
            engine.build(
                onnx_path=onnx_model_path,
                fp16=True if precision == "fp16" else False,
                input_profile=[
                    {
                        "img0": [(engine_min_batch, engine_channel, engine_min_h, engine_min_w), (engine_opt_batch, engine_channel, engine_opt_h, engine_opt_w), (engine_max_batch, engine_channel, engine_max_h, engine_max_w)],
                        "img1": [(engine_min_batch, engine_channel, engine_min_h, engine_min_w), (engine_opt_batch, engine_channel, engine_opt_h, engine_opt_w), (engine_max_batch, engine_channel, engine_max_h, engine_max_w)],
                    }
                ],
            )
            e = time.time()
            rife_logger.info(f"Time taken to build: {(e-s)} seconds")

        rife_logger.info(f"Loading TensorRT engine: {tensorrt_model_path}")
        mm.soft_empty_cache()
        engine = Engine(tensorrt_model_path)
        engine.load()
        engine.model_name = model

        return (engine,)
