# https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/tree/main/vfi_utils.py
# https://github.com/yester31/TensorRT_Examples/blob/main/timm_to_trt_python1/onnx_export.py

import torch
import pathlib
import traceback
import os
from urllib.parse import urlparse
from torch.hub import download_url_to_file, get_dir
from rife_arch import IFNet
import onnx
from onnxsim import simplify

MODEL_TYPE = pathlib.Path(__file__).parent.name
CKPT_NAME_VER_DICT = {
    "rife40.pth": "4.0",
    "rife41.pth": "4.0", 
    "rife42.pth": "4.2", 
    "rife43.pth": "4.3", 
    "rife44.pth": "4.3", 
    "rife45.pth": "4.5",
    "rife46.pth": "4.6",
    "rife47.pth": "4.7",
    "rife48.pth": "4.7",
    "rife49.pth": "4.7",
    "sudo_rife4_269.662_testV1_scale1.pth": "4.0"
}
BASE_MODEL_DOWNLOAD_URLS = [
    "https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/",
    "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/",
    "https://github.com/dajes/frame-interpolation-pytorch/releases/download/v1.0.0/"
]
TORCH_DEVICE = "cuda:0"

def load_file_from_url(url, model_dir=None, progress=True, file_name=None):
    """Load file form http url, will download models if necessary.

    Ref:https://github.com/1adrianb/face-alignment/blob/master/face_alignment/utils.py

    Args:
        url (str): URL to be downloaded.
        model_dir (str): The path to save the downloaded model. Should be a full path. If None, use pytorch hub_dir.
            Default: None.
        progress (bool): Whether to show the download progress. Default: True.
        file_name (str): The downloaded file name. If None, use the file name in the url. Default: None.

    Returns:
        str: The path to the downloaded file.
    """
    if model_dir is None:  # use the pytorch hub_dir
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')

    os.makedirs(model_dir, exist_ok=True)

    parts = urlparse(url)
    file_name = os.path.basename(parts.path)
    if file_name is not None:
        file_name = file_name
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
    return cached_file

def load_file_from_github_release(model_type, ckpt_name):
    os.makedirs("models", exist_ok=True)
    error_strs = []
    for i, base_model_download_url in enumerate(BASE_MODEL_DOWNLOAD_URLS):
        try:
            return load_file_from_url(base_model_download_url + ckpt_name, "models")
        except Exception:
            traceback_str = traceback.format_exc()
            if i < len(BASE_MODEL_DOWNLOAD_URLS) - 1:
                print("Failed! Trying another endpoint.")
            error_strs.append(f"Error when downloading from: {base_model_download_url + ckpt_name}\n\n{traceback_str}")

    error_str = '\n\n'.join(error_strs)
    raise Exception(f"Tried all GitHub base urls to download {ckpt_name} but no suceess. Below is the error log:\n\n{error_str}")

def export_onnx(ckpt_name, ensemble, scale_factor):
    print(f"PyTorch version: {torch.__version__}")
    print(f"ONNX version: {onnx.__version__}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_path = load_file_from_github_release(MODEL_TYPE, ckpt_name)
    arch_ver = CKPT_NAME_VER_DICT[ckpt_name]
    interpolation_model = IFNet(arch_ver=arch_ver)
    interpolation_model.load_state_dict(torch.load(model_path))
    interpolation_model.eval().to(TORCH_DEVICE)

    # # dummy data
    img0 = torch.randn(1, 3, 512, 512).to(TORCH_DEVICE)
    img1 = torch.randn(1, 3, 512, 512).to(TORCH_DEVICE)
    timestep = torch.tensor([0.5], dtype=torch.float32).to(TORCH_DEVICE)

    # result = (interpolation_model(img0, img1, timestep))
    # print(result)

    onnx_save_name = f"{ckpt_name.split('.')[0]}_ensemble_{ensemble}_scale_{scale_factor}.onnx"
    onnx_save_path = os.path.join("models",onnx_save_name)

    dynamic_axes = {
        'img0': {2: 'height', 3: 'width'},
        'img1': {2: 'height', 3: 'width'},
        'output': {2: 'height', 3: 'width'},
    }
    
    with torch.no_grad():  # Disable gradients for efficiency
        torch.onnx.export(interpolation_model,
                (img0, img1, timestep), # ensemble + scale_factor set in forward fn
                onnx_save_path,
                export_params=True,
                opset_version=19,
                do_constant_folding=True,
                input_names=['img0', 'img1', 'timestep'],
                output_names=['output'],
                dynamic_axes=dynamic_axes
        )
        print(f"ONNX model exported to: {onnx_save_path}")

        # Verify the exported ONNX model
        onnx_model = onnx.load(onnx_save_path)
        onnx.checker.check_model(onnx_model)  # Perform a validity check
        print("ONNX model validation successful!")
        
        # print(onnx.helper.printable_graph(onnx_model.graph))

        sim_model_path = os.path.join("models",  f"{ckpt_name.split('.')[0]}_ensemble_{ensemble}_scale_{scale_factor}_sim.onnx")
        print("=> ONNX simplify start!")
        sim_onnx_model, check = simplify(onnx_model)  # convert(simplify)
        onnx.save(sim_onnx_model, sim_model_path)
        print("=> ONNX simplify done!")

        sim_model = onnx.load(sim_model_path)
        onnx.checker.check_model(sim_model)
        print("=> ONNX Model exported at ", sim_model_path)
        print("=> sim ONNX Model check done!")


export_onnx(ckpt_name="rife47.pth", ensemble=True, scale_factor=1)