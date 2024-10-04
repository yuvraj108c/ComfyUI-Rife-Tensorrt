# https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/blob/main/vfi_utils.py

import os
import torch
import typing
import einops
from comfy.model_management import soft_empty_cache, get_torch_device
import numpy as np
from comfy.utils import ProgressBar
from colored import Fore, Back, Style  

DEVICE = get_torch_device()

def load_file_from_github_release(model_type, ckpt_name):
    error_strs = []
    for i, base_model_download_url in enumerate(BASE_MODEL_DOWNLOAD_URLS):
        try:
            return load_file_from_url(base_model_download_url + ckpt_name, get_ckpt_container_path(model_type))
        except Exception:
            traceback_str = traceback.format_exc()
            if i < len(BASE_MODEL_DOWNLOAD_URLS) - 1:
                print("Failed! Trying another endpoint.")
            error_strs.append(f"Error when downloading from: {base_model_download_url + ckpt_name}\n\n{traceback_str}")

    error_str = '\n\n'.join(error_strs)
    raise Exception(f"Tried all GitHub base urls to download {ckpt_name} but no suceess. Below is the error log:\n\n{error_str}")

def logger(msg):
    print(f'{Style.reset}{Fore.cyan}⚡ [Rife Tensorrt] - {msg}{Style.reset}')

def preprocess_frames(frames):
    return einops.rearrange(frames[..., :3], "n h w c -> n c h w")

def postprocess_frames(frames):
    return einops.rearrange(frames, "n c h w -> n h w c")[..., :3].cpu()

def get_frame_0_idx(frame_itr):
    frame_0_start_idx = frame_itr
    frame_0_end_idx = frame_itr+1
    return frame_0_start_idx, frame_0_end_idx

def get_frame_1_idx(frame_itr):
    frame_1_start_idx = frame_itr+1
    frame_1_end_idx = frame_itr+2
    return frame_1_start_idx, frame_1_end_idx

def generate_frames_rife(
        frames,
        clear_cache_after_n_frames,
        multiplier,
        return_middle_frame_function,
        cuda_streams
    ):
        
    output_frames = torch.zeros(multiplier*frames.shape[0], *frames.shape[1:], device="cpu")
    out_len = 0

    number_of_frames_processed_since_last_cleared_cuda_cache = 0
    pbar = ProgressBar(len(frames))

    rife_inference_datas = []

    for frame_itr in range(len(frames) - 1): # Skip the final frame since there are no frames after it

        # frame_0 = frames[frame_itr:frame_itr+1]
        # # frame_1 = frames[frame_itr+1:frame_itr+2]
        # output_frames[out_len] = frame_0 # Start with first frame
        # out_len += 1

        for middle_i in range(1, multiplier):
            timestep = middle_i/multiplier
            # rife_inference_datas.append({
            #     "frame_0": frame_0,
            #     "frame_1": frame_1,
            #     "timestep": timestep,
            # })

            rife_inference_datas.append({
                "frame_0_idx":frame_itr,
                "frame_1_idx":frame_itr + 1,
                "timestep": timestep,
            })
    
    logger("data generated")
    batch_size = cuda_streams
    for data_idx in range(0, len(rife_inference_datas), batch_size):
        current_batch_size = min(batch_size, len(rife_inference_datas) - data_idx)

        data_batch = []
        for batch_no in range(current_batch_size):
            current_data_idx = data_idx + batch_no

            rife_inference_data = rife_inference_datas[current_data_idx]
            timestep = rife_inference_data["timestep"]
            timestep_t = torch.tensor([timestep], dtype=torch.float32).to(get_torch_device())

            frame_0_idx = rife_inference_data["frame_0_idx"]
            frame_1_idx = rife_inference_data["frame_1_idx"]

            frame_0 = frames[frame_0_idx].unsqueeze(0)#.to(DEVICE)
            frame_1 = frames[frame_1_idx].unsqueeze(0)#to(DEVICE)

            data_batch.append({
                "img0": frame_0,
                "img1": frame_1,
                "timestep": timestep_t,
            })

        middle_frames = return_middle_frame_function(data_batch)
        for middle_frame in middle_frames:
            output_frames[out_len] = middle_frame['output'].detach().cpu()
            out_len +=1
            # number_of_frames_processed_since_last_cleared_cuda_cache += 1

            # # Try to avoid a memory overflow by clearing cuda cache regularly
            # if number_of_frames_processed_since_last_cleared_cuda_cache >= clear_cache_after_n_frames:
            #     soft_empty_cache()
            #     number_of_frames_processed_since_last_cleared_cuda_cache = 0
            #     logger("Clearing cache...")

        pbar.update(current_batch_size)


    # Append final frame
    output_frames[out_len] = frames[-1:]
    logger(f"done! - {(len(frames) -1) * (multiplier-1)} new frames generated at resolution: {output_frames[0].shape}")
    out_len += 1

    # clear cache for courtesy
    soft_empty_cache()
    logger("Final clearing cache done ...")

    res = output_frames[:out_len]
    return res