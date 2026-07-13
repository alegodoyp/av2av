"""Drives ByteDance's LatentSync (https://github.com/bytedance/LatentSync) as
an alternative video renderer, targeting the mouth-interior/teeth softness
that's the resolution ceiling of both unit2av's own renderer and Wav2Lip
(both fixed at 96x96). LatentSync is a diffusion model operating at
256x256/512x512 and is specifically reported to render teeth/tongue more
faithfully.

Unlike wav2lip_render.py, this is NOT an in-process integration:
LatentSync needs its own conda env (torch 2.5.1, diffusers, insightface,
onnxruntime-gpu -- incompatible with this project's pinned deps, e.g.
numpy==1.23.5 vs. their numpy==1.26.4, opencv-python==4.5.4.60 vs. 4.9.0.80),
and does its own face detection/alignment/diffusion sampling/paste-back
internally as a single black-box pipeline (see their LipsyncPipeline). So we
shell out to their scripts/inference.py in its own environment, the same
pattern already used by scripts/prepare_data.py for fairseq-preprocess.

LatentSync's pipeline produces a finished, audio-muxed video directly --
callers should treat its output as the final out_vid_path and skip
util.save_video() (and GFPGAN restoration) entirely for this renderer.
"""
import os
import shutil
import subprocess

import numpy as np
import cv2

from util import save_audio


def _write_silent_video(frames_bgr, out_path, fps=25):
    h, w = frames_bgr.shape[1], frames_bgr.shape[2]
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames_bgr:
        out.write(frame)
    out.release()


def render_video_latentsync(
    wav, full_video, out_vid_path,
    repo_dir, python_bin,
    unet_config_path="configs/unet/stage2_512.yaml",
    ckpt_path="checkpoints/latentsync_unet.pt",
    inference_steps=20, guidance_scale=1.5, fps=25,
):
    """Runs LatentSync end-to-end and writes the result to out_vid_path.

    wav: synthesized audio (from unit2av's CodeHiFiGANModel_spk), 16kHz.
    full_video: duration-matched background frames (BGR uint8), same as
        what unit2av/wav2lip_render use -- LatentSync does its own face
        detection on these directly, no pre-cropping needed.
    repo_dir: path to a local clone of bytedance/LatentSync.
    python_bin: path to that repo's own conda env's python executable.
    """
    out_vid_path = os.path.abspath(out_vid_path)
    work_dir = os.path.join(os.path.dirname(out_vid_path), "_latentsync_tmp")
    os.makedirs(work_dir, exist_ok=True)
    temp_video_in = os.path.abspath(os.path.join(work_dir, "driving_video.mp4"))
    temp_audio_in = os.path.abspath(os.path.join(work_dir, "driving_audio.wav"))

    _write_silent_video(full_video, temp_video_in, fps=fps)
    save_audio(np.asarray(wav, dtype=np.float32), temp_audio_in, sampling_rate=16000)

    cmd = [
        python_bin, "-m", "scripts.inference",
        "--unet_config_path", unet_config_path,
        "--inference_ckpt_path", ckpt_path,
        "--video_path", temp_video_in,
        "--audio_path", temp_audio_in,
        "--video_out_path", out_vid_path,
        "--inference_steps", str(inference_steps),
        "--guidance_scale", str(guidance_scale),
        "--enable_deepcache",
    ]
    try:
        subprocess.run(cmd, cwd=repo_dir, check=True)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
