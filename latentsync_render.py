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
import imageio_ffmpeg

from util import save_audio


def _fix_blank_frames(frames_bgr, mean_threshold=3.0, std_threshold=3.0):
    """Some source videos have a black/blank frame at the very start (camera
    warm-up, fade-in) or scattered elsewhere -- confirmed here: samples/video1.mp4
    itself starts with a solid black frame. This is harmless for our own
    pipeline (get_crops() already forward/backward-fills missing detections),
    but LatentSync's own per-frame InsightFace detector has no such tolerance
    and raises immediately on any frame with no detectable face.

    Also flags near-uniform/flat frames (very low pixel variance), not just
    dark ones -- confirmed on video2: with duration stretching now able to
    pad a clip out well beyond its natural length, LatentSync's detector
    failed on a frame near the *original* clip's natural ending (proportional
    position is preserved by the nearest-neighbor stretch in
    unit2av/model.py), a common place for camera artifacts like fade-out or
    motion blur that aren't dark enough to trip a brightness-only check.

    Replaces any such frame with the nearest good frame so every frame handed
    to LatentSync has a real, detectable face, without changing the frame
    count (which must stay in sync with the driving audio).
    """
    flat = frames_bgr.reshape(len(frames_bgr), -1)
    means = flat.mean(axis=1)
    stds = flat.std(axis=1)
    degenerate = (means < mean_threshold) | (stds < std_threshold)
    if not degenerate.any():
        return frames_bgr

    good_idx = np.where(~degenerate)[0]
    if len(good_idx) == 0:
        return frames_bgr  # every frame is degenerate; nothing we can do here

    fixed = frames_bgr.copy()
    for i in np.where(degenerate)[0]:
        nearest = good_idx[np.argmin(np.abs(good_idx - i))]
        fixed[i] = frames_bgr[nearest]
    return fixed


def _write_silent_video(frames_bgr, out_path, fps=25):
    # Piping raw frames into a real ffmpeg process avoids the codec quirks
    # some cv2.VideoWriter backends have (e.g. mp4v producing garbled frames).
    h, w = frames_bgr.shape[1], frames_bgr.shape[2]
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "-",
        "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p",
        out_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for frame in frames_bgr:
        proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg failed while writing the LatentSync driving video")


def render_video_latentsync(
    wav, full_video, out_vid_path,
    repo_dir, python_bin,
    unet_config_path="configs/unet/stage2_512.yaml",
    ckpt_path="checkpoints/latentsync_unet.pt",
    inference_steps=20, guidance_scale=1.5, fps=25, sampling_rate=16000,
):
    """Runs LatentSync end-to-end and writes the result to out_vid_path.

    wav: synthesized audio (from unit2av's CodeHiFiGANModel_spk, optionally
        already passed through audio_restore's VoiceFixer pass -- in which
        case sampling_rate will be 44100, not the vocoder's native 16000).
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

    full_video = _fix_blank_frames(full_video)
    _write_silent_video(full_video, temp_video_in, fps=fps)
    save_audio(np.asarray(wav, dtype=np.float32), temp_audio_in, sampling_rate=sampling_rate)

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
    # Only clean up the driving video/audio on success -- on failure they're
    # left in place under work_dir for debugging (e.g. checking whether the
    # written video actually shows a normal, correctly-colored face).
    subprocess.run(cmd, cwd=repo_dir, check=True)
    shutil.rmtree(work_dir, ignore_errors=True)
