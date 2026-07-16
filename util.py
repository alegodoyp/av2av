import os
import soundfile as sf
import cv2
import ffmpeg
import imageio_ffmpeg

from face_restore import restore_frame

def process_units(units, reduce=False):
    if not reduce:
        return units

    out = [u for i, u in enumerate(units) if i == 0 or u != units[i - 1]]
    return out

def save_unit(unit, unit_path):
    os.makedirs(os.path.dirname(unit_path), exist_ok=True)
    with open(unit_path, "w") as f:
        f.write(unit)

import torch
import numpy as np

def save_audio(audio, audio_path, sampling_rate=16000):
    if os.path.dirname(audio_path):
        os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    
    if torch.is_tensor(audio):
        audio = audio.detach().cpu().numpy()
        
    if isinstance(audio, np.ndarray) and audio.dtype == np.float16:
         audio = audio.astype(np.float32)
         
    sf.write(
        audio_path,
        audio,
        sampling_rate,
    )

def get_audio_duration(audio_path):
    info = sf.info(audio_path)
    return info.frames / info.samplerate

def extract_audio_from_video(video_path, save_audio_path, sampling_rate=16000):
    os.makedirs(os.path.dirname(save_audio_path), exist_ok=True)
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    try:
        (
            ffmpeg.input(video_path)
            .output(
                save_audio_path,
                acodec="pcm_s16le",
                ac=1,
                ar=sampling_rate,
                loglevel="error",
            )
            .run(overwrite_output=True, cmd=ffmpeg_exe, capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        print(f"FFmpeg error: {e.stderr.decode('utf8')}")
        raise e

def save_video(audio, video, full_video, bbox, save_video_path, sampling_rate=16000, fps=25, vcodec="libx264", restorer=None):
    os.makedirs(os.path.dirname(save_video_path), exist_ok=True)
    temp_audio_path = os.path.splitext(save_video_path)[0]+".temp.wav"
    temp_video_path = os.path.splitext(save_video_path)[0]+".temp.avi"

    save_audio(audio, temp_audio_path, sampling_rate)

    frame_h, frame_w = full_video.shape[1], full_video.shape[2]
    out = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))
    
    for p, f, c in zip(video, full_video, bbox):
        x1, y1, x2, y2 = [int(_) for _ in c]
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, frame_w), min(y2, frame_h)
        width, height = x2 - x1, y2 - y1

        p = cv2.resize(p, (width, height), interpolation=cv2.INTER_CUBIC)

        # Poisson (seamless) blend instead of a hard pixel copy: a direct
        # f[y1:y2, x1:x2] = p paste leaves a visible rectangular seam with a
        # tone/exposure mismatch against the surrounding skin. seamlessClone
        # blends the color gradient at the boundary, but a full-rectangle
        # mask still shows a faint edge (esp. at the corners) where sharp
        # background texture meets the renderer's soft interior. An inset
        # ellipse follows the face's shape and keeps the seam off the
        # rectangle's corners entirely.
        center = (x1 + width // 2, y1 + height // 2)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(
            mask,
            (width // 2, height // 2),
            (max(int(width * 0.42), 1), max(int(height * 0.42), 1)),
            0, 0, 360, 255, -1,
        )
        try:
            f = cv2.seamlessClone(p, f, mask, center, cv2.NORMAL_CLONE)
        except cv2.error:
            f[y1:y2, x1:x2] = p

        f = restore_frame(restorer, f)
        out.write(f)

    out.release()
    
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg.output(
        ffmpeg.input(temp_video_path),
        ffmpeg.input(temp_audio_path),
        save_video_path,
        vcodec="libx264",
        acodec="aac",
        loglevel="panic",
    ).run(overwrite_output=True, cmd=ffmpeg_exe)

    os.remove(temp_audio_path)
    os.remove(temp_video_path)

