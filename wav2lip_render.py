"""Drives unit2av's video generation using Wav2Lip's official wav2lip_gan.pth
instead of this repo's own zero-shot FaceRenderer.

unit2av.model.FaceRenderer is architecturally a direct fork of Wav2Lip's own
Generator (face_encoder_blocks/face_decoder_blocks are identical layer for
layer) -- the only difference is the audio front-end (units vs. mel
spectrograms). Their weights are trained jointly with their own mel-based
audio_encoder, so we drive the whole Wav2Lip model end-to-end from the audio
we already synthesize (unit2av's CodeHiFiGANModel_spk output), rather than
mixing weights between the two models.

The raw generated 96x96 patches this produces are meant to be fed through
this repo's existing util.save_video() (seamlessClone blending + optional
GFPGAN restoration) exactly like unit2av's own FaceRenderer output would be --
this module only replaces the "generate a face patch per frame" step.
"""
import numpy as np
import torch
import cv2

from wav2lip_vendor.wav2lip_model import Wav2Lip
from wav2lip_vendor import audio as w2l_audio
from wav2lip_vendor.hparams import hparams as w2l_hparams

MEL_STEP_SIZE = 16
IMG_SIZE = 96


def load_wav2lip_model(checkpoint_path, use_cuda=True):
    device = "cuda" if use_cuda else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
        model = Wav2Lip()
        model.load_state_dict(state_dict)
        model = model.to(device)
    else:
        # This release of wav2lip_gan.pth was saved as a TorchScript archive
        # rather than a plain state_dict. torch.load already detected that
        # and dispatched to torch.jit.load internally, so `checkpoint` here
        # is already a loaded, callable RecursiveScriptModule -- use it as
        # the model directly instead of trying to index into it as a dict.
        print("wav2lip checkpoint is a TorchScript archive; using it directly.")
        model = checkpoint

    return model.eval()


def _pad_and_smooth_boxes(crops, frame_shape, pad_ratio=0.15, smooth_window=5):
    """Matches Wav2Lip's own face-crop conventions (see their inference.py
    face_detect()/get_smoothened_boxes()): per-frame face boxes are detected
    independently and wobble slightly frame to frame even on a static head,
    and their default padding extends the box (mainly downward) to make sure
    the chin is included -- the model was trained on crops built that way.
    We pad proportionally to box size (rather than their fixed 10px) so it
    scales with source resolution, then smooth over a short temporal window.
    """
    h, w = frame_shape[:2]
    padded = []
    for x1, y1, x2, y2 in crops:
        box_h, box_w = y2 - y1, x2 - x1
        pad_y, pad_x = box_h * pad_ratio, box_w * pad_ratio * 0.3
        padded.append([
            max(x1 - pad_x, 0),
            max(y1 - pad_x, 0),
            min(x2 + pad_x, w),
            min(y2 + pad_y, h),
        ])
    boxes = np.array(padded, dtype=np.float32)

    smoothed = boxes.copy()
    n = len(boxes)
    for i in range(n):
        window = boxes[i:i + smooth_window] if i + smooth_window <= n else boxes[max(0, n - smooth_window):]
        smoothed[i] = window.mean(axis=0)
    return smoothed


def _mel_chunks_for_frames(wav, num_frames, fps):
    mel = w2l_audio.melspectrogram(wav)
    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError(
            "Wav2Lip mel spectrogram contains NaNs -- the synthesized audio "
            "may be silent or invalid."
        )

    mel_idx_multiplier = 80. / fps
    chunks = []
    i = 0
    while True:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + MEL_STEP_SIZE > mel.shape[1]:
            chunks.append(mel[:, mel.shape[1] - MEL_STEP_SIZE:])
            break
        chunks.append(mel[:, start_idx:start_idx + MEL_STEP_SIZE])
        i += 1
        if len(chunks) >= num_frames:
            break
    return chunks


def render_video(model, wav, frames, crops, fps=25, use_cuda=True, batch_size=32, sampling_rate=16000):
    """Generates one 96x96 face patch per frame, Wav2Lip-style.

    frames: full background frames (BGR uint8), same convention as
        unit2av's `full_video`.
    crops: per-frame (x1, y1, x2, y2) face boxes.
    wav/sampling_rate: driving audio (from unit2av, optionally already
        restored by audio_restore's VoiceFixer pass, in which case
        sampling_rate is 44100, not the vocoder's native 16000). Wav2Lip's
        own mel-spectrogram math (wav2lip_vendor/hparams.py) is hardcoded for
        16kHz, so anything else is resampled down before mel extraction --
        this only affects the lip-sync timing features, not final audio
        quality (the caller keeps using the original wav for the soundtrack).

    Returns (gen_vid, used_crops): gen_vid is an (N, 96, 96, 3) uint8 array,
    a drop-in replacement for unit2av.model.UnitAVRenderer.forward()'s
    `gen_vid`. used_crops is the padded/smoothed version of `crops` actually
    used to build the input -- the caller must paste back with these same
    boxes (not the original `crops`), or the generated patch's framing won't
    match the region it gets resized into.
    """
    device = "cuda" if use_cuda else "cpu"
    wav = np.asarray(wav, dtype=np.float32)
    if sampling_rate != w2l_hparams.sample_rate:
        import librosa
        wav = librosa.resample(wav, orig_sr=sampling_rate, target_sr=w2l_hparams.sample_rate)
    mel_chunks = _mel_chunks_for_frames(wav, len(frames), fps)

    n = min(len(frames), len(crops), len(mel_chunks))
    frames, mel_chunks = frames[:n], mel_chunks[:n]
    crops = _pad_and_smooth_boxes(crops[:n], frames.shape[1:3])

    outputs = [None] * n
    img_batch, mel_batch, idx_batch = [], [], []

    def _flush():
        if not img_batch:
            return
        imgs = np.asarray(img_batch)
        mels = np.asarray(mel_batch)

        imgs_masked = imgs.copy()
        imgs_masked[:, IMG_SIZE // 2:] = 0
        imgs_in = np.concatenate((imgs_masked, imgs), axis=3) / 255.
        mels_in = mels.reshape(len(mels), mels.shape[1], mels.shape[2], 1)

        imgs_in = torch.FloatTensor(np.transpose(imgs_in, (0, 3, 1, 2))).to(device)
        mels_in = torch.FloatTensor(np.transpose(mels_in, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            pred = model(mels_in, imgs_in)
        pred = (pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.).astype(np.uint8)

        for out_idx, patch in zip(idx_batch, pred):
            outputs[out_idx] = patch

        img_batch.clear()
        mel_batch.clear()
        idx_batch.clear()

    for i in range(n):
        x1, y1, x2, y2 = [int(v) for v in crops[i]]
        face = frames[i][max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)]
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

        img_batch.append(face)
        mel_batch.append(mel_chunks[i])
        idx_batch.append(i)

        if len(img_batch) >= batch_size:
            _flush()
    _flush()

    return np.stack(outputs, axis=0), crops
