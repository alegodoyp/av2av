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

MEL_STEP_SIZE = 16
IMG_SIZE = 96


def load_wav2lip_model(checkpoint_path, use_cuda=True):
    device = "cuda" if use_cuda else "cpu"
    model = Wav2Lip()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    return model.eval()


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


def render_video(model, wav, frames, crops, fps=25, use_cuda=True, batch_size=32):
    """Generates one 96x96 face patch per frame, Wav2Lip-style.

    frames: full background frames (BGR uint8), same convention as
        unit2av's `full_video`.
    crops: per-frame (x1, y1, x2, y2) face boxes -- reuse the same ones used
        for pasting back, so the input crop matches the paste region.

    Returns an (N, 96, 96, 3) uint8 array, a drop-in replacement for
    unit2av.model.UnitAVRenderer.forward()'s `gen_vid`.
    """
    device = "cuda" if use_cuda else "cpu"
    mel_chunks = _mel_chunks_for_frames(np.asarray(wav, dtype=np.float32), len(frames), fps)

    n = min(len(frames), len(crops), len(mel_chunks))
    frames, crops, mel_chunks = frames[:n], crops[:n], mel_chunks[:n]

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

    return np.stack(outputs, axis=0)
