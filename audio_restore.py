"""Speech restoration for unit2av's synthesized voice, analogous to
face_restore.py's GFPGAN pass for video.

unit2av's vocoder (see unit2av/config.json: sampling_rate=16000, fmax=8000)
has a hard architectural ceiling -- it was never trained to produce any
content above 8kHz, which is exactly what reads as "muffled" (missing the
sibilant/fricative detail natural full-bandwidth speech has), plus whatever
audible artifacts a zero-shot speaker-embedding-conditioned vocoder
introduces trying to clone an unseen voice ("noisy"). VoiceFixer
(https://github.com/haoheliu/voicefixer) restores degraded speech -- noise,
low bandwidth (2kHz-44.1kHz), clipping -- to full-bandwidth 44.1kHz audio in
one pass.
"""
import os


def load_voice_restorer(use_cuda=True):
    try:
        from voicefixer import VoiceFixer
    except ImportError as e:
        print(f"Warning: could not import voicefixer ({e}). Skipping audio restoration.")
        return None
    try:
        return VoiceFixer()
    except Exception as e:
        print(f"Warning: failed to load VoiceFixer ({e}). Skipping audio restoration.")
        return None


def restore_audio_file(restorer, in_path, out_path, use_cuda=True, mode=0):
    """Restores in_path (any rate VoiceFixer supports) to a 44.1kHz
    full-bandwidth file at out_path. Returns out_path on success, or
    in_path unchanged if restoration wasn't available/failed (caller should
    treat the returned path's actual sample rate as authoritative -- read
    it back rather than assuming 44100).
    """
    if restorer is None:
        return in_path
    try:
        restorer.restore(input=in_path, output=out_path, cuda=use_cuda, mode=mode)
        return out_path
    except Exception as e:
        print(f"Warning: audio restoration failed ({e}). Using unrestored audio.")
        return in_path
