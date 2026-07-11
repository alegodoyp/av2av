"""GFPGAN-based face restoration.

The unit2av FaceRenderer only ever generates faces at a fixed 96x96
resolution (see unit2av/model.py), which is then upscaled to paste back
into the source frame. This pass runs the pasted frame through GFPGAN to
recover detail the renderer's low output resolution can't provide.
"""

def load_face_restorer(use_cuda=True):
    try:
        from gfpgan import GFPGANer
    except ImportError as e:
        print(f"Warning: could not import gfpgan ({e}). Skipping face restoration.")
        return None

    device = "cuda" if use_cuda else "cpu"
    model_url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
    try:
        return GFPGANer(
            model_path=model_url,
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
            device=device,
        )
    except Exception as e:
        print(f"Warning: failed to load GFPGAN ({e}). Skipping face restoration.")
        return None


def restore_frame(restorer, frame_bgr, weight=0.8):
    # weight balances fidelity-to-input vs. GFPGAN's generative prior: higher
    # stays closer to the source, lower leans harder into GFPGAN's own
    # detail synthesis. The mouth/teeth region here is already a GAN
    # hallucination from the unit2av renderer, so a low weight makes GFPGAN
    # hallucinate a *second* time on top of that (invented, mismatched
    # teeth). Bias toward fidelity so GFPGAN mainly sharpens/denoises
    # instead of re-inventing mouth structure.
    if restorer is None:
        return frame_bgr
    try:
        _, _, restored = restorer.enhance(
            frame_bgr, has_aligned=False, only_center_face=True, paste_back=True,
            weight=weight,
        )
        return restored if restored is not None else frame_bgr
    except Exception as e:
        print(f"Warning: face restoration failed on a frame ({e}). Using unrestored frame.")
        return frame_bgr
