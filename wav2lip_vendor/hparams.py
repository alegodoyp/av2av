# Vendored from https://github.com/Rudrabha/Wav2Lip (hparams.py), for
# research/non-commercial use per that repo's license terms. These exact
# values must match what wav2lip_gan.pth was trained with.

class HParams:
    def __init__(self, **kwargs):
        self.data = {}

        for key, value in kwargs.items():
            self.data[key] = value

    def __getattr__(self, key):
        if key not in self.data:
            raise AttributeError("'HParams' object has no attribute %s" % key)
        return self.data[key]

    def set_hparam(self, key, value):
        self.data[key] = value


# Default hyperparameters
hparams = HParams(
    num_mels=80,  # Number of mel-spectrogram channels and local conditioning dimensionality

    use_lws=False,

    n_fft=800,  # Extra window size is filled with 0 paddings to match this parameter
    hop_size=200,  # For 16000Hz, 200 = 12.5 ms (0.0125 * sample_rate)
    win_size=800,  # For 16000Hz, 800 = 50 ms (If None, win_size = n_fft) (0.05 * sample_rate)
    sample_rate=16000,  # 16000Hz (corresponding to librispeech) (sox --i <filename>)

    frame_shift_ms=None,  # Can replace hop_size parameter. (Recommended: 12.5)

    # Mel and Linear spectrograms normalization/scaling and clipping
    signal_normalization=True,
    allow_clipping_in_normalization=True,  # Only relevant if mel_normalization = True
    symmetric_mels=True,
    max_abs_value=4.,

    preemphasize=True,  # whether to apply filter
    preemphasis=0.97,  # filter coefficient.

    # Limits
    min_level_db=-100,
    ref_level_db=20,
    fmin=55,
    fmax=7600,

    ###################### Wav2Lip's own training parameters #################################
    img_size=96,
    fps=25,
)
