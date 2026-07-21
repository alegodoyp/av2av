import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import wave
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import pickle
import imageio
import face_alignment
import webrtcvad

from fairseq import utils
from fairseq_cli.generate import get_symbols_to_strip_from_output

from av2unit.inference import load_model as load_av2unit_model
from unit2unit.inference import load_model as load_unit2unit_model
from unit2av.inference import load_model as load_unit2av_model, load_speaker_encoder_model

from util import extract_audio_from_video, save_video, get_audio_duration, save_audio, load_audio
from face_restore import load_face_restorer
from audio_restore import load_voice_restorer, restore_audio_file
from wav2lip_render import load_wav2lip_model, render_video as render_wav2lip_video
from latentsync_render import render_video_latentsync

def _debug_check_repeats(label, unit_str, ngram_size=8):
    # Ground-truth check for repeated content at the token level, since
    # audio/spectrogram-based inspection turned out to be unreliable (normal
    # speech's regular syllable-stress cadence can look "periodic" without
    # any content actually repeating, and phase-cancellation found no exact
    # repeated audio segment either). This finds any exact repeated n-gram
    # directly in the unit sequence, with no guessing involved.
    if not unit_str:
        print(f"DEBUG repeat-check [{label}]: empty")
        return
    tokens = unit_str.strip().split()
    print(f"DEBUG repeat-check [{label}]: {len(tokens)} tokens")

    positions = {}
    for i in range(len(tokens) - ngram_size + 1):
        ngram = tuple(tokens[i:i + ngram_size])
        positions.setdefault(ngram, []).append(i)

    repeated = {ng: pos for ng, pos in positions.items() if len(pos) > 1}
    if not repeated:
        print(f"DEBUG repeat-check [{label}]: no repeated {ngram_size}-gram found")
        return

    worst_ngram, worst_positions = max(repeated.items(), key=lambda kv: len(kv[1]))
    print(
        f"DEBUG repeat-check [{label}]: FOUND a {ngram_size}-gram repeated "
        f"{len(worst_positions)}x at token positions {worst_positions}: {' '.join(worst_ngram)}"
    )

def _dedupe_repeated_tail(unit_str, ngram_size=8, min_repeats=3):
    # Safety net for unit2unit's decoder degenerating into a repetition loop
    # (confirmed on video3: an 8-gram repeated 15x at ~55-token intervals,
    # despite cfg.generation.no_repeat_ngram_size=3 being set -- whatever the
    # exact reason that constraint isn't preventing it, this truncates the
    # sequence at the first sign of looping rather than depending on
    # generation-time blocking alone). min_repeats=3 (not 2) so a phrase
    # that's legitimately repeated once or twice for emphasis isn't cut.
    tokens = unit_str.strip().split()
    positions = {}
    for i in range(len(tokens) - ngram_size + 1):
        ngram = tuple(tokens[i:i + ngram_size])
        positions.setdefault(ngram, []).append(i)

    repeated = {ng: pos for ng, pos in positions.items() if len(pos) >= min_repeats}
    if not repeated:
        return unit_str, False

    cutoff = min(pos[1] for pos in repeated.values())
    print(f"DEBUG dedupe: truncating tgt_unit at token {cutoff}/{len(tokens)} (repetition loop detected)")
    return " ".join(tokens[:cutoff]), True

def _reduce_with_run_lengths(units):
    # Like util.process_units(reduce=True), but also returns how many raw
    # tokens each reduced token collapsed from. A run length well above 1
    # means the source held the same unit for a while -- a sustained
    # sound/breath -- which is a reasonable proxy for a natural pause between
    # phrases, used below to pick better chunk-split points than an arbitrary
    # fixed position.
    reduced, run_lengths = [], []
    for u in units:
        if reduced and reduced[-1] == u:
            run_lengths[-1] += 1
        else:
            reduced.append(u)
            run_lengths.append(1)
    return reduced, run_lengths

def _find_pause_split_points(run_lengths, num_chunks, search_frac=0.2):
    # Chooses num_chunks-1 split positions near evenly-spaced targets, each
    # snapped to the most pause-like (highest run-length) position within a
    # local window, instead of cutting at the exact fixed target -- avoids
    # splitting mid-sentence, which starves the second chunk of context and
    # was confirmed (see conversation) to make translations incoherent past
    # the split, even though it fixed the earlier repetition-loop bug.
    total_len = len(run_lengths)
    ideal_size = total_len / num_chunks
    window = max(1, int(ideal_size * search_frac))

    splits = []
    prev_split = 0
    for i in range(1, num_chunks):
        target = int(i * ideal_size)
        lo = max(prev_split + 1, target - window)
        hi = min(total_len - 1, target + window)
        if lo >= hi:
            split = target
        else:
            split = max(range(lo, hi), key=lambda p: run_lengths[p])
        splits.append(split)
        prev_split = split
    return splits

def _measure_pause_ratio(wav_path, frame_ms=30, aggressiveness=2):
    # Fraction of the source audio classified as non-speech by webrtcvad --
    # a continuous/fast talker has few silent frames, a slower speaker with
    # natural breathing pauses between phrases has many. Used as a proxy for
    # how much "catch-up" pause time the translation should be allowed to
    # add, instead of always forcing the translation to fill the source's
    # exact duration regardless of how it was paced (see _source_pace_factor).
    # Requires mono 16-bit PCM, which is what extract_audio_from_video writes.
    vad = webrtcvad.Vad(aggressiveness)
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        assert wf.getnchannels() == 1 and wf.getsampwidth() == 2, \
            f"webrtcvad needs mono 16-bit PCM, got {wf.getnchannels()}ch/{wf.getsampwidth()*8}bit"
        pcm = wf.readframes(wf.getnframes())

    frame_bytes = int(sr * frame_ms / 1000) * 2
    n_frames = len(pcm) // frame_bytes
    if n_frames == 0:
        return 0.0
    silent = sum(
        1 for i in range(n_frames)
        if not vad.is_speech(pcm[i * frame_bytes:(i + 1) * frame_bytes], sr)
    )
    return silent / n_frames

def _source_pace_factor(pause_ratio, low=0.15, high=0.45):
    # Maps a source pause ratio to how much of the gap between the source
    # video's duration and the (naturally-paced) translated audio's duration
    # we're willing to fill with added pause time: 0 for a continuous/fast
    # talker (keep the translation at its own natural pace, even if that
    # ends up shorter than the source clip), 1 for a slow/pausy talker (fill
    # the whole gap so the translated clip runs the source's full length).
    # low/high are a first-pass heuristic -- no calibration data yet, so
    # these may need tuning once tried against real videos.
    return float(np.clip((pause_ratio - low) / (high - low), 0.0, 1.0))

def _pad_with_silence(wav, video, full_video, bbox, target_duration_sec, sr, fps=25):
    # Adds genuine trailing silence (and correspondingly frozen final video
    # frames) instead of stretching phoneme durations to fill extra time --
    # tried extending code durations to simulate pauses (see unit2av/model.py
    # history) and it garbled real speech content, because there's no
    # reliable way to tell a "pause" code from an ordinary long-held phoneme
    # from duration alone. Appending real silence after generation can't
    # distort a single word, since it only ever extends the clip past where
    # the actual (untouched) speech already ended.
    current_sec = len(wav) / sr
    gap_sec = target_duration_sec - current_sec
    if gap_sec <= 0:
        return wav, video, full_video, bbox

    n_samples = int(round(gap_sec * sr))
    n_frames = int(round(gap_sec * fps))
    print(f"DEBUG pace-matching: appending {gap_sec:.2f}s of trailing silence "
          f"({n_samples} samples / {n_frames} frames) to reach {target_duration_sec:.2f}s")

    wav = np.concatenate([wav, np.zeros(n_samples, dtype=wav.dtype)])
    if n_frames > 0:
        video = np.concatenate([video, np.repeat(video[-1:], n_frames, axis=0)])
        full_video = np.concatenate([full_video, np.repeat(full_video[-1:], n_frames, axis=0)])
        bbox = np.concatenate([bbox, np.repeat(bbox[-1:], n_frames, axis=0)])
    return wav, video, full_video, bbox

def extract_bbox(video_path, save_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading FaceAlignment on {device}...")
    try:
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=device)
    except AttributeError:
        # Fallback if _2D is not found (some versions use TWO_D or just string)
        try:
             fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device=device)
        except:
             # Try passing int 2
             fa = face_alignment.FaceAlignment(2, flip_input=False, device=device)

    print(f"Extracting bboxes from {video_path}...")
    reader = imageio.get_reader(video_path)
    bboxes = []
    
    for i, frame in enumerate(reader):
        try:
            preds = fa.get_landmarks(frame)
        except Exception as e:
            preds = None
            
        if preds is not None and len(preds) > 0:
            lm = preds[0]
            x_min, y_min = np.min(lm, axis=0)
            x_max, y_max = np.max(lm, axis=0)
            bboxes.append([x_min, y_min, x_max, y_max])
        else:
            bboxes.append(None)
    
    reader.close()
    
    with open(save_path, 'wb') as f:
        pickle.dump(bboxes, f)
    print(f"Saved bboxes to {save_path}")


def inference_av2unit(task, model, sample):
    # Adapted from AVHubertUnitPretrainingTask.inference
    x, padding_mask = model.extract_finetune(**sample)

    label_embs_list = model.label_embs_concat.split(model.num_classes, 0)
    proj_x = model.final_proj(x)
    if model.untie_final_proj:
        proj_x_list = proj_x.chunk(len(model.num_classes), dim=-1)
    else:
        proj_x_list = [proj_x for _ in range(len(model.num_classes))]
        
    logit_list = [model.compute_logits(proj, emb).view(-1, num_class) for proj, emb, num_class in zip(proj_x_list, label_embs_list, model.num_classes)]

    pred_even = logit_list[0].argmax(dim=-1).cpu()
    if len(logit_list) > 1:
        pred_odd = logit_list[1].argmax(dim=-1).cpu()
        pred = torch.stack([pred_even, pred_odd]).transpose(0,1).reshape(-1)
    else:
        pred = pred_even

    return pred

def inference_av2unit_chunked(task, model, sample, chunk_size=500):
    # Chunked inference to avoid OOM on long videos
    # audio: (B, T_audio, C), video: (B, T_video, C)
    # T_audio ~ 4 * T_video
    
    audio = sample['source']['audio']
    video = sample['source']['video']
    
    # Video: (B, T, H, W, C)
    B, T_video, H, W, C_video = video.shape
    # Audio: (B, T_audio)
    
    # Check if chunking is needed
    if T_video <= chunk_size:
        return inference_av2unit(task, model, sample)
        
    print(f"Video length {T_video} > {chunk_size}. Using chunked inference...")
    
    preds = []
    
    # Iterate over video frames
    for i in range(0, T_video, chunk_size):
        end = min(i + chunk_size, T_video)
        
        # Audio alignment: assume proportional length
        if audio.ndim == 2:
            _, T_total_audio = audio.shape
        else:
             # handle (B, T, C) if that's the case
            _, T_total_audio, _ = audio.shape

        ratio = T_total_audio / T_video
        a_start = int(i * ratio)
        a_end = int(end * ratio)
        a_end = min(a_end, T_total_audio)
        
        chunk_video = video[:, i:end, :, :, :]
        if audio.ndim == 2:
            chunk_audio = audio[:, a_start:a_end]
        else:
            chunk_audio = audio[:, a_start:a_end, :]
        
        chunk_sample = {
            "source": {
                "audio": chunk_video.new(chunk_audio), 
                "video": chunk_video
            }
        }
        
        # We need to handle padding mask if it exists? 
        # In this pipeline, we collate with len, so explicit padding mask might be generated inside extract_finetune if not provided.
        # But here we are passing raw tensors in 'source'.
        
        with torch.no_grad():
            # inference_av2unit calls extract_finetune, then projects
            # We call it on the chunk
            chunk_pred = inference_av2unit(task, model, chunk_sample)
            preds.append(chunk_pred)
            
    # Concatenate predictions
    full_pred = torch.cat(preds, dim=0) # shape (T_total,)
    
    return full_pred

class AVSpeechToAVSpeechPipeline:
    def __init__(self,
        av2unit_model, av2unit_task,
        unit2unit_task, unit2unit_generator,
        unit2av_model, speaker_encoder,
        use_cuda=False
    ):
        self.av2unit_model = av2unit_model
        self.av2unit_task = av2unit_task
        self.unit2unit_task = unit2unit_task
        self.unit2unit_generator = unit2unit_generator
        self.unit2av_model = unit2av_model
        self.speaker_encoder = speaker_encoder
        self.use_cuda = use_cuda



    def _to_device(self, obj, device):
        if hasattr(obj, 'models'): # SequenceGenerator
            for model in obj.models:
                model.to(device)
        elif isinstance(obj, torch.nn.Module):
             obj.to(device)
        # Add handling for other types if needed

    def _to_gpu(self, obj):
        if self.use_cuda:
            self._to_device(obj, 'cuda')

    def _to_cpu(self, obj):
        if self.use_cuda:
            self._to_device(obj, 'cpu')
            torch.cuda.empty_cache()

    def process_av2unit(self, lip_video_path, audio_path):
        task = self.av2unit_task
        # Append dummy ID because hubert_dataset expects path:id format and splits on colon
        # And we patched hubert_dataset to use rsplit, so this structure is required on Windows.
        audio_path_with_id = f"{audio_path}:dummy_id"
        video_feats, audio_feats = task.dataset.load_feature((lip_video_path, audio_path_with_id))

        if video_feats is None or audio_feats is None:
            print(f"Error: Failed to load features for {lip_video_path}")
            return None

        audio_feats, video_feats = torch.from_numpy(audio_feats.astype(np.float32)), torch.from_numpy(video_feats.astype(np.float32))
        
        if task.dataset.normalize and 'audio' in task.dataset.modalities:
            with torch.no_grad():
                audio_feats = F.layer_norm(audio_feats, audio_feats.shape[1:])

        collated_audios, _, _ = task.dataset.collater_audio([audio_feats], len(audio_feats))
        collated_videos, _, _ = task.dataset.collater_audio([video_feats], len(video_feats))

        sample = {"source": {
            "audio": collated_audios, "video": collated_videos,
        }}
        sample = utils.move_to_cuda(sample) if self.use_cuda else sample

        with torch.cuda.amp.autocast():
            pred = inference_av2unit_chunked(task, self.av2unit_model, sample, chunk_size=4000)


        pred_str = task.dictionaries[0].string(pred.int().cpu())

        return pred_str

    # Threshold on the *reduced* (post run-length-collapse) source token
    # count. Confirmed via direct comparison (see conversation): video1's
    # reduced_len=674 translates cleanly on both the pretrained and
    # fine-tuned checkpoints; video3's reduced_len=690 degenerates into a
    # beam-search repetition loop on *both* checkpoints (so it's not a
    # checkpoint-specific issue). The gap between 674 and 690 is too small to
    # be a hard length cliff -- it's some property of that content the
    # decoder can't sustain coherent generation over -- but since we can't
    # predict which content will trigger it, this keeps a comfortable margin
    # below the smallest confirmed failure for any input, by translating in
    # independent chunks instead of trusting one long decode to stay coherent.
    UNIT2UNIT_CHUNK_SIZE = 500
    # Even after the initial pause-aligned split, a specific chunk can still
    # degenerate (confirmed on video3: chunk 2, 385 tokens, truncated from a
    # natural 329 down to 151 on *both* checkpoints -- this content is
    # genuinely hard for the model, not a fluke of one checkpoint or one
    # split point). Rather than accept the lost tail (which took the video's
    # final sentence with it), a truncated chunk gets re-split and retried.
    MIN_SUBCHUNK_SIZE = 120
    MAX_SPLIT_DEPTH = 3

    def process_unit2unit(self, unit):
        unit = list(map(int, unit.strip().split()))
        raw_len = len(unit)
        reduced, run_lengths = _reduce_with_run_lengths(unit)
        print(f"DEBUG unit2unit input: raw_len={raw_len}, reduced_len={len(reduced)} (encoder sees reduced_len+2)")
        return self._translate_reduced(reduced, run_lengths)

    def _translate_reduced(self, reduced, run_lengths, depth=0):
        if len(reduced) <= self.UNIT2UNIT_CHUNK_SIZE:
            translated, was_truncated = self._translate_unit_chunk(reduced)
            if not was_truncated or len(reduced) <= self.MIN_SUBCHUNK_SIZE or depth >= self.MAX_SPLIT_DEPTH:
                return translated
            print(f"DEBUG unit2unit: chunk truncated at depth {depth} ({len(reduced)} tokens) -- "
                  f"retrying as smaller sub-chunks")
            num_chunks = 2
        else:
            num_chunks = -(-len(reduced) // self.UNIT2UNIT_CHUNK_SIZE)  # ceil div

        split_points = _find_pause_split_points(run_lengths, num_chunks)
        bounds = [0] + split_points + [len(reduced)]
        print(f"DEBUG unit2unit: depth={depth}, splitting {len(reduced)} tokens into {num_chunks} "
              f"chunk(s) at pause-aligned points {split_points}")

        translated_chunks = []
        for i in range(len(bounds) - 1):
            lo, hi = bounds[i], bounds[i + 1]
            print(f"DEBUG unit2unit: translating chunk {i + 1}/{num_chunks} ({hi - lo} tokens) at depth {depth}")
            translated_chunks.append(
                self._translate_reduced(reduced[lo:hi], run_lengths[lo:hi], depth + 1)
            )

        return " ".join(translated_chunks)

    def _translate_unit_chunk(self, reduced_units):
        task = self.unit2unit_task
        unit = task.source_dictionary.encode_line(
            " ".join(map(str, reduced_units)),
            add_if_not_exist=False,
            append_eos=True,
        ).long()
        unit = torch.cat([
            unit.new([task.source_dictionary.bos()]),
            unit,
            unit.new([task.source_dictionary.index("[{}]".format(task.source_language))])
        ])

        sample = {"net_input": {
            "src_tokens": torch.LongTensor(unit).view(1,-1),
        }}

        print(f"DEBUG unit2unit: max_token={unit.max().item()}, min_token={unit.min().item()}, vocab_size={len(task.source_dictionary)}")
        if unit.max().item() >= len(task.source_dictionary):
            print(f"WARNING: Token index {unit.max().item()} exceeds vocabulary size {len(task.source_dictionary)}!")
            # Truncating or clipping off-vocab to unk to prevent CUDA crash
            unit[unit >= len(task.source_dictionary)] = task.source_dictionary.unk()
            sample["net_input"]["src_tokens"] = torch.LongTensor(unit).view(1,-1)

        sample = utils.move_to_cuda(sample) if self.use_cuda else sample

        with torch.cuda.amp.autocast():
            pred = task.inference_step(
                self.unit2unit_generator,
                None,
                sample,
            )[0][0]

        pred_str = task.target_dictionary.string(
            pred["tokens"].int().cpu(),
            extra_symbols_to_ignore=get_symbols_to_strip_from_output(self.unit2unit_generator)
        )
        return _dedupe_repeated_tail(pred_str)

    def process_unit2av(self, unit, audio_path, video_path, bbox_path, speaker_embed_scale=1.0):
        unit = list(map(int, unit.strip().split()))

        # Filter out special tokens or language tokens that exceed the unit vocabulary (0-1999)
        # unit2av model's embedding size is exactly 2000.
        unit = [u for u in unit if u < 2000]

        spkr_embed = self.speaker_encoder.get_embed(audio_path)
        if speaker_embed_scale != 1.0:
            # Experimental: the embedding is L2-normalized before this, so
            # scaling its magnitude is a cheap way to test whether the
            # vocoder's speaker-conditioning is under-weighted relative to
            # content -- unvalidated, may help or hurt, hence opt-in only.
            spkr_embed = spkr_embed * speaker_embed_scale

        sample = {
            "code": torch.LongTensor(unit).view(1,-1),
            "spkr": torch.from_numpy(spkr_embed).view(1,1,-1),
        }
        sample = utils.move_to_cuda(sample) if self.use_cuda else sample

        # No tgt_dur forced here -- the vocoder generates at its own natural
        # predicted pace. Matching the source video's duration (when
        # warranted) is handled afterwards in main() by appending real
        # trailing silence rather than distorting phoneme durations, gated by
        # how slowly/pausily the *source* speaker actually talks (see
        # _measure_pause_ratio/_source_pace_factor/_pad_with_silence).
        with torch.cuda.amp.autocast():
            wav, video, full_video, bbox = self.unit2av_model(sample, video_path, bbox_path, dur_prediction=True)


        return wav, video, full_video, bbox

def main(args):
    use_cuda = torch.cuda.is_available() and not args.cpu
    
    # Generate bbox if needed BEFORE loading large models to avoid OOM or long waits
    bbox_path = os.path.splitext(args.in_vid_path)[0]+".bbox.pkl"
    if not os.path.exists(bbox_path):
        print(f"Generating bbox for {args.in_vid_path}...")
        extract_bbox(args.in_vid_path, bbox_path)
    
    temp_audio_path = os.path.splitext(args.in_vid_path)[0]+".temp.wav"
    extract_audio_from_video(args.in_vid_path, temp_audio_path)

    # Load models directly to GPU to maximize inference speed
    # We remove use_cuda=False overrides since we have plenty of VRAM
    av2unit_model, av2unit_task = load_av2unit_model(args.av2unit_path, args.modalities, use_cuda=use_cuda)
    unit2unit_task, unit2unit_generator = load_unit2unit_model(args.utut_path, args.src_lang, args.tgt_lang, use_cuda=use_cuda)
    cfg_path = os.path.join("unit2av", "config.json")
    unit2av_model = load_unit2av_model(args.unit2av_path, cfg_path, args.tgt_lang, use_cuda=use_cuda, fp16=True)
    speaker_encoder_model = load_speaker_encoder_model(os.path.join("unit2av", "encoder.pt"), use_cuda=use_cuda)
    # GFPGAN restoration and Wav2Lip both feed into util.save_video(); LatentSync
    # produces a finished, audio-muxed video on its own and skips both.
    face_restorer = None
    if args.video_renderer != "latentsync" and not args.no_face_restore:
        face_restorer = load_face_restorer(use_cuda=use_cuda)
    wav2lip_model = None
    if args.video_renderer == "wav2lip":
        wav2lip_model = load_wav2lip_model(args.wav2lip_checkpoint, use_cuda=use_cuda)
    voice_restorer = None if args.no_audio_restore else load_voice_restorer(use_cuda=use_cuda)

    pipeline = AVSpeechToAVSpeechPipeline(
        av2unit_model, av2unit_task,
        unit2unit_task, unit2unit_generator,
        unit2av_model, speaker_encoder_model,
        use_cuda=use_cuda
    )

    lip_video_path = os.path.splitext(args.in_vid_path)[0]+".lip.mp4"
    if not os.path.exists(lip_video_path):
        print(f"Warning: {lip_video_path} not found. Using raw video {args.in_vid_path} instead.")
        lip_video_path = args.in_vid_path

    src_unit = pipeline.process_av2unit(lip_video_path, temp_audio_path)
    if src_unit is None:
        print(f"Error: Inference failed for {args.in_vid_path}")
        if os.path.exists(temp_audio_path):
             os.remove(temp_audio_path)
        sys.exit(1)

    tgt_unit = pipeline.process_unit2unit(src_unit)

    _debug_check_repeats("src_unit", src_unit)
    _debug_check_repeats("tgt_unit", tgt_unit)

    # Used later to decide how much (if any) trailing pause to add so the
    # translated clip's length tracks the source video's -- see the
    # pace-matching block below, after tgt_audio is generated.
    src_duration_sec = get_audio_duration(temp_audio_path)

    # SpeakerEncoder.preprocess_wav only normalizes volume and trims long
    # silences (VAD) -- it never actually denoises the speech itself. If the
    # source video's audio has background noise/reverb/compression
    # artifacts, the extracted embedding partly captures those rather than
    # just the speaker's voice. Reuse the same VoiceFixer pass that already
    # measurably helped the *output* audio, but on the *reference* clip the
    # embedding is extracted from -- untried previously, since audio_restore
    # was only ever applied after synthesis.
    speaker_ref_path = temp_audio_path
    if voice_restorer is not None:
        cleaned_ref_path = os.path.splitext(args.in_vid_path)[0] + ".speaker_ref.wav"
        result_path = restore_audio_file(voice_restorer, temp_audio_path, cleaned_ref_path, use_cuda=use_cuda)
        if result_path == cleaned_ref_path:
            speaker_ref_path = cleaned_ref_path

    tgt_audio, tgt_video, full_video, bbox = pipeline.process_unit2av(
        tgt_unit, speaker_ref_path, args.in_vid_path, bbox_path,
        speaker_embed_scale=args.speaker_embed_scale,
    )

    if speaker_ref_path != temp_audio_path and os.path.exists(speaker_ref_path):
        os.remove(speaker_ref_path)
    tgt_sr = 16000  # unit2av's CodeHiFiGANModel_spk native rate (config.json)

    if voice_restorer is not None:
        # unit2av's vocoder is architecturally capped at 16kHz/8kHz-fmax (see
        # unit2av/config.json) -- it was never trained to produce anything
        # above 8kHz, which is what reads as "muffled", plus whatever GAN
        # artifacts a zero-shot speaker-conditioned vocoder introduces
        # ("noisy"). VoiceFixer restores this to full-bandwidth 44.1kHz.
        presynth_path = os.path.splitext(args.out_vid_path)[0] + ".presynth.wav"
        restored_path = os.path.splitext(args.out_vid_path)[0] + ".restored.wav"
        save_audio(tgt_audio, presynth_path, sampling_rate=tgt_sr)
        result_path = restore_audio_file(voice_restorer, presynth_path, restored_path, use_cuda=use_cuda)
        if result_path == restored_path:
            tgt_audio, tgt_sr = load_audio(restored_path)
            os.remove(restored_path)
        os.remove(presynth_path)

    # The translation was generated at its own natural pace above (no forced
    # duration target), so it commonly ends up shorter than the source clip.
    # Rather than always padding the gap out to an exact match (which would
    # force a fast/continuous talker's translation to run needlessly long),
    # gate how much gets filled by how slowly/pausily the *source* speaker
    # actually talks -- a slow, pause-heavy source earns a longer trailing
    # pause to match; a fast, continuous source keeps its natural (shorter)
    # length instead of being padded out.
    gap_sec = src_duration_sec - (len(tgt_audio) / tgt_sr)
    if gap_sec > 0:
        pause_ratio = _measure_pause_ratio(temp_audio_path)
        pace_factor = _source_pace_factor(pause_ratio)
        target_duration_sec = (len(tgt_audio) / tgt_sr) + pace_factor * gap_sec
        print(f"DEBUG source pace: gap={gap_sec:.2f}s, pause_ratio={pause_ratio:.3f} -> "
              f"pace_factor={pace_factor:.3f}, target_duration={target_duration_sec:.2f}s "
              f"(source was {src_duration_sec:.2f}s)")
        tgt_audio, tgt_video, full_video, bbox = _pad_with_silence(
            tgt_audio, tgt_video, full_video, bbox, target_duration_sec, sr=tgt_sr,
        )

    if args.video_renderer == "latentsync":
        # LatentSync does its own face detection/alignment/diffusion sampling/
        # paste-back as a single black-box pipeline (run in its own conda env
        # via subprocess) and writes a finished, audio-muxed video directly --
        # unlike unit2av/wav2lip, there's no patch to feed through save_video().
        render_video_latentsync(
            tgt_audio, full_video, args.out_vid_path,
            repo_dir=args.latentsync_repo,
            python_bin=args.latentsync_python,
            bbox_path=bbox_path,
            unet_config_path=args.latentsync_config,
            ckpt_path=args.latentsync_ckpt,
            sampling_rate=tgt_sr,
        )
    else:
        if wav2lip_model is not None:
            # Replace unit2av's own (zero-shot, unit-conditioned) generated
            # patches with Wav2Lip's official checkpoint, driven by the same
            # synthesized audio. render_video pads/smooths `bbox` to match
            # Wav2Lip's own crop conventions, so we must paste back with
            # those same boxes (returned here), not the original ones.
            tgt_video, bbox = render_wav2lip_video(
                wav2lip_model, tgt_audio, full_video, bbox, fps=25, use_cuda=use_cuda,
                sampling_rate=tgt_sr,
            )

        save_video(tgt_audio, tgt_video, full_video, bbox, args.out_vid_path, restorer=face_restorer, sampling_rate=tgt_sr)

    os.remove(temp_audio_path)

def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-vid-path", type=str, required=True, help="File path of source video input"
    )
    parser.add_argument(
        "--out-vid-path", type=str, required=True, help="File path of translated video output"
    )
    parser.add_argument(
        "--src-lang", type=str, required=True,
        choices=["en","es","fr","it","pt"],
        help="source language"
    )
    parser.add_argument(
        "--tgt-lang", type=str, required=True,
        choices=["en","es","fr","it","pt"],
        help="target language"
    )
    parser.add_argument(
        "--modalities", type=str, default="audio,video", help="input modalities",
        choices=["audio,video","audio","video"],
    )
    parser.add_argument(
        "--av2unit-path", type=str, required=True, help="path to the mAV-HuBERT pre-trained model"
    )
    parser.add_argument(
        "--utut-path", type=str, required=True, help="path to the UTUT pre-trained model"
    )
    parser.add_argument(
        "--unit2av-path", type=str, required=True, help="path to the Unit AV Renderer"
    )
    parser.add_argument("--cpu", action="store_true", help="run on CPU")
    parser.add_argument(
        "--no-face-restore", action="store_true",
        help="Disable GFPGAN face restoration post-process (enabled by default)"
    )
    parser.add_argument(
        "--no-audio-restore", action="store_true",
        help="Disable VoiceFixer audio restoration post-process (enabled by default). "
             "unit2av's vocoder is capped at 16kHz/8kHz-fmax; this restores full-bandwidth "
             "44.1kHz audio and reduces vocoder noise artifacts."
    )
    parser.add_argument(
        "--speaker-embed-scale", type=float, default=1.0,
        help="Experimental: multiplies the (L2-normalized) speaker embedding's magnitude "
             "before it's fed to the vocoder, to test whether speaker-conditioning is "
             "under-weighted relative to content. 1.0 = unchanged. Unvalidated -- may help "
             "or hurt voice cloning fidelity; try values like 1.3-1.5 to compare."
    )
    parser.add_argument(
        "--video-renderer", type=str, default="unit2av",
        choices=["unit2av", "wav2lip", "latentsync"],
        help="'unit2av' uses this repo's own zero-shot renderer (default). "
             "'wav2lip' drives Wav2Lip's official wav2lip_gan.pth from the "
             "same synthesized audio instead. 'latentsync' drives ByteDance's "
             "LatentSync diffusion renderer (needs its own conda env, see "
             "--latentsync-python)."
    )
    parser.add_argument(
        "--wav2lip-checkpoint", type=str, default="checkpoints/wav2lip_gan.pth",
        help="path to Wav2Lip's official checkpoint (only used with --video-renderer wav2lip)"
    )
    parser.add_argument(
        "--latentsync-repo", type=str, default="latentsync_repo",
        help="path to a local clone of bytedance/LatentSync (only used with --video-renderer latentsync)"
    )
    parser.add_argument(
        "--latentsync-python", type=str, default=None,
        help="path to the LatentSync repo's own conda env python executable, "
             "e.g. ~/miniconda3/envs/latentsync/bin/python (required with --video-renderer latentsync)"
    )
    parser.add_argument(
        "--latentsync-config", type=str, default="configs/unet/stage2_512.yaml",
        help="LatentSync UNet config, relative to --latentsync-repo (1.6/512px by default)"
    )
    parser.add_argument(
        "--latentsync-ckpt", type=str, default="checkpoints/latentsync_unet.pt",
        help="LatentSync checkpoint path, relative to --latentsync-repo"
    )
    args = parser.parse_args()
    if args.video_renderer == "latentsync" and not args.latentsync_python:
        parser.error("--latentsync-python is required when --video-renderer latentsync")
    main(args)

if __name__ == "__main__":
    cli_main()
