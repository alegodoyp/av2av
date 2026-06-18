
import os
import argparse
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import shutil
import subprocess

# Add project root to sys.path
sys.path.append(os.getcwd())

from fairseq import checkpoint_utils, utils
from util import process_units, extract_audio_from_video

# Register AV-HuBERT tasks (including av_hubert_unit_pretraining)
try:
    from av2unit.task import AVHubertUnitPretrainingTask
except ImportError:
    print("Warning: Could not import av2unit.task. Task 'av_hubert_unit_pretraining' might not be registered.")
except Exception as e:
    print(f"Warning: Error importing AV-HuBERT task: {e}")

import tempfile

def load_av2unit_model(model_path, modalities="audio,video", use_cuda=True):
    # AV-HuBERT Base checkpoint config misses `_name` in older versions.
    # Load state directly to patch it first before checkpoint_utils crashes.
    state = torch.load(model_path, map_location="cpu")
    if 'cfg' in state:
        if 'model' in state['cfg']:
            if getattr(state['cfg']['model'], '_name', None) is None:
                if isinstance(state['cfg']['model'], dict):
                    state['cfg']['model']['_name'] = 'av_hubert'
                else:
                    state['cfg']['model']._name = 'av_hubert'
                    
            # Fix negative dimension crash in AV-HuBERT Base when loading checkpoint
            # due to missing `audio_feat_dim` in older config format.
            if isinstance(state['cfg']['model'], dict):
                if state['cfg']['model'].get('audio_feat_dim', -1) <= 0:
                    state['cfg']['model']['audio_feat_dim'] = 104
            else:
                if getattr(state['cfg']['model'], 'audio_feat_dim', -1) <= 0:
                    state['cfg']['model'].audio_feat_dim = 104
                    
        # Preserve projection weights from state dict as they cause size mismatches
        # relative to the dummy validation subset dictionaries, but we need them for accurate extraction.
        preserved_weights = {}
        if 'model' in state:
            keys_to_preserve = [k for k in state['model'].keys() if 'final_proj' in k or 'label_embs_concat' in k or 'target_glu' in k]
            for k in keys_to_preserve:
                preserved_weights[k] = state['model'][k]
                del state['model'][k]
                
        if 'task' in state['cfg']:
            # The older base checkpoint also misses 'label_rate' which Omegaconf demands
            if getattr(state['cfg']['task'], 'label_rate', None) is None:
                if isinstance(state['cfg']['task'], dict):
                    state['cfg']['task']['label_rate'] = 25
                else:
                    state['cfg']['task'].label_rate = 25
                    
        # Resave temp patched version
        import tempfile
        patched_path = os.path.join(tempfile.gettempdir(), "patched_model.pt")
        torch.save(state, patched_path)
        model_path = patched_path
        
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [model_path], 
        strict=False
    )
    
    model = models[0]
    
    # Manually re-attach preserved projection weights to guarantee exact token unit boundaries
    if hasattr(model, 'num_classes') is False:
        model.num_classes = []
    
    if 'label_embs_concat' in preserved_weights:
        model.label_embs_concat = torch.nn.Parameter(preserved_weights['label_embs_concat'])
        model.num_classes = [preserved_weights['label_embs_concat'].shape[0]]
        
    if 'final_proj.weight' in preserved_weights:
        has_bias = 'final_proj.bias' in preserved_weights
        model.final_proj = torch.nn.Linear(
            preserved_weights['final_proj.weight'].shape[1],
            preserved_weights['final_proj.weight'].shape[0],
            bias=has_bias
        )
        model.final_proj.weight = torch.nn.Parameter(preserved_weights['final_proj.weight'])
        if has_bias:
            model.final_proj.bias = torch.nn.Parameter(preserved_weights['final_proj.bias'])

    if 'target_glu.0.weight' in preserved_weights:
        has_bias = 'target_glu.0.bias' in preserved_weights
        model.target_glu = torch.nn.Sequential(
            torch.nn.Linear(
                preserved_weights['target_glu.0.weight'].shape[1],
                preserved_weights['target_glu.0.weight'].shape[0],
                bias=has_bias
            ),
            torch.nn.GLU()
        )
        model.target_glu[0].weight = torch.nn.Parameter(preserved_weights['target_glu.0.weight'])
        if has_bias:
            model.target_glu[0].bias = torch.nn.Parameter(preserved_weights['target_glu.0.bias'])

    for model in models:
        if use_cuda:
            model.cuda()
        model.prepare_for_inference_(cfg)
    
    task.cfg.modalities = modalities.split(",")
    # Disable verification to accept dummy data
    task.cfg.skip_verify = True
    
    # Disable noise augmentation (avoids looking for missing /checkpoint/ paths)
    task.cfg.noise_prob = 0.0
    task.cfg.noise_wav = None
    
    # Create temp dummy manifest to satisfy load_dataset
    dataset_dir = tempfile.mkdtemp()
    
    sample_rate = getattr(task.cfg, 'sample_rate', 16000)
    label_rate = getattr(task.cfg, 'label_rate', 25)
    max_size = getattr(task.cfg, 'max_sample_size', 2000)
    min_size = getattr(task.cfg, 'min_sample_size', 5)
    
    # Handle None or dynamically interpolated OmegaConf strings
    if not isinstance(max_size, int):
        max_size = 2000
    if not isinstance(min_size, int):
        min_size = 5
        
    sz = min(max_size - 1, 400)
    if sz <= min_size:
        sz = min_size + 1
        
    duration_sec = sz / sample_rate
    seq_len_tokens = max(1, int(duration_sec * label_rate))
    
    # Create valid.tsv
    with open(os.path.join(dataset_dir, "valid.tsv"), "w") as f:
        f.write("/\n") # Root
        f.write(f"dummy_id\tdummy.mp4\tdummy.wav\t{sz}\tdummy_extra\n")
    
    # Create valid.{label} for each label type expected by the model
    labels = task.cfg.labels if hasattr(task.cfg, 'labels') else ["ltr"]
    for label in labels:
        with open(os.path.join(dataset_dir, f"valid.{label}"), "w") as f:
            f.write("1 " * seq_len_tokens + "\n")
            
    # Override task config to look at temp dir
    task.cfg.data = dataset_dir
    task.cfg.label_dir = dataset_dir
    
    # Synchronize single_target metric to bypass hubert_dataset assertion requirements
    if hasattr(task.cfg, 'label_rate'):
        task.cfg.single_target = (task.cfg.label_rate == -1)
    elif hasattr(task.cfg, 'label_rates'):
        task.cfg.single_target = (task.cfg.label_rates[0] == -1)
    
    # Load dataset (required to initialize task.dataset utilities)
    task.load_dataset(split="valid")
    # Make the dataset accessible as task.dataset (common expectation in inference scripts)
    task.dataset = task.datasets["valid"]
    
    return models[0], task

def extract_units(model, task, video_path, use_cuda=True):
    temp_audio_path = os.path.splitext(video_path)[0] + ".temp.wav"

    try:
        extract_audio_from_video(video_path, temp_audio_path)

        task_audio_input = temp_audio_path + ":0"
        video_feats, audio_feats = task.dataset.load_feature((video_path, task_audio_input))

        if audio_feats is None or video_feats is None:
            print(f"Failed to load features for {video_path}")
            return None

        audio_feats = torch.from_numpy(audio_feats.astype(np.float32))
        video_feats = torch.from_numpy(video_feats.astype(np.float32))
        print(f"  DEBUG raw feats: audio={audio_feats.shape}, video={video_feats.shape}", flush=True)
        print(f"  DEBUG stack_order_audio={task.dataset.stack_order_audio}", flush=True)

        if task.dataset.normalize and 'audio' in task.dataset.modalities:
            with torch.no_grad():
                audio_feats = F.layer_norm(audio_feats, audio_feats.shape[1:])

        collated_audios, _, _ = task.dataset.collater_audio([audio_feats], len(audio_feats))
        collated_videos, _, _ = task.dataset.collater_audio([video_feats], len(video_feats))
        print(f"  DEBUG collated: audio={collated_audios.shape}, video={collated_videos.shape}", flush=True)

        sample = {"source": {
            "audio": collated_audios, "video": collated_videos,
        }}
        sample = utils.move_to_cuda(sample) if use_cuda else sample

        with torch.no_grad():
            src_a, src_v = sample["source"]["audio"], sample["source"]["video"]
            feat_a = model.forward_features(src_a, modality='audio')
            feat_v = model.forward_features(src_v, modality='video')
            print(f"  DEBUG forward_features: audio={feat_a.shape}, video={feat_v.shape}, fuse={model.modality_fuse}", flush=True)
            x, padding_mask = model.extract_finetune(**sample)

            label_embs_list = model.label_embs_concat.split(model.num_classes, 0)
            proj_x = model.final_proj(x)
            if model.untie_final_proj:
                proj_x_list = proj_x.chunk(len(model.num_classes), dim=-1)
            else:
                proj_x_list = [proj_x for _ in range(len(model.num_classes))]

            logit_list = [
                model.compute_logits(proj, emb).view(-1, num_class)
                for proj, emb, num_class in zip(proj_x_list, label_embs_list, model.num_classes)
            ]

            pred_even = logit_list[0].argmax(dim=-1).cpu()
            pred_odd = logit_list[1].argmax(dim=-1).cpu()
            pred = torch.stack([pred_even, pred_odd]).transpose(0, 1).reshape(-1)

        units = pred.numpy()
        reduced_units = process_units(units, reduce=True)
        pred_str = " ".join(map(str, reduced_units))

        return pred_str

    except Exception as e:
        print(f"Error extracting units for {video_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        if os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except:
                pass

def create_dictionary(dict_path, num_units=2000):
    """Creates a fairseq-compatible dictionary file."""
    with open(dict_path, 'w') as f:
        for i in range(num_units):
            # Format: <symbol> <count>
            f.write(f"{i} 1\n")

def process_batch(source_files, target_files, output_dir, av2unit_path, tgt_av2unit_path=None, dict_path=None, split='train', src_lang='src', tgt_lang='tgt'):
    """
    Process a batch of files.
    source_files: list of paths to source lang videos
    target_files: list of paths to target lang videos (aligned by index)
    output_dir: directory to write fairseq data
    av2unit_path: path to av2unit checkpoint (for source)
    tgt_av2unit_path: path to av2unit checkpoint for target (optional, defaults to av2unit_path)
    src_lang: source language code
    tgt_lang: target language code
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    use_cuda = torch.cuda.is_available()
    src_model, src_task = load_av2unit_model(av2unit_path, use_cuda=use_cuda)
    
    tgt_model, tgt_task = src_model, src_task
    if tgt_av2unit_path is not None and tgt_av2unit_path != av2unit_path:
        print(f"Loading separate target model from {tgt_av2unit_path}")
        tgt_model, tgt_task = load_av2unit_model(tgt_av2unit_path, use_cuda=use_cuda)
    
    # If dict_path not provided, create one in output_dir
    if dict_path is None:
        dict_path = output_dir / "dict.txt"
        # Heuristic: Assume 2000 units if not known. 
        # Better: check model dictionary.
        dictionary = src_task.dictionaries[0]
        # Fairseq dictionary has .save() method
        dictionary.save(str(dict_path))
    
    # Write raw unit files with language extensions
    src_out = output_dir / f"{split}.{src_lang}"
    tgt_out = output_dir / f"{split}.{tgt_lang}"
    
    n_success = 0
    n_total = len(source_files)
    with open(src_out, 'w') as f_src, open(tgt_out, 'w') as f_tgt:
        for src_vid, tgt_vid in zip(source_files, target_files):
            try:
                print(f"Processing {src_vid} -> {tgt_vid}")
                src_units = extract_units(src_model, src_task, src_vid, use_cuda)
                tgt_units = extract_units(tgt_model, tgt_task, tgt_vid, use_cuda)

                if src_units is None or tgt_units is None:
                    print(f"Skipping pair {src_vid}, {tgt_vid} due to extraction failure.")
                    continue

                f_src.write(src_units + "\n")
                f_tgt.write(tgt_units + "\n")

                # Flush to ensure data is written
                f_src.flush()
                f_tgt.flush()
                n_success += 1
            except Exception as e:
                print(f"Error processing pair {src_vid}, {tgt_vid}: {e}")
                import traceback
                traceback.print_exc()
                continue

    print(f"Unit extraction: {n_success}/{n_total} pairs succeeded.")
    if n_success == 0:
        raise RuntimeError(
            f"All {n_total} unit extractions failed — text files are empty. "
            f"Check the 'Error extracting units' messages above for the root cause."
        )

    # Run fairseq-preprocess independently for each language to create monolingual-like structure
    # Expected by MultilingualDenoisingTask: bin/lang/split.bin
    
    for lang in [src_lang, tgt_lang]:
        lang_dir = output_dir / "bin" / lang
        lang_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            sys.executable, "-m", "fairseq_cli.preprocess",
            "--only-source", # Treat as monolingual
            "--source-lang", lang,
            "--target-lang", lang, # processed as source, target ignored with only-source
            "--destdir", str(lang_dir),
            "--srcdict", str(dict_path),
            "--workers", "32"
        ]
        
        if split == 'train':
             cmd.extend(["--trainpref", str(output_dir / split)])
        else:
             cmd.extend(["--validpref", str(output_dir / split)])
        
        print(f"Running fairseq-preprocess for {lang}...")
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Rename fairseq-preprocess output to standard names expected by MultilingualDenoisingTask
            # It usually generates train.src-tgt.bin or train.src-src.bin (since we used --source-lang lang --target-lang lang)
            # We need just train.bin and train.idx
            
            # Debug: List generated files
            print(f"DEBUG: Files in {lang_dir}:")
            for f in os.listdir(lang_dir):
                print(f" - {f}")

            # Robust Renaming Strategy
            # fairseq-preprocess with --only-source might generate train.lang.bin or train.bin or train.lang-lang.bin
            # We look for any file starting with {split} and ending with .bin or .idx
            
            found_bin = None
            found_idx = None
            
            for f in os.listdir(lang_dir):
                if f.startswith(split) and f.endswith(".bin") and f != f"{split}.bin":
                    found_bin = f
                if f.startswith(split) and f.endswith(".idx") and f != f"{split}.idx":
                    found_idx = f
            
            import time
            def robust_move(src, dst, max_retries=10):
                for i in range(max_retries):
                    try:
                        if os.path.exists(dst):
                            try:
                                os.remove(dst)
                            except PermissionError:
                                # If we can't remove it, maybe we can't rename over it either,
                                # but let's try os.replace which is sometimes better
                                pass
                        
                        if os.path.exists(dst):
                             os.replace(src, dst)
                        else:
                             os.rename(src, dst)
                        return
                    except (PermissionError, OSError) as e:
                        if i < max_retries - 1:
                            print(f"File locked: {dst} (Attempt {i+1}/{max_retries}). Waiting...")
                            time.sleep(2)
                            continue
                        print(f"Failed to move {src} to {dst} after {max_retries} attempts.")
                        raise e

            if found_bin:
                print(f"Renaming {found_bin} to {split}.bin")
                src = lang_dir / found_bin
                dst = lang_dir / f"{split}.bin"
                robust_move(src, dst)
                
            if found_idx:
                print(f"Renaming {found_idx} to {split}.idx")
                src = lang_dir / found_idx
                dst = lang_dir / f"{split}.idx"
                robust_move(src, dst)

        except subprocess.CalledProcessError as e:
            print(f"fairseq-preprocess failed for {lang} with return code {e.returncode}")
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)
            raise
    
    # Copy dict.txt to bin/dict.txt (and bin/lang/dict.txt just in case)
    # MultilingualDenoisingTask might look in root bin or lang folders
    dest_dict_root = output_dir / "bin" / "dict.txt"
    if not dest_dict_root.exists():
        shutil.copy(dict_path, dest_dict_root)
        
    for lang in [src_lang, tgt_lang]:
         dest_dict_lang = output_dir / "bin" / lang / "dict.txt"
         if not dest_dict_lang.exists():
             shutil.copy(dict_path, dest_dict_lang)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-list", help="File containing list of source videos")
    parser.add_argument("--tgt-list", help="File containing list of target videos")
    parser.add_argument("--src-dir", help="Directory of source videos (if lists not provided, matches by filename)")
    parser.add_argument("--tgt-dir", help="Directory of target videos")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--av2unit-path", required=True)
    parser.add_argument("--dict-path", help="Path to fixed dictionary")
    args = parser.parse_args()
    
    # Simple logic to gather files
    src_files = []
    tgt_files = []
    
    if args.src_dir and args.tgt_dir:
        # Match by filename
        s_files = sorted(os.listdir(args.src_dir))
        for f in s_files:
            if f.endswith('.mp4'): # Add other extensions if needed
                s_path = os.path.join(args.src_dir, f)
                t_path = os.path.join(args.tgt_dir, f)
                if os.path.exists(t_path):
                    src_files.append(s_path)
                    tgt_files.append(t_path)
    
    print(f"Found {len(src_files)} pairs.")
    process_batch(src_files, tgt_files, args.output_dir, args.av2unit_path, args.dict_path)
