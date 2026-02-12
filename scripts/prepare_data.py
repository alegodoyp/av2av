
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

# Register AV-HuBERT task
try:
    import av2unit.avhubert.hubert_pretraining
except ImportError:
    print("Warning: Could not import av2unit.avhubert.hubert_pretraining. Task might not be registered.")
except Exception as e:
    print(f"Warning: Error importing AV-HuBERT task: {e}")

import tempfile

def load_av2unit_model(model_path, modalities="audio,video", use_cuda=True):
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([model_path])
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
    
    # Create valid.tsv
    with open(os.path.join(dataset_dir, "valid.tsv"), "w") as f:
        f.write("/\n") # Root
        # Dummy entry: id, video_path, audio_path, duration, dummy_extra
        # Set duration to 100 frames (safe for min_sample_size)
        f.write("dummy_id\tdummy.mp4\tdummy.wav\t100\tdummy_extra\n")
    
    # Create valid.{label} for each label type expected by the model
    # Model config often has 'labels': ['km'] or similar
    labels = task.cfg.labels if hasattr(task.cfg, 'labels') else ["ltr"]
    for label in labels:
        with open(os.path.join(dataset_dir, f"valid.{label}"), "w") as f:
            # Write 100 tokens to match the 100 frames duration
            f.write("1 " * 100 + "\n")
            
    # Override task config to look at temp dir
    task.cfg.data = dataset_dir
    task.cfg.label_dir = dataset_dir
    
    # Load dataset (required to initialize task.dataset utilities)
    task.load_dataset(split="valid")
    # Make the dataset accessible as task.dataset (common expectation in inference scripts)
    task.dataset = task.datasets["valid"]
    
    return models[0], task

def get_units(model, sample):
    """
    Extracts discrete units from AVHubertModel using the internal label embeddings.
    """
    with torch.no_grad():
        # 1. Extract features (disable masking for inference)
        # extract_features returns (x, padding_mask) where x is [B, T, D]
        x, _ = model.extract_features(
            sample["net_input"]["source"],
            padding_mask=sample["net_input"]["padding_mask"],
            mask=False
        )
        
        # 2. Project to final dimension
        if hasattr(model, 'final_proj') and model.final_proj is not None:
            x = model.final_proj(x)
        
        # 3. Compute logits against label embeddings
        # model.compute_logits takes (feats, emb_mat)
        # Default to the first label embedding set (usually used for pretraining)
        # model.label_embs_concat contains all, but we usually want the first codebook?
        # AVHubert pretraining usually has 1 codebook unless multi-codebook.
        # num_classes is a list.
        
        if hasattr(model, 'label_embs_concat'):
             # Assume single codebook or we want the first one
             # If multiple codebooks, we might need to split label_embs_concat
             # But for simplicity, let's look at how compute_logits does it in forward
             # It splits label_embs_concat.
             
             num_classes = model.num_classes
             label_embs_list = model.label_embs_concat.split(num_classes, 0)
             
             # We assume we want the units from the first codebook (standard usage)
             # If untie_final_proj is True, we also index final_proj, but untie is usually False.
             
             emb = label_embs_list[0]
             
             # If untie_final_proj is True, x needs to be chunked too?
             # AVHubert defaults check: untie_final_proj=False.
             
             logits = model.compute_logits(x, emb) # [B, T, V]
             
             # 4. Argmax to get units
             units = logits.argmax(dim=-1) # [B, T]
             return units
        else:
             print("Error: Model does not have label_embs_concat (not a pretraining model?)")
             return None

def extract_units(model, task, video_path, use_cuda=True):
    temp_audio_path = os.path.splitext(video_path)[0] + ".temp.wav"
    
    try:
        extract_audio_from_video(video_path, temp_audio_path)
        
        # hubert_dataset expects path:id
        task_audio_input = temp_audio_path + ":0"
        
        # We pass video_path as first arg
        # Note: load_feature returns numpy arrays
        video_feats, audio_feats = task.dataset.load_feature((video_path, task_audio_input))
        
        if audio_feats is None or video_feats is None:
            print(f"Failed to load features for {video_path}")
            return None

        audio_feats = torch.from_numpy(audio_feats.astype(np.float32)) if audio_feats is not None else None
        video_feats = torch.from_numpy(video_feats.astype(np.float32)) if video_feats is not None else None
        
        if task.dataset.normalize and 'audio' in task.dataset.modalities:
            if audio_feats is not None:
                with torch.no_grad():
                    audio_feats = F.layer_norm(audio_feats, audio_feats.shape[1:])

        # Collate (add batch dimension)
        # collater_audio returns (collated, padding_mask, starts)
        # We need to wrap in list for collater input
        collated_audios, padding_mask, _ = task.dataset.collater_audio([audio_feats], len(audio_feats))
        
        # For video, we also need to collate
        # Note: pad_audio might be irrelevant here if single sample, but we reuse logic
        collated_videos, _, _ = task.dataset.collater_audio([video_feats], len(video_feats))

        # Construct sample dict matching what model.extract_features expects
        # Fairseq models usually expect: sample['net_input']['source'], sample['net_input']['padding_mask']
        # BUT AVHubertModel.forward expects source dictionary
        
        sample = {
            "net_input": {
                "source": {"audio": collated_audios, "video": collated_videos},
                "padding_mask": padding_mask
            }
        }
        
        sample = utils.move_to_cuda(sample) if use_cuda else sample

        # call custom get_units
        units_tensor = get_units(model, sample)
        
        if units_tensor is None:
            return None
            
        # units_tensor is [B, T]. B=1.
        units = units_tensor[0].cpu().numpy()
        
        # Deduplicate (reduce)
        reduced_units = process_units(units, reduce=True)
        # Convert to string
        pred_str = " ".join(map(str, reduced_units))
        
        return pred_str

    except Exception as e:
        print(f"Error extracting units for {video_path}: {e}")
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

def process_batch(source_files, target_files, output_dir, av2unit_path, dict_path=None, split='train', src_lang='src', tgt_lang='tgt'):
    """
    Process a batch of files.
    source_files: list of paths to source lang videos
    target_files: list of paths to target lang videos (aligned by index)
    output_dir: directory to write fairseq data
    av2unit_path: path to av2unit checkpoint
    src_lang: source language code
    tgt_lang: target language code
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    use_cuda = torch.cuda.is_available()
    model, task = load_av2unit_model(av2unit_path, use_cuda=use_cuda)
    
    # If dict_path not provided, create one in output_dir
    if dict_path is None:
        dict_path = output_dir / "dict.txt"
        # Heuristic: Assume 2000 units if not known. 
        # Better: check model dictionary.
        dictionary = task.dictionaries[0]
        # Fairseq dictionary has .save() method
        dictionary.save(str(dict_path))
    
    # Write raw unit files with language extensions
    src_out = output_dir / f"{split}.{src_lang}"
    tgt_out = output_dir / f"{split}.{tgt_lang}"
    
    with open(src_out, 'w') as f_src, open(tgt_out, 'w') as f_tgt:
        for src_vid, tgt_vid in zip(source_files, target_files):
            try:
                print(f"Processing {src_vid} -> {tgt_vid}")
                src_units = extract_units(model, task, src_vid, use_cuda)
                tgt_units = extract_units(model, task, tgt_vid, use_cuda)
                
                if src_units is None or tgt_units is None:
                    print(f"Skipping pair {src_vid}, {tgt_vid} due to extraction failure.")
                    continue

                f_src.write(src_units + "\n")
                f_tgt.write(tgt_units + "\n")
                
                # Flush to ensure data is written
                f_src.flush()
                f_tgt.flush()
            except Exception as e:
                print(f"Error processing pair {src_vid}, {tgt_vid}: {e}")
                continue

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
            "--workers", "4"
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
            
            if found_bin:
                print(f"Renaming {found_bin} to {split}.bin")
                src = lang_dir / found_bin
                dst = lang_dir / f"{split}.bin"
                if dst.exists(): os.remove(dst)
                os.rename(src, dst)
                
            if found_idx:
                print(f"Renaming {found_idx} to {split}.idx")
                src = lang_dir / found_idx
                dst = lang_dir / f"{split}.idx"
                if dst.exists(): os.remove(dst)
                os.rename(src, dst)

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
