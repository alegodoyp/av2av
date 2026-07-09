import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import sys
import shutil
import subprocess
from pathlib import Path
import random

# Add project root to sys.path
sys.path.append(os.getcwd())

import drive_utils
try:
    from scripts.prepare_data import process_batch
except ModuleNotFoundError:
    try:
        from prepare_data import process_batch
    except ModuleNotFoundError:
        sys.path.append(os.path.dirname(__file__))
        from prepare_data import process_batch
import torch

def get_parser():
    parser = argparse.ArgumentParser(description="Pipeline to train on the FULL dataset recursively from Drive")
    parser.add_argument("--drive-folder", required=True, help="Name of the dataset folder on Drive")
    parser.add_argument("--src-lang", required=True, help="Source language code (e.g. pt)")
    parser.add_argument("--tgt-lang", required=True, help="Target language code (e.g. en)")
    parser.add_argument("--local-dir", default="temp_full_dataset", help="Local directory for temporary storage of downloads")
    parser.add_argument("--save-dir", default="checkpoints/full_model", help="Directory to save checkpoints")
    parser.add_argument("--batch-size", type=int, default=16, help="Number of video pairs per batch in training")
    parser.add_argument("--av2unit-path", required=True, help="Path to av2unit model (for source language)")
    parser.add_argument("--tgt-av2unit-path", default=None, help="Path to av2unit model for target language (if different)")
    parser.add_argument("--dict-path", default="dict.txt", help="Path to dictionary file")
    parser.add_argument("--use-raw-video", action="store_true", help="Use raw videos instead of mouth_cropped")
    parser.add_argument("--split-ratio", type=float, default=0.8, help="Train/Validation split ratio (default: 0.8)")
    
    # Model args
    parser.add_argument("--arch", default="conformer_utut", help="Model architecture")
    parser.add_argument("--max-tokens", type=int, default=200000)
    parser.add_argument("--update-freq", type=int, default=1)
    parser.add_argument("--max-epoch", type=int, default=100, help="Maximum number of training epochs")
    parser.add_argument("--validate-interval", type=int, default=1)
    
    # MLFlow args
    parser.add_argument("--mlflow-tracking-uri", default=None, help="MLFlow tracking URI")
    parser.add_argument("--mlflow-experiment-name", default=None, help="MLFlow experiment name")
    
    return parser

def create_extended_dict(output_path):
    """Create dict.txt with 1020 entries matching utut_sts_ft.pt vocab=1024.

    Layout: 4 specials (bos/pad/eos/unk) + 1000 units + 19 lang tokens + <mask> = 1024.
    Lang token order must match the checkpoint's --langs argument exactly.
    """
    LANG_ORDER = ["en","es","fr","it","pt","el","ru","cs","da","de","fi","hr","hu","lt","nl","pl","ro","sk","sl"]
    with open(output_path, 'w') as f:
        for i in range(1000):
            f.write(f"{i} 1\n")
        for lang in LANG_ORDER:
            f.write(f"[{lang}] 1\n")
        f.write("<mask> 1\n")


def patch_utut_checkpoint(src_path):
    """Strip stale SinusoidalPositionalEmbedding buffers so strict=True load works.

    utut_sts_ft.pt was saved with an older fairseq that kept _float_tensor as a
    persistent buffer on SinusoidalPositionalEmbedding. Current fairseq no longer
    registers it, so it's absent from the model's state_dict but present in the
    checkpoint — causing strict=True to raise "unexpected keys". We remove those
    two keys and write a patched copy next to the original.
    """
    import tempfile
    patched_path = os.path.join(tempfile.gettempdir(), "utut_sts_ft_patched.pt")
    state = torch.load(src_path, map_location="cpu")
    stale_keys = [k for k in state.get("model", {}) if k.endswith(".embed_positions._float_tensor")]
    for k in stale_keys:
        del state["model"][k]
    if stale_keys:
        print(f"Patched checkpoint: removed stale buffer keys {stale_keys}")
    torch.save(state, patched_path)
    return patched_path


def run_training(data_bin, save_dir, args):
    """Invokes fairseq-train on the complete prepared data."""
    try:
        data_bin_rel = os.path.relpath(data_bin, os.getcwd())
    except ValueError:
        data_bin_rel = str(data_bin)

    # Strip stale _float_tensor buffers from the pretrained checkpoint so that
    # --finetune-from-model can load it with strict=True (trainer.py:586).
    finetune_ckpt = patch_utut_checkpoint("checkpoints/utut_sts_ft.pt")

    cmd_args = [
        data_bin_rel,
        "--save-dir", str(save_dir),
        "--task", "utut_pretraining",
        # utut_large is registered in unit2unit/model.py and matches utut_sts_ft.pt exactly:
        # 12 layers, dim=1024, heads=16, ffn=4096, normalize_before=True,
        # sinusoidal pos (learned_pos=False), no layernorm_embedding.
        "--arch", "utut_large",
        # Pass only the two data languages; all 19 lang tokens are pre-baked in dict_full.txt
        # so add_lang_token just looks up existing indices without growing the vocab.
        "--langs", f"{args.src_lang},{args.tgt_lang}",
        "--add-lang-token",
        "--encoder-normalize-before",
        "--decoder-normalize-before",
        "--attention-dropout", "0.1",
        "--criterion", "label_smoothed_cross_entropy",
        "--label-smoothing", "0.2",
        "--optimizer", "adam",
        "--adam-betas", "(0.9, 0.98)",
        "--lr-scheduler", "polynomial_decay",
        "--total-num-update", "50000",
        "--warmup-updates", "3000",
        "--lr", "1e-4",
        "--clip-norm", "0.1",
        "--batch-size", str(args.batch_size),
        "--max-tokens", str(args.max_tokens),
        "--update-freq", str(args.update_freq),
        "--max-epoch", str(args.max_epoch),
        "--validate-interval", str(args.validate_interval),
        "--patience", "10",
        "--no-epoch-checkpoints",
        "--finetune-from-model", finetune_ckpt,
        "--user-dir", os.path.join(os.getcwd(), "unit2unit"),
        "--tokens-per-sample", "1020",
        "--sample-break-mode", "eos",
        "--max-source-positions", "1024",
        "--max-target-positions", "1024",
        "--num-workers", "32",
        "--skip-invalid-size-inputs-valid-test",
        # required-batch-size-multiple=8 trims batches to the nearest multiple of 8.
        # With a small dataset (<8 valid samples), every batch becomes empty → crash.
        # Truncate sequences longer than tokens_per_sample-2 (=1018) instead of
        # filtering them. Without this, all samples from long videos (>40s at 25fps)
        # exceed max_positions=1024 and are silently dropped, leaving empty datasets.
        # Empty shorten_data_split_list (default) applies to both train and valid.
        "--shorten-method", "truncate",
        # fairseq defaults distributed_world_size to torch.cuda.device_count().
        # With a small valid set, ShardedIterator can give 0 batches per shard.
        # Force single-process training here; remove for multi-GPU production runs.
        "--distributed-world-size", "1",
    ]
    
    print(f"Starting training on full dataset in {data_bin}...", flush=True)

    env = os.environ.copy()
    if args.mlflow_tracking_uri:
        env["MLFLOW_TRACKING_URI"] = args.mlflow_tracking_uri
    if args.mlflow_experiment_name:
        env["MLFLOW_EXPERIMENT_NAME"] = args.mlflow_experiment_name
    fairseq_path = os.path.abspath("fairseq")
    env["PYTHONPATH"] = fairseq_path + os.pathsep + env.get("PYTHONPATH", "")

    train_script = os.path.join(fairseq_path, "fairseq_cli", "train.py")
    cmd = [sys.executable, train_script] + cmd_args
    
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with return code {e.returncode}")
        raise e

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    service = drive_utils.get_drive_service()
    root_id = drive_utils.find_folder(service, args.drive_folder)
    if not root_id:
        print(f"Error: Drive folder '{args.drive_folder}' not found.")
        sys.exit(1)
        
    # 1. Locate Source Folder (mouth_cropped or raw)
    if args.use_raw_video:
        print("Using RAW videos as source.")
        src_root_id = root_id
    else:
        src_root_id = drive_utils.find_folder(service, "mouth_cropped", root_id)
        if not src_root_id:
            print("Warning: 'mouth_cropped' folder not found under root folder. Falling back to root folder.")
            src_root_id = root_id
            
    # 2. Locate Target Folder (synthetic_targets)
    tgt_root_id = drive_utils.find_folder(service, "synthetic_targets", root_id)
    if not tgt_root_id:
        print("Error: 'synthetic_targets' folder not found under root folder.")
        sys.exit(1)
        
    print("Recursively traversing Google Drive folders to find all video pairs...")
    
    # 3. Traverse Source Files
    src_files_map = {}
    for item, rel_path in drive_utils.traverse_drive_folder(service, src_root_id):
        if item['mimeType'] != 'application/vnd.google-apps.folder':
            if item['name'].lower().endswith(('.mp4', '.avi', '.mov')):
                norm_path = rel_path.replace("\\", "/")
                src_files_map[norm_path] = item['id']
                
    # 4. Traverse Target Files
    tgt_files_map = {}
    for item, rel_path in drive_utils.traverse_drive_folder(service, tgt_root_id):
        if item['mimeType'] != 'application/vnd.google-apps.folder':
            if item['name'].lower().endswith(('.mp4', '.avi', '.mov')):
                norm_path = rel_path.replace("\\", "/")
                tgt_files_map[norm_path] = item['id']
                
    print(f"Found {len(src_files_map)} source videos and {len(tgt_files_map)} target videos on Drive.")
    
    # 5. Match Pairs
    pairs = []
    for rel_path, src_id in src_files_map.items():
        parts = rel_path.split('/')
        filename = parts[-1]
        
        # Handle 'cropped_' prefix in source files
        clean_filename = filename
        if filename.startswith("cropped_"):
            clean_filename = filename.replace("cropped_", "", 1)
            
        tgt_rel_path = "/".join(parts[:-1] + [clean_filename])
        
        if tgt_rel_path in tgt_files_map:
            pairs.append((filename, src_id, tgt_files_map[tgt_rel_path]))
        elif rel_path in tgt_files_map:
            pairs.append((filename, src_id, tgt_files_map[rel_path]))
            
    print(f"Matched {len(pairs)} video pairs.")
    if not pairs:
        print("No video pairs matched. Exiting.")
        sys.exit(0)
        
    # Shuffle and split into Train/Validation
    random.shuffle(pairs)
    split_idx = int(len(pairs) * args.split_ratio)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]
    
    print(f"Split: {len(train_pairs)} Train / {len(val_pairs)} Validation")
    
    # 6. Setup Local Dirs
    local_dir = Path(os.path.abspath(args.local_dir))
    train_dir = local_dir / "train_data"
    val_dir = local_dir / "val_data"

    # Create extended dict compatible with utut_sts_ft.pt (vocab=1024).
    # This must be done before download_and_preprocess so process_batch uses it.
    extended_dict_path = os.path.abspath("dict_full.txt")
    create_extended_dict(extended_dict_path)
    args.dict_path = extended_dict_path
    print(f"Extended dict written to {extended_dict_path} (1020 entries, vocab=1024).")

    # Clean up previous runs if any
    if local_dir.exists():
        shutil.rmtree(local_dir)
        
    # Helper to download and process a subset
    def download_and_preprocess(pairs_list, subset_name, subset_dir):
        if not pairs_list:
            print(f"No pairs to process for {subset_name}.")
            return
            
        print(f"\n--- Downloading and processing {subset_name} set ({len(pairs_list)} pairs) ---")
        src_dir = subset_dir / args.src_lang
        tgt_dir = subset_dir / args.tgt_lang
        src_dir.mkdir(parents=True, exist_ok=True)
        tgt_dir.mkdir(parents=True, exist_ok=True)
        
        src_files = []
        tgt_files = []
        
        for idx, (name, src_id, tgt_id) in enumerate(pairs_list):
            src_path = src_dir / name
            tgt_path = tgt_dir / name
            
            print(f"[{idx+1}/{len(pairs_list)}] Downloading {name}...", flush=True)
            drive_utils.download_file(service, src_id, str(src_path))
            drive_utils.download_file(service, tgt_id, str(tgt_path))
            
            src_files.append(str(src_path))
            tgt_files.append(str(tgt_path))
            
        print(f"Extracting units and preprocessing {subset_name} binaries...", flush=True)
        
        # Ensure dict exists
        dict_path = os.path.abspath(args.dict_path)
        if not os.path.exists(dict_path):
            print(f"Dictionary not found at {dict_path}. Creating default.")
            os.makedirs(os.path.dirname(dict_path), exist_ok=True)
            with open(dict_path, 'w') as f:
                for i in range(2000):
                    f.write(f"{i} 1\n")
                    
        process_batch(
            src_files,
            tgt_files,
            str(subset_dir),
            args.av2unit_path,
            tgt_av2unit_path=args.tgt_av2unit_path,
            dict_path=dict_path,
            split=subset_name,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang
        )
        
    # Download and process train and valid subsets
    download_and_preprocess(train_pairs, "train", train_dir)
    download_and_preprocess(val_pairs, "valid", val_dir)
    
    # 7. Merge Binaries into one unified data-bin
    # fairseq-preprocess generates binaries in subset_dir/bin/lang/...
    # We combine them into a single data-bin structure:
    # data_bin/bin/pt/train.bin, train.idx, valid.bin, valid.idx
    unified_bin_dir = local_dir / "unified_bin"
    unified_bin_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy dictionary
    shutil.copy(args.dict_path, unified_bin_dir / "dict.txt")
    
    for lang in [args.src_lang, args.tgt_lang]:
        lang_bin_dir = unified_bin_dir / lang
        lang_bin_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(args.dict_path, lang_bin_dir / "dict.txt")
        
        # Copy train files
        train_lang_dir = train_dir / "bin" / lang
        if (train_lang_dir / "train.bin").exists():
            shutil.copy(train_lang_dir / "train.bin", lang_bin_dir / "train.bin")
            shutil.copy(train_lang_dir / "train.idx", lang_bin_dir / "train.idx")
            
        # Copy validation files
        val_lang_dir = val_dir / "bin" / lang
        if (val_lang_dir / "valid.bin").exists():
            shutil.copy(val_lang_dir / "valid.bin", lang_bin_dir / "valid.bin")
            shutil.copy(val_lang_dir / "valid.idx", lang_bin_dir / "valid.idx")
            
    # 8. Validate binary files before training
    missing = []
    empty = []
    for lang in [args.src_lang, args.tgt_lang]:
        for split_name in ["train", "valid"]:
            for ext in [".bin", ".idx"]:
                fpath = unified_bin_dir / lang / f"{split_name}{ext}"
                if not fpath.exists():
                    missing.append(str(fpath))
                elif fpath.stat().st_size == 0:
                    empty.append(str(fpath))
    if missing:
        print(f"ERROR: Missing binary files: {missing}")
    if empty:
        print(f"ERROR: Empty binary files (0 bytes): {empty}")
    if missing or empty:
        raise RuntimeError(
            "Cannot start training — binary data files are missing or empty. "
            "Check unit extraction and fairseq-preprocess output above."
        )

    # 9. Start fairseq training on the unified binaries
    os.makedirs(args.save_dir, exist_ok=True)
    run_training(unified_bin_dir, args.save_dir, args)
    
    # Cleanup downloads to free disk space
    print("Training finished! Cleaning up temporary data directory...")
    # Optional: comment this out if you want to keep the preprocessed data-bin
    # shutil.rmtree(local_dir)
    print("Done!")

if __name__ == "__main__":
    main()
