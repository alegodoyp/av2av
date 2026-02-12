
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import sys
import shutil
import subprocess
import json
from pathlib import Path
from collections import defaultdict

# Add project root to sys.path
sys.path.append(os.getcwd())

import drive_utils
from scripts.prepare_data import process_batch

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive-folder", required=True, help="Name of the dataset folder on Drive")
    parser.add_argument("--src-lang", required=True, help="Source language code (e.g. pt)")
    parser.add_argument("--tgt-lang", required=True, help="Target language code (e.g. en)")
    parser.add_argument("--local-dir", default="temp_data", help="Local directory for temporary storage")
    parser.add_argument("--save-dir", default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--batch-size", type=int, default=50, help="Number of video pairs per batch")
    parser.add_argument("--av2unit-path", required=True, help="Path to av2unit model")
    parser.add_argument("--dict-path", default="data/dict.txt", help="Path to dictionary file")
    parser.add_argument("--daily", default=None, help="Date YYYY-MM-DD or 'today' for daily processing")
    parser.add_argument("--use-raw-video", action="store_true", help="Use raw videos instead of mouth_cropped")
    
    # Model args
    parser.add_argument("--arch", default="conformer_utut", help="Model architecture")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--update-freq", type=int, default=1)
    parser.add_argument("--max-epoch", type=int, default=100)
    parser.add_argument("--validate-interval", type=int, default=1)
    
    return parser

def sync_dictionary(dict_path):
    # Ensure dictionary exists
    if not os.path.exists(dict_path):
        print(f"Dictionary not found at {dict_path}. Creating default 2000-unit dictionary.")
        os.makedirs(os.path.dirname(dict_path), exist_ok=True)
        with open(dict_path, 'w') as f:
            for i in range(2000):
                f.write(f"{i} 1\n")

def run_training_step(data_bin, save_dir, args):
    """Invokes fairseq-train on the prepared data."""
    
    # We always resume from checkpoint_last.pt if it exists
    restore_file = "checkpoint_last.pt"
    
    # Fairseq on Windows splits paths by ':' which breaks absolute paths (C:\...)
    # We must use a relative path for the data directory argument.
    try:
        data_bin_rel = os.path.relpath(data_bin, os.getcwd())
    except ValueError:
        # If paths are on different drives, relpath fails. Fallback to abs (and hope) or warn.
        print("Warning: Could not determine relative path for data_bin. Using absolute path which may fail on Windows.")
        data_bin_rel = str(data_bin)

    cmd = [
        sys.executable, "-m", "fairseq_cli.train",
        data_bin_rel,
        "--save-dir", str(save_dir),
        "--task", "utut_pretraining",
        "--arch", args.arch,
        # UTUT specific args
        # "--src-lang", args.src_lang, # Multilingual Denoising Task does not support these
        # "--tgt-lang", args.tgt_lang,
        "--langs", f"{args.src_lang},{args.tgt_lang}",
        # Conformer args (defaults usually fine, but user can tune)
        "--criterion", "label_smoothed_cross_entropy",
        "--label-smoothing", "0.1",
        "--optimizer", "adam", 
        "--adam-betas", "(0.9, 0.98)",
        "--lr-scheduler", "inverse_sqrt", 
        "--warmup-init-lr", "1e-07",
        "--warmup-updates", "4000",
        "--lr", "0.0005",
        "--clip-norm", "0.0",
        "--max-tokens", str(args.max_tokens),
        "--update-freq", str(args.update_freq),
        "--max-epoch", str(args.max_epoch),
        "--validate-interval", str(args.validate_interval),
        "--patience", "10",
        "--no-epoch-checkpoints", # Save space
        "--user-dir", os.path.join(os.getcwd(), "unit2unit"), # Ensure our custom model/tasks are found
        "--disable-validation", 
        "--tokens-per-sample", "4096",
        "--sample-break-mode", "eos",
        "--max-source-positions", "4096",
        "--max-target-positions", "4096",
        "--num-workers", "0",
        "--skip-invalid-size-inputs-valid-test",
    ]
    
    # If using custom dictionary path, Fairseq usually expects it in data-bin/dict.txt
    # prepare_data.py puts it there.
    
    print(f"Starting training on batch in {data_bin}...")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with return code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise e

def main():
    parser = get_parser()
    args = parser.parse_args()
    

    
    service = drive_utils.get_drive_service()
    root_id = drive_utils.find_folder(service, args.drive_folder)
    # --- New Logic for Daily + Synthetic ---
    
    if args.daily:
        from datetime import datetime
        date_str = args.daily if args.daily != "today" else datetime.now().strftime("%Y-%m-%d")
        date_obj = datetime.strptime(date_str, "%Y-%m-%d") if args.daily != "today" else datetime.now()
        
        year = str(date_obj.year)
        month = str(date_obj.month).zfill(2)
        day = str(date_obj.day).zfill(2)
        
        print(f"Daily Mode: Processing {year}/{month}/{day}")
        
        # 1. Locate Source Folder (mouth_cropped or raw)
        # root -> [mouth_cropped?] -> YYYY -> MM -> DD
        if args.use_raw_video:
            print("Using RAW videos (skipping mouth_cropped folder)")
            src_root = root_id
        else:
            src_root = drive_utils.ensure_folder_exists(service, root_id, "mouth_cropped")
            
        m_year = drive_utils.ensure_folder_exists(service, src_root, year)
        m_month = drive_utils.ensure_folder_exists(service, m_year, month)
        src_folder_id = drive_utils.ensure_folder_exists(service, m_month, day)
        
        # 2. Locate Target Folder (synthetic_targets)
        # root -> synthetic_targets -> YYYY -> MM -> DD
        syn_root = drive_utils.ensure_folder_exists(service, root_id, "synthetic_targets")
        s_year = drive_utils.ensure_folder_exists(service, syn_root, year)
        s_month = drive_utils.ensure_folder_exists(service, s_year, month)
        tgt_folder_id = drive_utils.ensure_folder_exists(service, s_month, day)
        
        # 3. List and Pair
        print("Listing Source and Target files...")
        def list_files(folder_id):
            q = f"'{folder_id}' in parents and mimeType != 'application/vnd.google-apps.folder' and trashed = false"
            res = service.files().list(q=q, fields="files(id, name)").execute()
            return {f['name']: f['id'] for f in res.get('files', [])}
            
        src_map = list_files(src_folder_id)
        tgt_map = list_files(tgt_folder_id)
        
        print(f"DEBUG: Source Files in 'mouth_cropped' ({len(src_map)}): {list(src_map.keys())}")
        print(f"DEBUG: Target Files in 'synthetic_targets' ({len(tgt_map)}): {list(tgt_map.keys())}")
        
        pairs = []
        for name, sid in src_map.items():
            # Handle 'cropped_' prefix if present
            clean_name = name
            if name.startswith("cropped_"):
                clean_name = name.replace("cropped_", "", 1)
            
            # Also handle potential 'result_' or '_translated' suffix if it existed (but logs show mp4 match)
            
            if clean_name in tgt_map:
                pairs.append((name, sid, tgt_map[clean_name]))
                print(f"MATCH: {name} -> {clean_name}")
            elif name in tgt_map:
                 # Direct match fallback
                 pairs.append((name, sid, tgt_map[name]))
                 print(f"MATCH: {name} -> {name}")
            else:
                 print(f"NO MATCH: {name} (looked for {clean_name})")
        
        print(f"Found {len(pairs)} matched pairs for {date_str}.")
        
    else:
        # Fallback to recursively scanning standard structure if --daily not used
        # (Original Logic, simplified here or kept if user wants legacy support)
        # For this request, we prioritize the new logic.
        print("Error: --daily flag is required for this pipeline version.")
        return

    import random
    random.shuffle(pairs)
    
    # 80/20 Split
    split_idx = int(len(pairs) * 0.8)
    train_pairs = pairs[:split_idx]
    test_pairs = pairs[split_idx:]
    
    print(f"Split: {len(train_pairs)} Train / {len(test_pairs)} Test")
    
    # Process Train Batches
    process_dataset(train_pairs, args, service, "train")
    
    # Process Test Batches (Validation only? Or just prepare? Fairseq needs valid set)
    # We will process test set separate and maybe run evaluation or just simple validation step
    if test_pairs:
        process_dataset(test_pairs, args, service, "valid")

def process_dataset(pairs, args, service, subset_name):
    total_batches = len(pairs) // args.batch_size + 1
    
    for i in range(0, len(pairs), args.batch_size):
        batch = pairs[i : i + args.batch_size]
        if not batch: continue
        
        batch_id = i // args.batch_size + 1
        print(f"Processing {subset_name.upper()} Batch {batch_id}/{total_batches}...")
        
        batch_dir = Path(os.path.abspath(args.local_dir)) / f"{subset_name}_batch_{batch_id}"
        src_dir = batch_dir / args.src_lang
        tgt_dir = batch_dir / args.tgt_lang
        
        src_dir.mkdir(parents=True, exist_ok=True)
        tgt_dir.mkdir(parents=True, exist_ok=True)
        
        src_files = []
        tgt_files = []
        
        for name, src_id, tgt_id in batch:
            src_path = src_dir / name
            tgt_path = tgt_dir / name
            
            drive_utils.download_file(service, src_id, str(src_path))
            drive_utils.download_file(service, tgt_id, str(tgt_path))
            
            src_files.append(str(src_path))
            tgt_files.append(str(tgt_path))
            
        # Prepare Data
        # We need to tell prepare_data which subset this is? 
        # prepare_data.py typically generates a binary. 
        # We might need to accumulate data binaries or train iteratively.
        # Simple approach: Train on this batch, then delete. 
        # For 'valid', we just compute loss? 
        # fairseq-train usually runs validation on 'valid' subset.
        
        # NOTE: Iterative training on small batches with Fairseq is complex because 
        # it usually expects the full dataset or pre-sharded data. 
        # For this pipeline, we are effectively fine-tuning continuously.
        
        # DEBUG PRINTS REMOVED

        process_batch(
            src_files, 
            tgt_files, 
            str(batch_dir), 
            args.av2unit_path,
            dict_path=os.path.abspath(os.path.join("data", "dict.txt")), 
            split=subset_name,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang
        ) 

        
        data_bin = batch_dir / "bin"
        
        if subset_name == "train":
            run_training_step(data_bin, args.save_dir, args)
        else:
            # Maybe run validation command? 
            # run_validation_step(data_bin, args.save_dir, args)
            pass
        
        # Cleanup
        shutil.rmtree(batch_dir)


if __name__ == "__main__":
    main()
