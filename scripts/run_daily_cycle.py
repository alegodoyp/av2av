
import subprocess
import argparse
import sys
import os
from datetime import datetime

# Fix for OMP: Error #15
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Orchestrator Script

def run_command(cmd, cwd=None):
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=cwd)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="today", help="Date YYYY-MM-DD or 'today'")
    parser.add_argument("--drive-folder", default="videos_mestrado", help="Root folder on Drive")
    parser.add_argument("--inference-repo", default="../model-stst-1", help="Path to model-stst-1 repo")
    parser.add_argument("--use-raw-video", action="store_true", help="Use raw videos instead of mouth_cropped")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU to avoid OOM")
    args = parser.parse_args()
    
    if args.date == "today":
        date_str = datetime.now().strftime("%Y-%m-%d")
    else:
        date_str = args.date

    print(f"=== Starting Daily Cycle for {date_str} ===")
    
    # 1. Download Missing Models (av2unit)
    print("\n--- Step 1: Check/Download Models ---")
    run_command([sys.executable, "scripts/download_models.py"])
    
    # 2. Generate Synthetic Targets
    print("\n--- Step 2: Generate Synthetic Targets ---")
    gen_cmd = [
        sys.executable, "scripts/generate_synthetic_data.py",
        "--date", date_str,
        "--drive-folder", args.drive_folder,
        "--inference-repo", args.inference_repo
    ]
    if args.cpu:
        gen_cmd.append("--cpu")
        
    run_command(gen_cmd)
    
    # 3. Training Loop (Train 80% / Test 20%)
    print("\n--- Step 3: Train/Test ---")
    av2unit_path = "checkpoints/mavhubert_large_noise.pt" # Default download loc
    
    train_cmd = [
        sys.executable, "scripts/train_drive_pipeline.py",
        "--daily", date_str,
        "--drive-folder", args.drive_folder,
        "--src-lang", "pt",
        "--tgt-lang", "en",
        "--av2unit-path", av2unit_path,
        "--batch-size", "50"
    ]
    if args.use_raw_video:
        train_cmd.append("--use-raw-video")
        
    run_command(train_cmd)
    
    print("\n=== Daily Cycle Complete! ===")

if __name__ == "__main__":
    main()
