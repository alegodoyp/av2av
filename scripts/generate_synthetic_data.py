
import os
import argparse
import sys
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

# Fix for OMP: Error #15
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add project root to sys.path
sys.path.append(os.getcwd())
import drive_utils

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="Date in YYYY-MM-DD format (default: today)")
    parser.add_argument("--drive-folder", required=True, help="Root folder on Drive (videos_mestrado)")
    parser.add_argument("--inference-repo", required=True, help="Path to model-stst-1 repository")
    parser.add_argument("--temp-dir", default="temp_synthetic_gen", help="Temporary directory")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    if args.date:
        date_obj = datetime.strptime(args.date, "%Y-%m-%d")
    else:
        date_obj = datetime.now()
        
    year = str(date_obj.year)
    month = str(date_obj.month).zfill(2)
    day = str(date_obj.day).zfill(2)
    
    rel_path_to_raw = f"{year}/{month}/{day}"
    print(f"Processing data for date: {rel_path_to_raw}")
    
    # 1. Setup Drive
    print("Authenticating with Drive...")
    service = drive_utils.get_drive_service()
    
    root_id = drive_utils.find_folder(service, args.drive_folder)
    if not root_id:
        print(f"Root folder '{args.drive_folder}' not found.")
        return

    # 2. Find Raw Videos Folder (videos_mestrado/2025/11/21)
    # We need to traverse down to the specific date folder
    # This might be tricky with just 'traverse_drive_folder' if it's recursive.
    # Let's find the specific folder ID for the date.
    
    current_id = root_id
    path_found = True
    for part in [year, month, day]:
        found = False
        # List children of current_id
        # We can't easily list children with drive_utils.traverse_drive_folder efficiently without modification
        # But we can use drive_utils.find_folder logic if we adapt it or just search.
        # drive_utils.find_folder searches in root by default. 
        # We need a 'find_child_folder' function. 
        # For now, we'll assume we can search by query using a modified approach or just listing.
        
        # Let's use a naive search for now: list all children and match name.
        # Ideally drive_utils should have a 'find_path' or similar.
        # Since I can't easily change drive_utils right now effectively without checking it, 
        # I will rely on a helper here or modify drive_utils if needed.
        # Actually, let's just use the recursive traverse but filter by path? No, too slow.
        
        # We will assume standard Drive query:
        query = f"'{current_id}' in parents and name = '{part}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        results = service.files().list(q=query, fields="files(id, name)").execute()
        files = results.get('files', [])
        
        if not files:
            print(f"Folder '{part}' not found in path.")
            path_found = False
            break
        current_id = files[0]['id']
        
    if not path_found:
        print("Date folder not found. Nothing to generate.")
        return

    raw_folder_id = current_id
    print(f"Found raw folder ID: {raw_folder_id}")
    
    # 3. List Raw Videos
    videos = []
    # List files in this specific folder
    query = f"'{raw_folder_id}' in parents and mimeType != 'application/vnd.google-apps.folder' and trashed = false"
    # We want video files
    results = service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get('files', [])
    for f in files:
        if f['name'].lower().endswith(('.mp4', '.avi', '.mov')):
            videos.append(f)
            
    print(f"Found {len(videos)} raw videos.")
    
    # 4. Prepare Synthetic Targets Output Folder
    # videos_mestrado/synthetic_targets/YYYY/MM/DD
    # We need to ensure this path exists on Drive.
    
    # Find or Create 'synthetic_targets' in root
    syn_root_id = drive_utils.ensure_folder_exists(service, root_id, "synthetic_targets")
    year_id = drive_utils.ensure_folder_exists(service, syn_root_id, year)
    month_id = drive_utils.ensure_folder_exists(service, year_id, month)
    day_id = drive_utils.ensure_folder_exists(service, month_id, day)
    target_folder_id = day_id
    
    # 5. Process Loop
    temp_dir = Path(args.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    for vid in videos:
        vid_name = vid['name']
        print(f"Processing {vid_name}...")
        
        # Check if target already exists
        # (Naive check: list children of target_folder_id)
        # For efficiency, maybe fetch all existing targets once?
        # Doing per-file check for robustness.
        query = f"'{target_folder_id}' in parents and name = '{vid_name}' and trashed = false"
        res = service.files().list(q=query).execute()
        if res.get('files'):
            print(f"Target for {vid_name} already exists. Skipping.")
            continue
            
        # Download Raw
        raw_local_path = temp_dir / vid_name
        drive_utils.download_file(service, vid['id'], str(raw_local_path))
        
        # Run Inference (cross-repo call)
        # output will be in args.inference_repo/test_videos_result/vid_name_translated.mp4 
        # (based on model-stst-1/inference.py logic)
        
        # Run Inference (Official AV2AV Pipeline)
        # Using local inference.py
        
        output_temp_path = temp_dir / (os.path.splitext(vid_name)[0] + "_translated.mp4")
        
        cmd = [
            sys.executable,
            "inference.py",
            "--in-vid-path", str(raw_local_path.absolute()),
            "--out-vid-path", str(output_temp_path.absolute()),
            "--src-lang", "pt",
            "--tgt-lang", "en",
            "--av2unit-path", "checkpoints/mavhubert_large_noise.pt",
            "--utut-path", "checkpoints/utut_sts_ft.pt",
            "--unit2av-path", "checkpoints/unit_av_renderer.pt"
        ]
        
        if args.cpu:
            cmd.append("--cpu")

        print(f"Running Official AV2AV inference on {vid_name}...")
        try:
            # properly capture output for debugging
            res = subprocess.run(cmd, cwd=str(os.getcwd()), capture_output=True, text=True, check=True)
            print(f"Inference STDOUT for {vid_name}:\n{res.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"Inference failed for {vid_name}: {e}")
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)
            # Cleanup and continue to next
            # if raw_local_path.exists(): os.remove(raw_local_path)
            # if output_temp_path.exists(): os.remove(output_temp_path)
            # continue
            sys.exit(1) # Stop to debug
            
        result_path = output_temp_path
        
        if not result_path.exists():
            print(f"ERROR: Result file NOT FOUND at: {result_path}")
            continue
        
        print(f"SUCCESS: Generated {result_path} ({result_path.stat().st_size} bytes)")
            
        # Upload
        print(f"Uploading {vid_name} to Drive folder {target_folder_id}...")
        try:
            drive_utils.upload_file(service, str(result_path), target_folder_id, upload_name=vid_name)
            print("Upload complete.")
        except Exception as e:
            print(f"Upload failed: {e}")
        # We rename it back to {vid_name} for the training pipeline to match easily?
        # Or keep it distinct?
        # The user's hierarchy showed matching names: clip_01.mp4 in raw, clip_01.mp4 in mouth_cropped.
        # So we should probably upload as {vid_name} (clip_01.mp4) to the target folder.
        
        print(f"Uploading synthetic target to Drive...")
        drive_utils.upload_file(service, str(result_path), target_folder_id, upload_name=vid_name)
        
        # Cleanup
        if result_path.exists(): os.remove(result_path)
        
    # shutil.rmtree(temp_dir)
    print("Synthetic data generation complete.")

if __name__ == "__main__":
    main()
