
import os
import argparse
import sys
import torch
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.getcwd())

from inference import (
    AVSpeechToAVSpeechPipeline,
    load_av2unit_model,
    load_unit2unit_model,
    load_unit2av_model,
    load_speaker_encoder_model,
    extract_audio_from_video,
    save_video
)

def main(args):
    use_cuda = torch.cuda.is_available() and not args.cpu
    print(f"Using CUDA: {use_cuda}")

    print("Loading models...")
    av2unit_model, av2unit_task = load_av2unit_model(args.av2unit_path, args.modalities, use_cuda=use_cuda)
    unit2unit_task, unit2unit_generator = load_unit2unit_model(args.utut_path, args.src_lang, args.tgt_lang, use_cuda=use_cuda)
    
    cfg_path = os.path.join("unit2av", "config.json")
    unit2av_model = load_unit2av_model(args.unit2av_path, cfg_path, args.tgt_lang, use_cuda=use_cuda)
    speaker_encoder_model = load_speaker_encoder_model(os.path.join("unit2av", "encoder.pt"), use_cuda=use_cuda)

    pipeline = AVSpeechToAVSpeechPipeline(
        av2unit_model, av2unit_task,
        unit2unit_task, unit2unit_generator,
        unit2av_model, speaker_encoder_model,
        use_cuda=use_cuda
    )

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Gather videos
    extensions = {'.mp4', '.avi', '.mov', '.mkv'}
    videos = [p for p in input_dir.glob('**/*') if p.suffix.lower() in extensions]
    
    print(f"Found {len(videos)} videos in {input_dir}")

    for vid_path in videos:
        try:
            rel_path = vid_path.relative_to(input_dir)
            out_path = output_dir / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            
            print(f"Processing {vid_path} -> {out_path}...")
            
            # Temporary files
            temp_audio_path = str(out_path.with_suffix(".temp.wav"))
            lip_video_path = str(out_path.with_suffix(".lip.mp4"))
            bbox_path = str(out_path.with_suffix(".bbox.pkl"))
            
            extract_audio_from_video(str(vid_path), temp_audio_path)
            
            # 1. AV -> Unit (Source)
            # Input to process_av2unit expects lip_video_path. 
            # We pass the raw video path if lip video not available, 
            # assuming the user uses raw videos or the model handles it.
            # Similar to prepare_data.py logic.
            src_unit = pipeline.process_av2unit(str(vid_path), temp_audio_path)
            
            # 2. Unit -> Unit (Translation)
            tgt_unit = pipeline.process_unit2unit(src_unit)
            
            # 3. Unit -> AV (Synthesis)
            tgt_audio, tgt_video, full_video, bbox = pipeline.process_unit2av(
                tgt_unit, temp_audio_path, str(vid_path), bbox_path
            )

            save_video(tgt_audio, tgt_video, full_video, bbox, str(out_path))
            
            # Cleanup
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            if os.path.exists(bbox_path):
                os.remove(bbox_path)

        except Exception as e:
            print(f"Failed to process {vid_path}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="samples", help="Folder containing source videos (default: samples)")
    parser.add_argument("--output-dir", default="results", help="Folder to save translated videos (default: results)")
    parser.add_argument(
        "--src-lang", type=str, default="pt",
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
    
    args = parser.parse_args()
    main(args)
