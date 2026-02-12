import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from fairseq import checkpoint_utils, utils

from util import process_units, save_unit, extract_audio_from_video
from av2unit.task import AVHubertUnitPretrainingTask

import tempfile

def load_model(model_path, modalities, use_cuda=False):
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([model_path])

    for model in models:
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()            
        model.prepare_for_inference_(cfg)

    task.cfg.modalities = modalities.split(",")
    
    # --- Fix: Dummy Dataset for Inference ---
    task.cfg.skip_verify = True
    
    # Disable noise augmentation (avoids looking for missing /checkpoint/ paths)
    task.cfg.noise_prob = 0.0
    task.cfg.noise_wav = None

    dataset_dir = tempfile.mkdtemp()
    
    # Create valid.tsv
    # Create valid.tsv
    with open(os.path.join(dataset_dir, "valid.tsv"), "w") as f:
        f.write("\n")
        # Dummy entry: id, video_path, audio_path, duration, dummy_extra
        f.write("dummy_id\tdummy.mp4\tdummy.wav\t100\tdummy_extra\n")
    
    # Create valid.km (or whatever label is needed)
    labels = task.cfg.labels if hasattr(task.cfg, 'labels') else ["ltr"]
    for label in labels:
        with open(os.path.join(dataset_dir, f"valid.{label}"), "w") as f:
            f.write("1 " * 100 + "\n")
            
    task.cfg.data = dataset_dir
    task.cfg.label_dir = dataset_dir
    
    task.load_dataset(split="valid")
    task.dataset = task.datasets["valid"]
    # ----------------------------------------

    return models[0], task

def main(args):
    use_cuda = torch.cuda.is_available() and not args.cpu

    model, task = load_model(args.av2unit_path, args.modalities, use_cuda=use_cuda)

    temp_audio_path = os.path.splitext(args.in_vid_path)[0]+".temp.wav"
    lip_video_path = os.path.splitext(args.in_vid_path)[0]+".lip.mp4"
    extract_audio_from_video(args.in_vid_path, temp_audio_path)

    video_feats, audio_feats = task.dataset.load_feature((lip_video_path, temp_audio_path))
    audio_feats, video_feats = torch.from_numpy(audio_feats.astype(np.float32)) if audio_feats is not None else None, torch.from_numpy(video_feats.astype(np.float32)) if video_feats is not None else None
    if task.dataset.normalize and 'audio' in task.dataset.modalities:
        with torch.no_grad():
            audio_feats = F.layer_norm(audio_feats, audio_feats.shape[1:])

    collated_audios, _, _ = task.dataset.collater_audio([audio_feats], len(audio_feats))
    collated_videos, _, _ = task.dataset.collater_audio([video_feats], len(video_feats))

    sample = {"source": {
        "audio": collated_audios, "video": collated_videos,
    }}
    sample = utils.move_to_cuda(sample) if use_cuda else sample

    try:
        # pred = task.inference(model, sample)
        pred = inference(task, model, sample)
    except Exception as e:
        print(f"Error during task.inference: {e}")
        import traceback
        traceback.print_exc()
        raise e
    pred_str = task.dictionaries[0].string(pred.int().cpu())

    save_unit(pred_str, args.out_unit_path)
    os.remove(temp_audio_path)

def inference(task, model, sample):
    # Adapted from AVHubertUnitPretrainingTask.inference
    x, padding_mask = model.extract_finetune(**sample)

    label_embs_list = model.label_embs_concat.split(model.num_classes, 0)
    proj_x = model.final_proj(x)
    if model.untie_final_proj:
        proj_x_list = proj_x.chunk(len(model.num_classes), dim=-1)
    else:
        proj_x_list = [proj_x for _ in range(len(model.num_classes))] # Fixed iteration
        
    logit_list = [model.compute_logits(proj, emb).view(-1, num_class) for proj, emb, num_class in zip(proj_x_list, label_embs_list, model.num_classes)] # [[B*T, V]]

    pred_even = logit_list[0].argmax(dim=-1).cpu()
    pred_odd = logit_list[1].argmax(dim=-1).cpu()
    pred = torch.stack([pred_even, pred_odd]).transpose(0,1).reshape(-1)

    return pred

def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-vid-path", type=str, required=True, help="File path of source video input"
    )
    parser.add_argument(
        "--out-unit-path", type=str, required=True, help="File path of target unit output"
    )
    parser.add_argument(
        "--av2unit-path", type=str, required=True, help="path to the mAV-HuBERT pre-trained model"
    )
    parser.add_argument(
        "--modalities", type=str, default="audio,video", help="input modalities",
        choices=["audio,video","audio","video"],
    )
    parser.add_argument("--cpu", action="store_true", help="run on CPU")

    args = parser.parse_args()

    main(args)

if __name__ == "__main__":
    cli_main()
