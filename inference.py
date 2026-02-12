import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import pickle
import imageio
import face_alignment

from fairseq import utils
from fairseq_cli.generate import get_symbols_to_strip_from_output

from av2unit.inference import load_model as load_av2unit_model
from unit2unit.inference import load_model as load_unit2unit_model
from unit2av.inference import load_model as load_unit2av_model, load_speaker_encoder_model

from util import process_units, extract_audio_from_video, save_video

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

        # On-demand Move to GPU
        self._to_gpu(self.av2unit_model)

        with torch.cuda.amp.autocast():
            pred = inference_av2unit_chunked(task, self.av2unit_model, sample, chunk_size=400)
        
        # Offload to CPU
        self._to_cpu(self.av2unit_model)

        pred_str = task.dictionaries[0].string(pred.int().cpu())

        return pred_str

    def process_unit2unit(self, unit):
        task = self.unit2unit_task
        unit = list(map(int, unit.strip().split()))
        unit = task.source_dictionary.encode_line(
            " ".join(map(lambda x: str(x), process_units(unit, reduce=True))),
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
        sample = utils.move_to_cuda(sample) if self.use_cuda else sample

        # On-demand Move to GPU
        self._to_gpu(self.unit2unit_generator)

        with torch.cuda.amp.autocast():
            pred = task.inference_step(
                self.unit2unit_generator,
                None,
                sample,
            )[0][0]

        # Offload to CPU
        self._to_cpu(self.unit2unit_generator)

        pred_str = task.target_dictionary.string(
            pred["tokens"].int().cpu(),
            extra_symbols_to_ignore=get_symbols_to_strip_from_output(self.unit2unit_generator)
        )

        return pred_str

    def process_unit2av(self, unit, audio_path, video_path, bbox_path):
        unit = list(map(int, unit.strip().split()))

        sample = {
            "code": torch.LongTensor(unit).view(1,-1),
            "spkr": torch.from_numpy(self.speaker_encoder.get_embed(audio_path)).view(1,1,-1),
        }
        sample = utils.move_to_cuda(sample) if self.use_cuda else sample

        # On-demand Move to GPU
        self._to_gpu(self.unit2av_model)
        self._to_gpu(self.speaker_encoder)

        with torch.cuda.amp.autocast():
            wav, video, full_video, bbox = self.unit2av_model(sample, video_path, bbox_path, dur_prediction=True)

        # Offload to CPU
        self._to_cpu(self.unit2av_model)
        self._to_cpu(self.speaker_encoder)

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

    # Load models on CPU initially to save VRAM and avoid OOM
    # The pipeline will move them to GPU on demand
    av2unit_model, av2unit_task = load_av2unit_model(args.av2unit_path, args.modalities, use_cuda=False)
    unit2unit_task, unit2unit_generator = load_unit2unit_model(args.utut_path, args.src_lang, args.tgt_lang, use_cuda=False)
    cfg_path = os.path.join("unit2av", "config.json")
    unit2av_model = load_unit2av_model(args.unit2av_path, cfg_path, args.tgt_lang, use_cuda=False, fp16=True)
    speaker_encoder_model = load_speaker_encoder_model(os.path.join("unit2av", "encoder.pt"), use_cuda=False)

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
    tgt_audio, tgt_video, full_video, bbox = pipeline.process_unit2av(tgt_unit, temp_audio_path, args.in_vid_path, bbox_path)

    save_video(tgt_audio, tgt_video, full_video, bbox, args.out_vid_path)

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
    args = parser.parse_args()
    main(args)

if __name__ == "__main__":
    cli_main()
