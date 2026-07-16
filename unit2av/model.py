import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from fairseq.models.text_to_speech.codehifigan import CodeGenerator as CodeHiFiGANModel
from fairseq.models.text_to_speech.vocoder import CodeHiFiGANVocoder

import torchvision
import pickle
import numpy as np
import cv2
import imageio

class UnitAVRenderer(CodeHiFiGANVocoder):
    def __init__(
        self, checkpoint_path: str, model_cfg: Dict[str, str], lang: str, fp16: bool = False
    ) -> None:
        super(CodeHiFiGANVocoder, self).__init__()
        self.model = CodeHiFiGANModel_spk(model_cfg)
        if torch.cuda.is_available():
            state_dict = torch.load(checkpoint_path)
        else:
            state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        self.model.load_state_dict(state_dict["audio"][lang])
        self.model.eval()

        self.face_model = FaceRenderer(unit_num=model_cfg["num_embeddings"])
        self.face_model.load_state_dict(state_dict["video"])
        self.face_model.eval()

        if fp16:
            self.model.half()
            self.face_model.half()
        self.model.remove_weight_norm()

        units_per_second = 50
        frames_per_second = 25
        self.num_frames = 10
        self.code_frame_ratio = units_per_second // frames_per_second
        self.num_units = self.num_frames * self.code_frame_ratio

    def get_crops(self, bbox_path):
        bbs = pickle.load(open(bbox_path, 'rb'))
        
        # Forward fill
        prev_val = None
        for i in range(len(bbs)):
            if bbs[i] is None:
                bbs[i] = prev_val
            else:
                prev_val = bbs[i]
                
        # Backward fill (for initial Nones)
        next_val = None
        for i in range(len(bbs) - 1, -1, -1):
            if bbs[i] is None:
                bbs[i] = next_val
            else:
                next_val = bbs[i]
                
        # If still None (no faces found at all), default to center crop?
        # Or just return a safe default [0, 0, 96, 96]? (Resize happens later)
        # Better to return full frame indices? 
        # Since we don't have frame size here easily without reading video, 
        # let's assume we can't do much better than failing or returing dummy.
        # But failure crashes pipeline. Let's return a "full frame" guess or similar.
        # Actually inference usage: img[y1:y2, x1:x2]. 
        # If we return [0,0,10,10] it will crop a tiny patch.
        # If we assume 1080p, [0,0,1920,1080].
        # Let's try to detect if all are None.
        
        if len(bbs) > 0 and bbs[0] is None:
            print(f"Warning: No faces detected in {bbox_path}. Using default crop.")
            # Default to a safe large crop or similar. 
            # We don't know image size here. 
            # Ideally we should handle this in inference.py but logic is split.
            # Let's return [0,0,0,0] and handle in read_window?
            # No, read_window does loops with it.
            # Let's return a dummy that triggers a full resize or safe behavior.
            # If we set 0,0,100,100 it works but might be wrong area.
            # Let's use [0,0,256,256] as a safe fallback?
            # Or better, just fill with [0,0,0,0] and handle invalid crop?
            # The code does: img[max(int(y1), 0): int(y2), max(int(x1), 0):int(x2)]
            # If we pass [0,0,W,H] it takes full image.
            # We don't know W,H.
            # Let's assume a large enough box? [0,0,10000,10000] might index out of bounds?
            # Python slicing clamps! img[0:10000, 0:10000] returns full image if too large!
            # So [0,0,10000,10000] effectively means "full image".
            for i in range(len(bbs)):
                bbs[i] = [0, 0, 10000, 10000]

        return np.array(bbs)

    def read_window(self, frames, crops):
        window = []
        for img, (x1, y1, x2, y2) in zip(frames, crops):
            img = img[max(int(y1), 0): int(y2), max(int(x1), 0):int(x2)]
            img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_AREA)
            window.append(img)
        return window 

    def prepare_window(self, window):
        # 3 x T x H x W
        x = np.asarray(window) / 255.
        x = np.transpose(x, (3, 0, 1, 2))
        return x

    def forward(self, x: Dict[str, torch.Tensor], video_path: str, bbox_path: str, dur_prediction=False, tgt_dur=None) -> torch.Tensor:
        assert "code" in x
        x["dur_prediction"] = dur_prediction
        if tgt_dur is not None:
            x["tgt_dur"] = tgt_dur

        if dur_prediction:
            x["code"] = torch.unique_consecutive(x["code"])

        # remove invalid code
        mask = x["code"] >= 0
        x["code"] = x["code"][mask].unsqueeze(dim=0)
        if "f0" in x:
            f0_up_ratio = x["f0"].size(1) // x["code"].size(1)
            mask = mask.unsqueeze(2).repeat(1, 1, f0_up_ratio).view(-1, x["f0"].size(1))
            x["f0"] = x["f0"][mask].unsqueeze(dim=0)

        # Handle batched synthesis to avoid OOM
        chunk_size = 400 # Token chunks
        out_wavs = []
        out_codes = []
        seq_len = x["code"].size(2) if x["code"].dim() == 3 else x["code"].size(1)
        
        # We need to chunk before duration prediction to avoid inflating to full size
        # Wait, if dur_prediction is True, we unique_consecutive FIRST, then the model handles duration.
        # So we can't easily chunk here if the model inflates it internally and then generates audio.
        # Actually, if the model inflates it, it returns `dedup_code` which is already padded!
        # If we chunk `x["code"]`, `tgt_dur` will be wrong for each chunk.
        # Let's chunk AFTER the model's inflate step? 
        # But `CodeHiFiGANModel` does inflate AND vocode altogether inside `forward`.
        # We must disable OOM inside `CodeHiFiGANModel`...
        
        with torch.no_grad():
            gen_wav, dedup_code = self.model(**x)
        gen_wav = gen_wav.squeeze().cpu().numpy()

        tgt_len = len(dedup_code) // self.code_frame_ratio
        remain = len(dedup_code) % self.num_units
        if remain != 0:
            repeat_num = self.num_units - remain
            dedup_code = torch.cat([dedup_code, dedup_code[-1].repeat(repeat_num)])
        padded_tgt_len = len(dedup_code) // self.code_frame_ratio
        
        # frames = torchvision.io.read_video(video_path, pts_unit="sec")[0]
        # Replace torchvision with imageio to avoid PyAV dependency
        reader = imageio.get_reader(video_path)
        frames_list = [im for im in reader]
        reader.close()
        frames = torch.from_numpy(np.stack(frames_list))
        
        # Load crops and handle mismatch
        crops = self.get_crops(bbox_path)
        min_len = min(len(frames), len(crops))
        if len(frames) != len(crops):
            print(f"Warning: Frame count mismatch! Video: {len(frames)}, Crops: {len(crops)}. Truncating to {min_len}.")
        
        frames = frames[:min_len]
        crops = crops[:min_len]
        
        len_frames = len(frames)
        reverse_frames = frames.flip(0)
        repeated_frames = torch.cat((reverse_frames[1:], frames[1:]))
        
        # Calculate exactly how many repetitions are needed
        if len(frames) < padded_tgt_len:
            repeats_needed = (padded_tgt_len - len(frames)) // len(repeated_frames) + 1
            print(f"DEBUG UnitAVRenderer: padded_tgt_len={padded_tgt_len}, len(frames)={len(frames)}, len(repeated_frames)={len(repeated_frames)}, repeats_needed={repeats_needed}")
            # Protect against catastrophic explosion
            if repeats_needed > 1000:
                print(f"ERROR: repeats_needed={repeats_needed} is insanely large! Clamping to 50.")
                repeats_needed = 50
                
            frames = torch.cat([frames] + [repeated_frames] * repeats_needed)
            
        frames = frames[:padded_tgt_len]
        frames = frames.flip(-1)
        
        # crops = self.get_crops(bbox_path) # Moved up
        # assert len(crops) == len_frames   # Handled by truncation
        reverse_crops = crops[::-1]
        repeated_crops = np.concatenate([reverse_crops[1:], crops[1:]])
        
        if len(crops) < padded_tgt_len:
            repeats_needed = (padded_tgt_len - len(crops)) // len(repeated_crops) + 1
            print(f"DEBUG UnitAVRenderer crops: padded_tgt_len={padded_tgt_len}, len(crops)={len(crops)}, repeats_needed={repeats_needed}")
            if repeats_needed > 1000:
                repeats_needed = 50
                
            crops = np.concatenate([crops] + [repeated_crops] * repeats_needed)
            
        crops = crops[:padded_tgt_len]

        frames_numpy = np.array(frames)
        window = self.read_window(frames_numpy, crops)
        wrong_window = window.copy()

        dedup_code_seq = dedup_code.view(-1, self.num_units)

        window = self.prepare_window(window)
        window[:, :, window.shape[2] // 2:] = 0.
        wrong_window = self.prepare_window(wrong_window)
        windows = np.concatenate([window, wrong_window], axis=0)
        windows = torch.FloatTensor(windows).to(dedup_code_seq.device)
        windows = windows.transpose(1,0)

        chunk_frames = 50 * self.num_units  # E.g. process 50 * 10 = 500 sequence tokens at a time representing 10s of sequence data. 
        # Wait, self.num_units is 50. code_frame_ratio is 2. (50 units = 25 frames = 1 second). Let's process 300 units (6 seconds) at a time.
        chunk_units = 300
        # Ensure chunk_units is a multiple of self.num_units
        chunk_units = (chunk_units // self.num_units) * self.num_units
        
        gen_vids = []
        with torch.no_grad():
            frames_per_seq = self.num_units // self.code_frame_ratio
            for i in range(0, dedup_code_seq.size(0), chunk_units // self.num_units):
                end = min(i + (chunk_units // self.num_units), dedup_code_seq.size(0))
                
                start_frame = i * frames_per_seq
                end_frame = end * frames_per_seq
                
                vid_chunk = self.face_model(dedup_code_seq[i:end], windows[start_frame:end_frame])
                vid_chunk = (vid_chunk.cpu().numpy().transpose(0, 2, 3, 1) * 255.0).astype(np.uint8)
                gen_vids.append(vid_chunk)

        gen_vid = np.concatenate(gen_vids, axis=0) if gen_vids else np.array([])
        
        return gen_wav, gen_vid[:tgt_len], frames_numpy[:tgt_len], crops[:tgt_len]


class CodeHiFiGANModel_spk(CodeHiFiGANModel):
    def forward(self, **kwargs):
        x = self.dict(kwargs["code"]).transpose(1, 2)
        tgt_dur = getattr(self, "tgt_dur", None)
        
        dur_pred_arg = kwargs.get("dur_prediction", False)
        print(f"DEBUG CodeHiFiGANModel_spk params: dur_predictor={self.dur_predictor is not None}, dur_prediction_arg={dur_pred_arg}, tgt_dur={tgt_dur}")

        if self.dur_predictor and dur_pred_arg:
            assert x.size(0) == 1, "only support single sample"
            log_dur_pred = self.dur_predictor(x.transpose(1, 2))
            dur_out = torch.clamp(
                torch.round((torch.exp(log_dur_pred) - 1)).long(), min=1
            )
            
            print(f"DEBUG CodeHiFiGANModel_spk: Initial dur_out sum={dur_out.sum().item()}, tgt_dur={tgt_dur}")
            if tgt_dur is not None:
                diff = tgt_dur - dur_out.sum().item()
                if diff > 0:
                    # All the padding lands on a single token, so an
                    # uncapped diff (e.g. a long source clip paired with a
                    # much shorter/truncated translation) turns into an
                    # obviously broken multi-second held note. Cap it at
                    # ~2s (100 units @ 50Hz); beyond that, let the audio
                    # just end early rather than drone on unnaturally.
                    max_pad = 100
                    applied = min(diff, max_pad)
                    dur_out[0, -1] += applied
                    print(f"DEBUG CodeHiFiGANModel_spk: Padded dur_out[0, -1] by {applied} (diff={diff}, capped at {max_pad}), new sum={dur_out.sum().item()}")
            else:
                print("DEBUG CodeHiFiGANModel_spk: tgt_dur is None... Skipping Padding.")
                    
            # B x C x T
            x = torch.repeat_interleave(x, dur_out.view(-1), dim=2)

        if self.f0:
            if self.f0_quant_embed:
                kwargs["f0"] = self.f0_quant_embed(kwargs["f0"].long()).transpose(1, 2)
            else:
                kwargs["f0"] = kwargs["f0"].unsqueeze(1)

            # Robust resizing to match x
            if x.shape[-1] != kwargs["f0"].shape[-1]:
                kwargs["f0"] = F.interpolate(kwargs["f0"], size=x.shape[-1], mode='linear', align_corners=False)
            
            x = torch.cat([x, kwargs["f0"]], dim=1)

        if self.multispkr:
            assert (
                "spkr" in kwargs
            ), 'require "spkr" input for multispeaker CodeHiFiGAN vocoder'
            spkr = self.spkr(kwargs["spkr"]).transpose(1, 2)
            
            # Robust resizing to match x
            if x.shape[-1] != spkr.shape[-1]:
                spkr = F.interpolate(spkr, size=x.shape[-1], mode='linear', align_corners=False)
                
            x = torch.cat([x, spkr], dim=1)

        for k, feat in kwargs.items():
            if k in ["spkr", "code", "f0", "dur_prediction", "tgt_dur"]:
                continue

            # Robust resizing
            if x.shape[-1] != feat.shape[-1]:
                feat = F.interpolate(feat, size=x.shape[-1], mode='linear', align_corners=False)
                
            x = torch.cat([x, feat], dim=1)

        # Chunked audio vocoding to prevent OOM
        chunk_size = 300  # Token frames (e.g. 300 * ~0.01s ~ 3 seconds audio at a time)
        out_wavs = []
        with torch.no_grad():
            for i in range(0, x.size(2), chunk_size):
                x_chunk = x[:, :, i : i + chunk_size]
                wav_chunk = super(CodeHiFiGANModel, self).forward(x_chunk)
                out_wavs.append(wav_chunk)
                
        wav_out = torch.cat(out_wavs, dim=-1)
        
        dedup_code_out = torch.repeat_interleave(kwargs["code"], dur_out.view(-1))
        print(f"DEBUG CodeHiFiGANModel_spk return: wav_out=({wav_out.shape}), dedup_code_out=({dedup_code_out.shape}), kwargs[code]={kwargs['code'].shape}")
        
        return wav_out, dedup_code_out


class FaceRenderer(nn.Module):
    def __init__(self, unit_num):
        super(FaceRenderer, self).__init__()
        self.unit_num = unit_num
        
        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(6, 16, kernel_size=7, stride=1, padding=3)),

            nn.Sequential(Conv2d(16, 32, kernel_size=3, stride=2, padding=1), 
                          Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(32, 64, kernel_size=3, stride=2, padding=1), 
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(64, 128, kernel_size=3, stride=2, padding=1), 
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(128, 256, kernel_size=3, stride=2, padding=1), 
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(256, 512, kernel_size=3, stride=2, padding=1), 
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), ),

            nn.Sequential(Conv2d(512, 512, kernel_size=3, stride=1, padding=0),  
                          Conv2d(512, 512, kernel_size=1, stride=1, padding=0)), ])

        self.unit_embed = nn.Embedding(self.unit_num, 512)
        self.unit2lip = nn.TransformerEncoderLayer(d_model=512, nhead=1, dim_feedforward=1024, dropout=0.1, activation='relu')

        self.face_decoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(512, 512, kernel_size=1, stride=1, padding=0), ),

            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=1, padding=0),  
                            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), ),

            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
                        Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                        Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), ),  

            nn.Sequential(Conv2dTranspose(768, 384, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True), ),  

            nn.Sequential(Conv2dTranspose(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), ),

            nn.Sequential(Conv2dTranspose(320, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), ), 

            nn.Sequential(Conv2dTranspose(160, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), ), ]) 

        self.output_block = nn.Sequential(Conv2d(80, 32, kernel_size=3, stride=1, padding=1),
                                          nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
                                          nn.Sigmoid())

    def forward(self, audio_sequences, face_sequences):
        audio_sequences = self.unit_embed(audio_sequences) # B,20,512 / T/10,20,512 
        audio_sequences = F.interpolate(audio_sequences.permute(0, 2, 1), scale_factor=0.5, mode='linear')  # B,512,10 / T/10,512,10
        audio_sequences = audio_sequences.permute(2, 0, 1) # 10,B,512 / 10,T/10,512
        audio_embedding = self.unit2lip(audio_sequences).permute(1,0,2)  # B,10,512
        audio_embedding = audio_embedding.contiguous().view(-1, 512).unsqueeze(-1).unsqueeze(-1)

        feats = []
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)
            feats.append(x)

        x = audio_embedding
        for f in self.face_decoder_blocks:
            x = f(x)
            try:
                x = torch.cat((x, feats[-1]), dim=1)
            except Exception as e:
                print(x.size())
                print(feats[-1].size())
                raise e

            feats.pop()

        outputs = self.output_block(x)
        return outputs
    
class nonorm_Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            )
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)
    
class Conv2dTranspose(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, output_padding=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)

class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out = out + x
        return self.act(out)
    