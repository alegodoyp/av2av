"""
Diagnostic script: inspect utut_sts_ft.pt architecture and compare with
what --arch mbart_large + current training flags would build.

Run on the VM from the repo root:
  conda activate av2av_env
  python scripts/diagnose_utut_ckpt.py --ckpt checkpoints/utut_sts_ft.pt
"""
import argparse, sys, os
sys.path.append(os.getcwd())
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch

def print_sep(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def inspect_checkpoint(ckpt_path):
    print(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")

    # ---- 1. Stored cfg['model'] ----------------------------------------
    print_sep("1. cfg['model'] stored inside checkpoint")
    cfg_model = state.get('cfg', {}).get('model', None)
    if cfg_model is None:
        print("  No cfg['model'] found.")
    elif isinstance(cfg_model, dict):
        for k, v in sorted(cfg_model.items()):
            print(f"  {k}: {v}")
    else:
        try:
            import dataclasses
            d = dataclasses.asdict(cfg_model)
            for k, v in sorted(d.items()):
                print(f"  {k}: {v}")
        except Exception:
            for k in sorted(vars(cfg_model)):
                print(f"  {k}: {getattr(cfg_model, k)}")

    # ---- 2. Key shapes from state['model'] --------------------------------
    print_sep("2. Key tensor shapes in checkpoint state dict")
    sd = state.get('model', {})

    interesting = [
        "encoder.embed_tokens.weight",
        "decoder.embed_tokens.weight",
        "encoder.embed_positions.weight",        # LearnedPos
        "encoder.embed_positions._float_tensor", # SinusoidalPos
        "decoder.embed_positions.weight",
        "decoder.embed_positions._float_tensor",
        "encoder.layernorm_embedding.weight",
        "decoder.layernorm_embedding.weight",
        "encoder.layer_norm.weight",
        "decoder.layer_norm.weight",
        "output_projection.weight",
        "encoder.output_projection.weight",
    ]
    for k in interesting:
        if k in sd:
            print(f"  PRESENT  {k}: {tuple(sd[k].shape)}")
        else:
            print(f"  ABSENT   {k}")

    # Count encoder and decoder layers
    enc_layers = sorted(set(
        int(k.split('.')[2]) for k in sd
        if k.startswith("encoder.layers.") and k.split('.')[2].isdigit()
    ))
    dec_layers = sorted(set(
        int(k.split('.')[2]) for k in sd
        if k.startswith("decoder.layers.") and k.split('.')[2].isdigit()
    ))
    print(f"\n  encoder.layers indices: {enc_layers}  → {len(enc_layers)} layers")
    print(f"  decoder.layers indices: {dec_layers}  → {len(dec_layers)} layers")

    # Sample layer shapes
    if enc_layers:
        n = enc_layers[0]
        for suffix in [
            f"encoder.layers.{n}.self_attn.q_proj.weight",
            f"encoder.layers.{n}.self_attn.out_proj.weight",
            f"encoder.layers.{n}.fc1.weight",
            f"encoder.layers.{n}.fc2.weight",
        ]:
            if suffix in sd:
                print(f"  {suffix}: {tuple(sd[suffix].shape)}")

    # ---- 3. Derive architecture from shapes --------------------------------
    print_sep("3. Architecture inferred from checkpoint shapes")

    vocab_size = None
    embed_dim = None
    if "encoder.embed_tokens.weight" in sd:
        vocab_size, embed_dim = sd["encoder.embed_tokens.weight"].shape
        print(f"  vocab_size   = {vocab_size}")
        print(f"  embed_dim    = {embed_dim}")

    has_learned_enc_pos = "encoder.embed_positions.weight" in sd
    has_sinusoidal_enc_pos = "encoder.embed_positions._float_tensor" in sd
    print(f"  encoder pos: {'LEARNED' if has_learned_enc_pos else ('SINUSOIDAL buffer' if has_sinusoidal_enc_pos else 'UNKNOWN/NO KEY')}")
    if has_learned_enc_pos:
        pos_shape = sd["encoder.embed_positions.weight"].shape
        print(f"    → embed_positions.weight shape: {tuple(pos_shape)}")
        print(f"    → max_source_positions inferred: {pos_shape[0] - 2}  (shape[0] - padding_idx - 1)")

    has_learned_dec_pos = "decoder.embed_positions.weight" in sd
    print(f"  decoder pos: {'LEARNED' if has_learned_dec_pos else ('SINUSOIDAL buffer' if 'decoder.embed_positions._float_tensor' in sd else 'UNKNOWN/NO KEY')}")
    if has_learned_dec_pos:
        pos_shape = sd["decoder.embed_positions.weight"].shape
        print(f"    → embed_positions.weight shape: {tuple(pos_shape)}")
        print(f"    → max_target_positions inferred: {pos_shape[0] - 2}")

    print(f"  layernorm_embedding encoder: {'YES' if 'encoder.layernorm_embedding.weight' in sd else 'NO'}")
    print(f"  layernorm_embedding decoder: {'YES' if 'decoder.layernorm_embedding.weight' in sd else 'NO'}")

    # Head dim and num heads from q_proj
    if enc_layers:
        n = enc_layers[0]
        qk = f"encoder.layers.{n}.self_attn.q_proj.weight"
        if qk in sd:
            q_shape = sd[qk].shape  # [embed_dim, embed_dim] typically, or [num_heads*head_dim, embed_dim]
            print(f"  self_attn q_proj shape: {tuple(q_shape)}")
            # In standard transformer q_proj: [embed_dim, embed_dim]
            # head_dim = embed_dim / num_heads → num_heads inferred from embed_dim/64 typically
        fc1k = f"encoder.layers.{n}.fc1.weight"
        if fc1k in sd:
            ffn_dim = sd[fc1k].shape[0]
            print(f"  encoder ffn_dim: {ffn_dim}")

    # ---- 4. What --arch mbart_large currently builds -----------------------
    print_sep("4. What --arch mbart_large + current flags builds")
    print("""  From fairseq/models/bart/model.py:mbart_large_architecture (calls bart_large_architecture):
  encoder_embed_dim          = 1024
  encoder_ffn_embed_dim      = 4096
  encoder_layers             = 12
  encoder_attention_heads    = 16
  encoder_normalize_before   = True   (from --encoder-normalize-before flag)
  encoder_learned_pos        = True   ← SET BY bart_large_architecture DEFAULT (line 333)
  decoder_learned_pos        = True   ← idem
  layernorm_embedding        = True   ← SET BY bart_large_architecture (line 361)
  no_scale_embedding         = False  ← mbart overrides bart's True (line 381)
  share_all_embeddings       = True
  activation_fn              = gelu
  max_source_positions       = 1024   (from --max-source-positions)
  max_target_positions       = 1024   (from --max-target-positions)
  attention_dropout          = 0.1    (from --attention-dropout)
  """)

    # ---- 5. Try actual load to expose the real error -----------------------
    print_sep("5. Attempting actual load_model_ensemble_and_task (uses checkpoint cfg)")
    try:
        from fairseq import checkpoint_utils
        # This path DOES work — it builds the model from the checkpoint's own cfg
        models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        print("  load_model_ensemble_and_task: SUCCESS")
        model = models[0]
        # Print actual model cfg
        mcfg = cfg.model
        learned_pos_enc = getattr(getattr(mcfg, 'encoder', mcfg), 'learned_pos', getattr(mcfg, 'encoder_learned_pos', '?'))
        learned_pos_dec = getattr(getattr(mcfg, 'decoder', mcfg), 'learned_pos', getattr(mcfg, 'decoder_learned_pos', '?'))
        layernorm_emb   = getattr(mcfg, 'layernorm_embedding', '?')
        no_scale        = getattr(mcfg, 'no_scale_embedding', '?')
        norm_before_enc = getattr(getattr(mcfg, 'encoder', mcfg), 'normalize_before', getattr(mcfg, 'encoder_normalize_before', '?'))
        print(f"  checkpoint cfg.model.encoder.learned_pos = {learned_pos_enc}")
        print(f"  checkpoint cfg.model.decoder.learned_pos = {learned_pos_dec}")
        print(f"  checkpoint cfg.model.layernorm_embedding = {layernorm_emb}")
        print(f"  checkpoint cfg.model.no_scale_embedding  = {no_scale}")
        print(f"  checkpoint cfg.model.encoder.normalize_before = {norm_before_enc}")
    except Exception as e:
        print(f"  load_model_ensemble_and_task FAILED: {e}")

    # ---- 6. Try simulated strict load with mbart_large args ----------------
    print_sep("6. Simulating strict load: build model with mbart_large args then load ckpt weights")
    try:
        import fairseq.models.bart  # register bart/mbart archs
        import fairseq.tasks.multilingual_denoising
        sys.path.insert(0, os.path.join(os.getcwd(), "unit2unit"))
        import task as utut_task  # register utut_pretraining

        from fairseq import options
        from fairseq.tasks import setup_task
        from fairseq.models import build_model
        from omegaconf import OmegaConf

        # This is what fairseq-train does internally:
        # build model from CLI arch, then try to load ckpt weights
        # We replicate only the shape-checking part
        sd = state.get('model', {})

        # Build model via load_model_ensemble_and_task with arg_overrides
        # to use the same args as our CLI but from checkpoint arch
        models2, cfg2, _ = checkpoint_utils.load_model_ensemble_and_task(
            [ckpt_path],
            arg_overrides={
                "encoder_normalize_before": True,
                "decoder_normalize_before": True,
                "attention_dropout": 0.1,
            }
        )
        model2 = models2[0]
        model_sd = model2.state_dict()

        missing = [k for k in model_sd if k not in sd]
        unexpected = [k for k in sd if k not in model_sd]
        mismatched = [k for k in model_sd if k in sd and model_sd[k].shape != sd[k].shape]

        print(f"  Missing in checkpoint (model expects but ckpt lacks): {missing[:10]}")
        print(f"  Unexpected in checkpoint (ckpt has but model lacks):  {unexpected[:10]}")
        print(f"  Shape mismatches:                                      {mismatched[:10]}")
        if not missing and not unexpected and not mismatched:
            print("  → All keys match! The issue is in how fairseq-train builds the model from CLI args.")
    except Exception as e:
        print(f"  Simulation failed: {e}")
        import traceback; traceback.print_exc()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/utut_sts_ft.pt")
    args = p.parse_args()
    inspect_checkpoint(args.ckpt)

if __name__ == "__main__":
    main()
