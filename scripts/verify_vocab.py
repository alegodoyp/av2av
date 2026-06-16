"""
Vocabulary verification script for utut_sts_ft.pt fine-tuning.

Run from the repository root on the Linux machine:
    PYTHONPATH=fairseq python scripts/verify_vocab.py

Checks three things:
  1. Ground truth: reconstruct vocab exactly as the checkpoint was built.
  2. Equivalence: compare (a) training dict vs (b) inference dict vs ground truth.
  3. Load test: build mbart_large with training vocab and load checkpoint weights strict=True.
"""

import os
import sys
import tempfile
import shutil

# Make fairseq and repo root importable without installing
REPO_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "fairseq"))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
os.environ.setdefault("PYTHONPATH", os.path.join(REPO_ROOT, "fairseq"))

import torch
from fairseq.data import Dictionary
from fairseq.checkpoint_utils import load_checkpoint_to_cpu
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq import tasks

CKPT = os.path.join(REPO_ROOT, "checkpoints", "utut_sts_ft.pt")
ROOT_DICT = os.path.join(REPO_ROOT, "dict.txt")
FULL_DICT = os.path.join(REPO_ROOT, "dict_full.txt")
USER_DIR  = os.path.join(REPO_ROOT, "unit2unit")

SEP = "=" * 72


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def create_extended_dict(path):
    """Mirror of train_full_pipeline.create_extended_dict — inlined to avoid
    importing the full pipeline (which pulls in drive_utils etc.)."""
    LANG_ORDER = ["en","es","fr","it","pt","el","ru","cs","da","de",
                  "fi","hr","hu","lt","nl","pl","ro","sk","sl"]
    with open(path, "w") as f:
        for i in range(1000):
            f.write(f"{i} 1\n")
        for lang in LANG_ORDER:
            f.write(f"[{lang}] 1\n")
        f.write("<mask> 1\n")
    return LANG_ORDER


def dict_tail(d, n=25):
    """Return the last n (index, symbol) pairs of a Dictionary."""
    start = max(0, len(d) - n)
    return [(i, d[i]) for i in range(start, len(d))]


def compare_dicts(label_a, da, label_b, db):
    """Return list of (idx, sym_a, sym_b) where they differ."""
    mismatches = []
    for i in range(max(len(da), len(db))):
        sa = da[i] if i < len(da) else "<MISSING>"
        sb = db[i] if i < len(db) else "<MISSING>"
        if sa != sb:
            mismatches.append((i, sa, sb))
    return mismatches


# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — Ground truth from checkpoint
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("PART 1 — CHECKPOINT GROUND TRUTH")
print(SEP)

assert os.path.exists(CKPT), f"Checkpoint not found: {CKPT}"
ck = torch.load(CKPT, map_location="cpu", weights_only=False)

print("Checkpoint top-level keys:", list(ck.keys()))
print()
print("task_state contents:", ck.get("task_state", {}))
print()

# Embedding shapes — the most direct evidence of vocab size
for key in ["encoder.embed_tokens.weight",
            "decoder.embed_tokens.weight",
            "decoder.output_projection.weight"]:
    if key in ck["model"]:
        print(f"  ck['model']['{key}'].shape = {list(ck['model'][key].shape)}")

args = ck.get("args")
print()
print(f"args.arch            = {args.arch!r}")
print(f"args._name           = {getattr(args, '_name', '<absent>')!r}")
print(f"args.langs           = {args.langs!r}")
print(f"args.add_lang_token  = {args.add_lang_token!r}")
print(f"args.data            = {args.data!r}")

# Reconstruct the vocabulary exactly as setup_task did at training time
# Base: 1000-entry dict.txt  +  19 lang tokens (in langs order)  +  <mask>
assert os.path.exists(ROOT_DICT), f"dict.txt not found: {ROOT_DICT}"
gt_dict = Dictionary.load(ROOT_DICT)
print(f"\nAfter loading dict.txt            : {len(gt_dict)} symbols")

ckpt_langs = args.langs.split(",")
for lang in ckpt_langs:
    gt_dict.add_symbol(f"[{lang}]")
print(f"After adding {len(ckpt_langs)} lang tokens      : {len(gt_dict)} symbols")

gt_dict.add_symbol("<mask>")
print(f"After add_symbol('<mask>')        : {len(gt_dict)} symbols  ← GROUND TRUTH")

print(f"\nGround-truth vocab size = {len(gt_dict)}")
print("Last 22 symbols:")
for idx, sym in dict_tail(gt_dict, 22):
    print(f"  [{idx:4d}]  {sym}")


# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — Equivalence check
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("PART 2 — EQUIVALENCE CHECK")
print(SEP)

# ── 2a: verify dict_full.txt structure ───────────────────────────────────────
print("\n── dict_full.txt structure ──")
LANG_ORDER = create_extended_dict(FULL_DICT)
with open(FULL_DICT) as fh:
    lines = fh.readlines()

print(f"  Total lines : {len(lines)}  (expected 1020)")
print(f"  Lines  0-1  : {[l.rstrip() for l in lines[:2]]}")
print(f"  Lines  999  : {lines[999].rstrip()!r}")
print(f"  Lines 1000  : {lines[1000].rstrip()!r}  ← first lang token")
print(f"  Lines 1018  : {lines[1018].rstrip()!r}  ← last lang token ([sl])")
print(f"  Lines 1019  : {lines[1019].rstrip()!r}  ← <mask>")
assert len(lines) == 1020, f"FAIL: expected 1020 lines, got {len(lines)}"
print("  ✓ exactly 1020 lines")

# ── 2b: training dict (dict_full.txt + add_lang_token pt,en) ─────────────────
print("\n── (a) TRAINING DICT (dict_full.txt + --add-lang-token pt,en) ──")
dict_train = Dictionary.load(FULL_DICT)
print(f"  After load               : {len(dict_train)} symbols")

added_train = {}
for lang in [args.langs.split(",")[i] for i in [4, 0]]:  # pt=index4, en=index0 in lang order
    # Use the actual --langs arg from run_training: "{src_lang},{tgt_lang}" = "pt,en"
    pass

for lang in ["pt", "en"]:   # --langs pt,en in run_training
    idx = dict_train.add_symbol(f"[{lang}]")
    added_train[lang] = idx
    print(f"  add_symbol('[{lang}]') → {idx}  {'(existing)' if idx < 1024 else '(NEW - bad!)'}")

mask_idx_train = dict_train.add_symbol("<mask>")
print(f"  add_symbol('<mask>') → {mask_idx_train}  {'(existing)' if mask_idx_train < 1024 else '(NEW - bad!)'}")
print(f"  Training dict final size : {len(dict_train)}")

# ── 2c: inference dict (dict.txt + all 19 langs + mask) ──────────────────────
print("\n── (b) INFERENCE DICT (dict.txt + setup_task 19 langs + mask) ──")
dict_infer = Dictionary.load(ROOT_DICT)
print(f"  After load dict.txt      : {len(dict_infer)} symbols")
for lang in ckpt_langs:
    dict_infer.add_symbol(f"[{lang}]")
print(f"  After 19 lang tokens     : {len(dict_infer)} symbols")
dict_infer.add_symbol("<mask>")
print(f"  Inference dict final size: {len(dict_infer)} symbols")

# ── comparison ────────────────────────────────────────────────────────────────
print("\n── Symbol-by-symbol comparison ──")
mm_gt_train  = compare_dicts("gt",    gt_dict,    "train", dict_train)
mm_gt_infer  = compare_dicts("gt",    gt_dict,    "infer", dict_infer)
mm_train_inf = compare_dicts("train", dict_train, "infer", dict_infer)

all_ok = (
    len(mm_gt_train)  == 0 and
    len(mm_gt_infer)  == 0 and
    len(mm_train_inf) == 0 and
    len(gt_dict)    == 1024 and
    len(dict_train) == 1024 and
    len(dict_infer) == 1024
)

if all_ok:
    print(f"  ✓ ALL THREE DICTS IDENTICAL — len = 1024")
else:
    sizes = f"gt={len(gt_dict)}, train={len(dict_train)}, infer={len(dict_infer)}"
    print(f"  ✗ MISMATCH DETECTED  ({sizes})")
    for label, mm in [("gt vs train", mm_gt_train),
                      ("gt vs infer", mm_gt_infer),
                      ("train vs infer", mm_train_inf)]:
        if mm:
            print(f"\n  {label} mismatches ({len(mm)}):")
            for idx, sa, sb in mm[:30]:
                print(f"    [{idx:4d}]  {sa!r}  vs  {sb!r}")


# ─────────────────────────────────────────────────────────────────────────────
# PART 3 — Model load test (mbart_large + finetune-from-model weights)
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("PART 3 — FINETUNE-FROM-MODEL LOAD TEST")
print(SEP)

# Build task using the same path as inference (root dict.txt + all 19 langs).
# This gives vocab=1024, matching both the checkpoint and training.
state = load_checkpoint_to_cpu(
    CKPT,
    arg_overrides={"user_dir": USER_DIR, "data": REPO_ROOT}
)

if "args" in state and state["args"] is not None:
    cfg = convert_namespace_to_omegaconf(state["args"])
elif "cfg" in state and state["cfg"] is not None:
    cfg = state["cfg"]
else:
    raise RuntimeError(f"Neither args nor cfg in checkpoint keys: {list(state.keys())}")

print(f"cfg.model._name     = {cfg.model._name!r}")
print(f"cfg.task.langs      = {cfg.task.langs!r}")
print(f"cfg.task.add_lang_token = {cfg.task.add_lang_token!r}")
print(f"cfg.task.data       = {cfg.task.data!r}")

# Patch data to repo root so dict.txt is found
cfg.task.data = REPO_ROOT

task = tasks.setup_task(cfg.task)
print(f"\nTask type           : {type(task).__name__}")
print(f"Task dict size      : {len(task.dictionary)}")
print(f"Task dict last 5    : {[(i, task.dictionary[i]) for i in range(len(task.dictionary)-5, len(task.dictionary))]}")

# Build new model from cfg
model = task.build_model(cfg.model)
print(f"\nNew model type      : {type(model).__name__}")

# Inspect positional embedding type
enc_pos = model.encoder.embed_positions
dec_pos = model.decoder.embed_positions
print(f"encoder.embed_positions type : {type(enc_pos).__name__}")
print(f"decoder.embed_positions type : {type(dec_pos).__name__}")

# Check shapes match checkpoint
ck_enc_embed = state["model"]["encoder.embed_tokens.weight"]
new_enc_embed = model.encoder.embed_tokens.weight
print(f"\nCheckpoint  encoder.embed_tokens.weight : {list(ck_enc_embed.shape)}")
print(f"New model   encoder.embed_tokens.weight : {list(new_enc_embed.shape)}")
shape_ok = (ck_enc_embed.shape == new_enc_embed.shape)
print(f"Shape match : {'✓' if shape_ok else '✗ MISMATCH'}")

# Simulate strict load_state_dict (same call as trainer.py:585)
print("\nRunning model.load_state_dict(strict=True) ...")
try:
    result = model.load_state_dict(state["model"], strict=True, model_cfg=cfg.model)
    missing   = getattr(result, "missing_keys",   [])
    unexpected = getattr(result, "unexpected_keys", [])
    print(f"  missing_keys   ({len(missing)})   : {missing[:10]}")
    print(f"  unexpected_keys({len(unexpected)}) : {unexpected[:10]}")
    if not missing and not unexpected:
        print("  ✓ ZERO missing, ZERO unexpected keys — checkpoint loads cleanly")
    else:
        print("  ✗ Keys mismatch — fine-tuning will crash")
except RuntimeError as e:
    print(f"  ✗ RuntimeError: {e}")

print()
print(SEP)
print("VERIFICATION COMPLETE")
print(SEP)
