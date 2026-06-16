"""
Vocabulary verification script for utut_sts_ft.pt fine-tuning.

Run from the repository root on the Linux machine:
    python scripts/verify_vocab.py

Checks three things:
  1. Ground truth: reconstruct vocab exactly as the checkpoint was built.
  2. Equivalence: compare (a) training dict vs (b) inference dict vs ground truth.
  3. Load test: build mbart_large with training vocab and load checkpoint weights strict=True.
"""

import os
import sys

# ── path setup ────────────────────────────────────────────────────────────────
# FAIRSEQ_PATH must come BEFORE REPO_ROOT in sys.path.
# If REPO_ROOT is listed first, Python 3 treats av2av/fairseq/ as a *namespace*
# package (no __init__.py at that level) and fairseq.data is never found.
# With FAIRSEQ_PATH first, Python finds av2av/fairseq/fairseq/ (which has
# __init__.py) and stops looking — correct package, correct .data submodule.
REPO_ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FAIRSEQ_PATH = os.path.join(REPO_ROOT, "fairseq")

for p in [REPO_ROOT, FAIRSEQ_PATH]:        # insert in reverse priority order
    if p not in sys.path:
        sys.path.insert(0, p)
# After the loop: sys.path[0] = FAIRSEQ_PATH, sys.path[1] = REPO_ROOT  ✓
# (each insert(0,...) shifts the previous one to position 1)

import torch
from fairseq.data import Dictionary
from fairseq.checkpoint_utils import load_checkpoint_to_cpu
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq import tasks

CKPT      = os.path.join(REPO_ROOT, "checkpoints", "utut_sts_ft.pt")
ROOT_DICT = os.path.join(REPO_ROOT, "dict.txt")
FULL_DICT = os.path.join(REPO_ROOT, "dict_full.txt")
USER_DIR  = os.path.join(REPO_ROOT, "unit2unit")

SEP = "=" * 72


# ─────────────────────────────────────────────────────────────────────────────
def create_extended_dict(path):
    """Inline copy of train_full_pipeline.create_extended_dict."""
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
    start = max(0, len(d) - n)
    return [(i, d[i]) for i in range(start, len(d))]


def compare_dicts(da, db):
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
print("task_state:", ck.get("task_state", {}))
print()

for key in ["encoder.embed_tokens.weight",
            "decoder.embed_tokens.weight",
            "decoder.output_projection.weight"]:
    if key in ck["model"]:
        print(f"  ck['model']['{key}'].shape = {list(ck['model'][key].shape)}")

args = ck.get("args")
print()
print(f"args.arch           = {args.arch!r}")
print(f"args._name          = {getattr(args, '_name', '<absent>')!r}")
print(f"args.langs          = {args.langs!r}")
print(f"args.add_lang_token = {args.add_lang_token!r}")
print(f"args.data           = {args.data!r}")

# Reconstruct vocab exactly as setup_task built it at training time
assert os.path.exists(ROOT_DICT), f"dict.txt not found: {ROOT_DICT}"
gt_dict = Dictionary.load(ROOT_DICT)
print(f"\nAfter Dictionary.load(dict.txt)        : {len(gt_dict)} symbols")

ckpt_langs = args.langs.split(",")
for lang in ckpt_langs:
    gt_dict.add_symbol(f"[{lang}]")
print(f"After adding {len(ckpt_langs)} lang tokens ({args.langs[:20]}...): {len(gt_dict)} symbols")

gt_dict.add_symbol("<mask>")
print(f"After add_symbol('<mask>')              : {len(gt_dict)} symbols  ← GROUND TRUTH")

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

# ── 2-0: verify dict_full.txt structure ──────────────────────────────────────
print("\n── dict_full.txt structure ──")
LANG_ORDER = create_extended_dict(FULL_DICT)
with open(FULL_DICT) as fh:
    lines = fh.readlines()

print(f"  Total lines  : {len(lines)}  (expected 1020)")
print(f"  lines[0]     : {lines[0].rstrip()!r}")
print(f"  lines[999]   : {lines[999].rstrip()!r}  ← last unit token")
print(f"  lines[1000]  : {lines[1000].rstrip()!r}  ← first lang token ([en])")
print(f"  lines[1018]  : {lines[1018].rstrip()!r}  ← last lang token ([sl])")
print(f"  lines[1019]  : {lines[1019].rstrip()!r}  ← <mask>")
if len(lines) == 1020:
    print("  ✓ exactly 1020 lines")
else:
    print(f"  ✗ expected 1020, got {len(lines)}")

# ── 2a: training dict ────────────────────────────────────────────────────────
print("\n── (a) TRAINING DICT  [ dict_full.txt + --add-lang-token --langs pt,en ] ──")
dict_train = Dictionary.load(FULL_DICT)
print(f"  After load                 : {len(dict_train)} symbols")
for lang in ["pt", "en"]:   # --langs from run_training (src_lang, tgt_lang)
    idx = dict_train.add_symbol(f"[{lang}]")
    note = "already exists ✓" if idx < len(dict_train) - 1 else "NEW ✗"
    print(f"  add_symbol('[{lang}]') → {idx}  ({note})")
mask_idx_tr = dict_train.add_symbol("<mask>")
note = "already exists ✓" if mask_idx_tr < len(dict_train) - 1 else "NEW ✗"
print(f"  add_symbol('<mask>') → {mask_idx_tr}  ({note})")
print(f"  Training dict final size   : {len(dict_train)}")

# ── 2b: inference dict ───────────────────────────────────────────────────────
print("\n── (b) INFERENCE DICT  [ dict.txt + setup_task(19 langs) + mask ] ──")
dict_infer = Dictionary.load(ROOT_DICT)
print(f"  After load dict.txt        : {len(dict_infer)} symbols")
for lang in ckpt_langs:
    dict_infer.add_symbol(f"[{lang}]")
print(f"  After 19 lang tokens       : {len(dict_infer)} symbols")
dict_infer.add_symbol("<mask>")
print(f"  Inference dict final size  : {len(dict_infer)} symbols")

# ── comparison ────────────────────────────────────────────────────────────────
print("\n── Symbol-by-symbol comparison ──")
mm_gt_train  = compare_dicts(gt_dict,    dict_train)
mm_gt_infer  = compare_dicts(gt_dict,    dict_infer)
mm_train_inf = compare_dicts(dict_train, dict_infer)

sizes_ok = (len(gt_dict) == len(dict_train) == len(dict_infer) == 1024)
syms_ok  = (len(mm_gt_train) == 0 and
            len(mm_gt_infer) == 0 and
            len(mm_train_inf) == 0)

if sizes_ok and syms_ok:
    print(f"  ✓ ALL THREE DICTS IDENTICAL — each has {len(gt_dict)} symbols")
else:
    print(f"  Sizes: gt={len(gt_dict)}, train={len(dict_train)}, infer={len(dict_infer)}")
    for label, mm in [("gt vs train",   mm_gt_train),
                      ("gt vs infer",   mm_gt_infer),
                      ("train vs infer", mm_train_inf)]:
        if mm:
            print(f"\n  {label} — {len(mm)} mismatch(es):")
            for idx, sa, sb in mm[:30]:
                print(f"    [{idx:4d}]  {sa!r}  vs  {sb!r}")


# ─────────────────────────────────────────────────────────────────────────────
# PART 3 — Model load test
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("PART 3 — FINETUNE-FROM-MODEL LOAD TEST")
print(SEP)

# Load checkpoint with data pointing to repo root (where dict.txt lives)
# and all 19 langs from the original args → vocab = 1024.
# This mirrors what run_training does (vocab matches because dict_full.txt
# pre-bakes the same 19 lang tokens + mask).
state = load_checkpoint_to_cpu(
    CKPT,
    arg_overrides={"user_dir": USER_DIR, "data": REPO_ROOT},
)

if "args" in state and state["args"] is not None:
    cfg = convert_namespace_to_omegaconf(state["args"])
elif "cfg" in state and state["cfg"] is not None:
    cfg = state["cfg"]
else:
    raise RuntimeError(f"No args/cfg in checkpoint. Keys: {list(state.keys())}")

print(f"cfg.model._name         = {cfg.model._name!r}")
print(f"cfg.task.langs          = {cfg.task.langs!r}")
print(f"cfg.task.add_lang_token = {cfg.task.add_lang_token!r}")
print(f"cfg.task.data (patched) = {cfg.task.data!r}")

task = tasks.setup_task(cfg.task)
print(f"\nTask type           : {type(task).__name__}")
print(f"Task dict size      : {len(task.dictionary)}")
print(f"Task dict last 5    :")
for idx, sym in dict_tail(task.dictionary, 5):
    print(f"  [{idx}] {sym}")

model = task.build_model(cfg.model)
print(f"\nNew model type      : {type(model).__name__}")

enc_pos = model.encoder.embed_positions
dec_pos = model.decoder.embed_positions
print(f"encoder.embed_positions : {type(enc_pos).__name__}")
print(f"decoder.embed_positions : {type(dec_pos).__name__}")

ck_enc   = state["model"]["encoder.embed_tokens.weight"]
new_enc  = model.encoder.embed_tokens.weight
ck_dec   = state["model"]["decoder.embed_tokens.weight"]
new_dec  = model.decoder.embed_tokens.weight
print(f"\nCheckpoint  encoder.embed_tokens.weight : {list(ck_enc.shape)}")
print(f"New model   encoder.embed_tokens.weight : {list(new_enc.shape)}")
print(f"Shape match encoder : {'✓' if ck_enc.shape == new_enc.shape else '✗ MISMATCH'}")
print(f"Checkpoint  decoder.embed_tokens.weight : {list(ck_dec.shape)}")
print(f"New model   decoder.embed_tokens.weight : {list(new_dec.shape)}")
print(f"Shape match decoder : {'✓' if ck_dec.shape == new_dec.shape else '✗ MISMATCH'}")

print("\nRunning model.load_state_dict(strict=True) ...")
try:
    result = model.load_state_dict(state["model"], strict=True, model_cfg=cfg.model)
    missing    = list(getattr(result, "missing_keys",    []))
    unexpected = list(getattr(result, "unexpected_keys", []))
    print(f"  missing_keys    ({len(missing)})    : {missing[:10]}")
    print(f"  unexpected_keys ({len(unexpected)}) : {unexpected[:10]}")
    if not missing and not unexpected:
        print("  ✓ ZERO missing, ZERO unexpected — checkpoint loads cleanly")
    else:
        print("  ✗ Keys mismatch — investigate before training")
except RuntimeError as e:
    print(f"  ✗ RuntimeError: {e}")

print()
print(SEP)
print("VERIFICATION COMPLETE")
print(SEP)
