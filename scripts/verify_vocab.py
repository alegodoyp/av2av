"""
Vocabulary verification script for utut_sts_ft.pt fine-tuning.

Run from the repository root:
    python scripts/verify_vocab.py

Prerequisites:
    pip install -e fairseq/ --no-deps   (run once after git submodule init)
"""

import os
import sys
import subprocess

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FSQ  = os.path.join(REPO, "fairseq")   # av2av/fairseq/  — pip install -e points here

# ── self-reinvoke so PYTHONPATH is set before Python's import machinery runs ──
# Running from av2av/ puts '' (= av2av/) in sys.path[0].  Python then finds
# the outer av2av/fairseq/ folder as a *namespace* package (no __init__.py
# there) before reaching the pip-installed regular package.  Setting PYTHONPATH
# before process start avoids this; the child sees the .pth path from pip
# install -e BEFORE the cwd namespace portion.
# NOTE: do NOT add REPO_ROOT to sys.path inside the script — that reintroduces
# the same namespace-package conflict.
if os.environ.get("_VOCAB_OK") != "1":
    env = dict(os.environ)
    env["PYTHONPATH"] = FSQ + os.pathsep + env.get("PYTHONPATH", "")
    env["_VOCAB_OK"]  = "1"
    sys.exit(subprocess.run([sys.executable] + sys.argv, env=env).returncode)
# ─────────────────────────────────────────────────────────────────────────────

import torch
from fairseq.data             import Dictionary
from fairseq.checkpoint_utils import load_checkpoint_to_cpu
from fairseq.dataclass.utils  import convert_namespace_to_omegaconf
from fairseq                  import tasks, utils

print(f"fairseq loaded from: {__import__('fairseq').__file__}")

CKPT      = os.path.join(REPO, "checkpoints", "utut_sts_ft.pt")
ROOT_DICT = os.path.join(REPO, "dict.txt")
FULL_DICT = os.path.join(REPO, "dict_full.txt")
USER_DIR  = os.path.join(REPO, "unit2unit")
SEP = "=" * 72


# ─────────────────────────────────────────────────────────────────────────────
def create_extended_dict(path):
    """Inline mirror of train_full_pipeline.create_extended_dict."""
    LANG_ORDER = ["en","es","fr","it","pt","el","ru","cs","da","de",
                  "fi","hr","hu","lt","nl","pl","ro","sk","sl"]
    with open(path, "w") as f:
        for i in range(1000):
            f.write(f"{i} 1\n")
        for lang in LANG_ORDER:
            f.write(f"[{lang}] 1\n")
        f.write("<mask> 1\n")
    return LANG_ORDER


def dict_tail(d, n=22):
    start = max(0, len(d) - n)
    return [(i, d[i]) for i in range(start, len(d))]


def compare_dicts(da, db):
    mm = []
    for i in range(max(len(da), len(db))):
        sa = da[i] if i < len(da) else "<MISSING>"
        sb = db[i] if i < len(db) else "<MISSING>"
        if sa != sb:
            mm.append((i, sa, sb))
    return mm


# ═════════════════════════════════════════════════════════════════════════════
# PART 1 — Ground truth from checkpoint
# ═════════════════════════════════════════════════════════════════════════════
print(SEP)
print("PART 1 — CHECKPOINT GROUND TRUTH")
print(SEP)

assert os.path.exists(CKPT), f"Checkpoint not found: {CKPT}"
ck = torch.load(CKPT, map_location="cpu", weights_only=False)

print("Checkpoint top-level keys:", list(ck.keys()))
print("task_state               :", ck.get("task_state", {}))
print()

for key in ["encoder.embed_tokens.weight",
            "decoder.embed_tokens.weight",
            "decoder.output_projection.weight"]:
    if key in ck["model"]:
        print(f"  ck['model'][{key!r}].shape = {list(ck['model'][key].shape)}")

# The checkpoint may store args as a Namespace (old format) or as cfg (new format).
raw_args = ck.get("args")
raw_cfg  = ck.get("cfg")
if raw_args is not None:
    arch           = raw_args.arch
    model_name     = getattr(raw_args, "_name", "<absent>")
    ckpt_langs_str = raw_args.langs
    add_lang_token = raw_args.add_lang_token
    ckpt_data      = raw_args.data
else:
    # raw_cfg is a plain Python dict (from torch.load directly).
    # raw_cfg["model"] is an argparse Namespace → attribute access only.
    # raw_cfg["task"]  is a plain dict → .get() works normally.
    _m = raw_cfg["model"]
    _t = raw_cfg["task"]
    arch           = getattr(_m, "_name", None) or getattr(_m, "arch", "?")
    model_name     = getattr(_m, "_name", "<absent>")
    ckpt_langs_str = _t.get("langs", "?")
    add_lang_token = _t.get("add_lang_token", "?")
    ckpt_data      = _t.get("data", "?")

print()
print(f"arch           = {arch!r}")
print(f"model _name    = {model_name!r}")
print(f"langs          = {ckpt_langs_str!r}")
print(f"add_lang_token = {add_lang_token!r}")
print(f"data           = {ckpt_data!r}")

# Reconstruct vocab exactly as the original setup_task built it:
#   dict.txt (1000 units) + 19 lang tokens (langs order) + <mask>
assert os.path.exists(ROOT_DICT), f"dict.txt not found: {ROOT_DICT}"
gt = Dictionary.load(ROOT_DICT)
print(f"\nAfter Dictionary.load(dict.txt) : {len(gt)} symbols")
ckpt_langs = ckpt_langs_str.split(",")
for lang in ckpt_langs:
    gt.add_symbol(f"[{lang}]")
print(f"After {len(ckpt_langs)} lang tokens          : {len(gt)} symbols")
gt.add_symbol("<mask>")
print(f"After <mask>                    : {len(gt)} symbols  ← GROUND TRUTH")

print(f"\nGround-truth vocab size = {len(gt)}")
print("Last 22 symbols:")
for idx, sym in dict_tail(gt):
    print(f"  [{idx:4d}]  {sym}")


# ═════════════════════════════════════════════════════════════════════════════
# PART 2 — Equivalence check
# ═════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("PART 2 — EQUIVALENCE CHECK")
print(SEP)

# ── validate dict_full.txt structure ─────────────────────────────────────────
print("\n── dict_full.txt structure ──")
LANG_ORDER = create_extended_dict(FULL_DICT)
with open(FULL_DICT) as fh:
    raw_lines = fh.readlines()

print(f"  Total lines  : {len(raw_lines)}  (expected 1020)")
print(f"  lines[0]     : {raw_lines[0].rstrip()!r}          ← first unit")
print(f"  lines[999]   : {raw_lines[999].rstrip()!r}        ← last unit")
print(f"  lines[1000]  : {raw_lines[1000].rstrip()!r}  ← first lang token [en]")
print(f"  lines[1018]  : {raw_lines[1018].rstrip()!r}  ← last lang token [sl]")
print(f"  lines[1019]  : {raw_lines[1019].rstrip()!r}     ← <mask>")
print(f"  {'✓' if len(raw_lines) == 1020 else '✗'} {len(raw_lines)} lines")

# ── (a) training dict ─────────────────────────────────────────────────────────
print("\n── (a) TRAINING DICT  [dict_full.txt + --add-lang-token --langs pt,en] ──")
dt = Dictionary.load(FULL_DICT)
print(f"  After load             : {len(dt)} symbols")
for lang in ["pt", "en"]:
    idx = dt.add_symbol(f"[{lang}]")
    print(f"  add_symbol('[{lang}]') → {idx}  "
          f"{'(pre-existing ✓)' if idx < len(dt) else '(NEW ✗)'}")
mi = dt.add_symbol("<mask>")
print(f"  add_symbol('<mask>') → {mi}  "
      f"{'(pre-existing ✓)' if mi == 1023 else '(NEW ✗)'}")
print(f"  Training dict size     : {len(dt)}")

# ── (b) inference dict ────────────────────────────────────────────────────────
print("\n── (b) INFERENCE DICT  [dict.txt + setup_task(19 langs) + mask] ──")
di = Dictionary.load(ROOT_DICT)
print(f"  After load dict.txt    : {len(di)} symbols")
for lang in ckpt_langs:
    di.add_symbol(f"[{lang}]")
print(f"  After 19 lang tokens   : {len(di)} symbols")
di.add_symbol("<mask>")
print(f"  Inference dict size    : {len(di)} symbols")

# ── symbol-by-symbol comparison ───────────────────────────────────────────────
print("\n── Symbol-by-symbol comparison ──")
mm_gt_t = compare_dicts(gt, dt)
mm_gt_i = compare_dicts(gt, di)
mm_t_i  = compare_dicts(dt, di)

ok = (len(mm_gt_t) == 0 and len(mm_gt_i) == 0 and len(mm_t_i) == 0
      and len(gt) == len(dt) == len(di) == 1024)

if ok:
    print(f"  ✓ ALL THREE DICTS IDENTICAL — each has 1024 symbols")
else:
    print(f"  ✗ MISMATCH  sizes: gt={len(gt)}, train={len(dt)}, infer={len(di)}")
    for label, mm in [("gt vs train",    mm_gt_t),
                      ("gt vs infer",    mm_gt_i),
                      ("train vs infer", mm_t_i)]:
        if mm:
            print(f"\n  {label} ({len(mm)} mismatch(es)):")
            for idx, sa, sb in mm[:30]:
                print(f"    [{idx:4d}]  {sa!r}  vs  {sb!r}")


# ═════════════════════════════════════════════════════════════════════════════
# PART 3 — Model load test  (strict=True, mbart_large + checkpoint weights)
# ═════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("PART 3 — FINETUNE-FROM-MODEL LOAD TEST")
print(SEP)

# Register UTUTPretrainingTask from unit2unit/ before calling setup_task.
# (In normal training, train.py calls import_user_module early; we do it here.)
utils.import_user_module(USER_DIR)

state = load_checkpoint_to_cpu(
    CKPT,
    arg_overrides={"user_dir": USER_DIR, "data": REPO},
)

if "args" in state and state["args"] is not None:
    cfg = convert_namespace_to_omegaconf(state["args"])
elif "cfg" in state and state["cfg"] is not None:
    cfg = state["cfg"]
else:
    raise RuntimeError(f"No args/cfg. Keys: {list(state.keys())}")

_m3 = cfg.model  # raw Namespace, returned from OmegaConf allow_objects node
print(f"cfg.model._name         = {getattr(_m3, '_name', None) or getattr(_m3, 'arch', '?')!r}")
print(f"cfg.task.langs          = {cfg.task.langs!r}")
print(f"cfg.task.add_lang_token = {cfg.task.add_lang_token!r}")
print(f"cfg.task.data (patched) = {cfg.task.data!r}")

task = tasks.setup_task(cfg.task)
print(f"\nTask type       : {type(task).__name__}")
print(f"Task dict size  : {len(task.dictionary)}")
print("Task dict last 5:")
for idx, sym in dict_tail(task.dictionary, 5):
    print(f"  [{idx}] {sym}")

model = task.build_model(cfg.model)
print(f"\nModel type      : {type(model).__name__}")

ep = model.encoder.embed_positions
dp = model.decoder.embed_positions
print(f"encoder.embed_positions : {type(ep).__name__}")
print(f"decoder.embed_positions : {type(dp).__name__}")

ck_e  = state["model"]["encoder.embed_tokens.weight"]
new_e = model.encoder.embed_tokens.weight
ck_d  = state["model"]["decoder.embed_tokens.weight"]
new_d = model.decoder.embed_tokens.weight
print(f"\nCheckpoint  encoder.embed_tokens : {list(ck_e.shape)}")
print(f"New model   encoder.embed_tokens : {list(new_e.shape)}  "
      f"{'✓' if ck_e.shape == new_e.shape else '✗ MISMATCH'}")
print(f"Checkpoint  decoder.embed_tokens : {list(ck_d.shape)}")
print(f"New model   decoder.embed_tokens : {list(new_d.shape)}  "
      f"{'✓' if ck_d.shape == new_d.shape else '✗ MISMATCH'}")

print("\nRunning model.load_state_dict(strict=True) ...")
try:
    res = model.load_state_dict(state["model"], strict=True, model_cfg=cfg.model)
    miss = list(getattr(res, "missing_keys",    []))
    unex = list(getattr(res, "unexpected_keys", []))
    print(f"  missing_keys    ({len(miss)})    : {miss[:10]}")
    print(f"  unexpected_keys ({len(unex)})    : {unex[:10]}")
    if not miss and not unex:
        print("  ✓ ZERO missing, ZERO unexpected — checkpoint loads cleanly")
    else:
        print("  ✗ Keys mismatch — fine-tuning will crash")
except RuntimeError as exc:
    print(f"  ✗ RuntimeError: {exc}")

print()
print(SEP)
print("VERIFICATION COMPLETE")
print(SEP)
