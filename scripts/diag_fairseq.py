"""
Minimal fairseq import diagnostic.
Run: python scripts/diag_fairseq.py
"""
import os, sys, subprocess, traceback

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FSQ  = os.path.join(REPO, "fairseq")   # av2av/fairseq/ — contains fairseq/fairseq/

if os.environ.get("_DIAG_OK") != "1":
    env = dict(os.environ)
    env["PYTHONPATH"] = FSQ + os.pathsep + env.get("PYTHONPATH", "")
    env["_DIAG_OK"] = "1"
    sys.exit(subprocess.run([sys.executable] + sys.argv, env=env).returncode)

print("=== ENV ===")
print(f"PYTHONPATH = {os.environ.get('PYTHONPATH','<not set>')}")
print(f"sys.version = {sys.version}")
print()

print("=== sys.path (first 10) ===")
for i, p in enumerate(sys.path[:10]):
    print(f"  [{i}] {p}")
print()

print("=== find fairseq on disk ===")
import importlib.util
spec = importlib.util.find_spec("fairseq")
print(f"  importlib.util.find_spec('fairseq') = {spec}")
if spec:
    print(f"  origin  = {spec.origin}")
    print(f"  submodule_search_locations = {list(spec.submodule_search_locations or [])}")
print()

print("=== import fairseq (with full traceback) ===")
try:
    import fairseq
    print(f"  fairseq.__file__ = {fairseq.__file__}")
    print(f"  fairseq.__path__ = {list(fairseq.__path__)}")
except Exception:
    traceback.print_exc()
print()

print("=== find fairseq.data on disk ===")
spec2 = importlib.util.find_spec("fairseq.data")
print(f"  importlib.util.find_spec('fairseq.data') = {spec2}")
print()

print("=== import fairseq.data (with full traceback) ===")
try:
    import fairseq.data
    print(f"  OK — fairseq.data.__file__ = {fairseq.data.__file__}")
except Exception:
    traceback.print_exc()
print()

print("=== import Dictionary directly (with full traceback) ===")
try:
    from fairseq.data import Dictionary
    print(f"  OK — {Dictionary}")
except Exception:
    traceback.print_exc()
