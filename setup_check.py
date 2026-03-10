"""
VeriVoice Sentinel — Environment Check
Run this first to verify everything is installed correctly.

Usage:  python setup_check.py
"""

import sys

PASS = "\033[92m  [OK]\033[0m"
FAIL = "\033[91m  [FAIL]\033[0m"


def check(label: str, fn) -> bool:
    try:
        result = fn()
        print(f"{PASS} {label}: {result}")
        return True
    except Exception as e:
        print(f"{FAIL} {label}: {e}")
        return False


print("\n" + "=" * 50)
print("  VeriVoice Sentinel — Environment Check")
print("=" * 50 + "\n")

results = []

results.append(check("Python", lambda: sys.version.split()[0]))

# Library imports
libs = [
    ("torch",        lambda: __import__("torch").__version__),
    ("torchaudio",   lambda: __import__("torchaudio").__version__),
    ("transformers", lambda: __import__("transformers").__version__),
    ("librosa",      lambda: __import__("librosa").__version__),
    ("streamlit",    lambda: __import__("streamlit").__version__),
    ("sklearn",      lambda: __import__("sklearn").__version__),
    ("numpy",        lambda: __import__("numpy").__version__),
    ("joblib",       lambda: __import__("joblib").__version__),
    ("soundfile",    lambda: __import__("soundfile").__version__),
]
for name, fn in libs:
    results.append(check(name, fn))

# CUDA
import torch
results.append(check("CUDA available", lambda: str(torch.cuda.is_available())))
if torch.cuda.is_available():
    results.append(check("GPU name",    lambda: torch.cuda.get_device_name(0)))
    results.append(check("VRAM (GB)",   lambda: f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f}"))

# Wav2Vec2 loads on GPU
def wav2vec_check():
    from transformers import Wav2Vec2Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)
    return f"loaded on {next(m.parameters()).device}"

results.append(check("Wav2Vec2 on GPU", wav2vec_check))

# Data folders
import os
results.append(check("data/real/ exists", lambda: f"{len(os.listdir('data/real'))} files" if os.path.isdir('data/real') else (_ for _ in ()).throw(FileNotFoundError("not found"))))
results.append(check("data/fake/ exists", lambda: f"{len(os.listdir('data/fake'))} files" if os.path.isdir('data/fake') else (_ for _ in ()).throw(FileNotFoundError("not found"))))

print()
passed = sum(results)
total  = len(results)
print(f"{'=' * 50}")
print(f"  {passed}/{total} checks passed")

if passed == total:
    print("  ✓ Ready — run: python extract_features.py")
else:
    print("  ✗ Fix the failed checks above first.")
print("=" * 50 + "\n")
