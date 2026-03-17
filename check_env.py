"""
Run this first to verify all dependencies are installed correctly.
  python check_env.py
"""
import sys
print(f"Python {sys.version}\n")

checks = [
    ("flask",        "Flask"),
    ("torch",        "PyTorch"),
    ("transformers", "Transformers"),
    ("peft",         "PEFT"),
    ("captum",       "Captum"),
    ("numpy",        "NumPy"),
]

all_ok = True
for mod, name in checks:
    try:
        m = __import__(mod)
        ver = getattr(m, '__version__', '?')
        print(f"  ✅  {name:<18} {ver}")
    except ImportError:
        print(f"  ❌  {name:<18} NOT INSTALLED  →  pip install {mod}")
        all_ok = False

print()
if all_ok:
    # Quick CUDA check
    import torch
    cuda = torch.cuda.is_available()
    dev  = torch.cuda.get_device_name(0) if cuda else "CPU only"
    print(f"  Device : {dev}")
    print("\n✅ All dependencies satisfied. Run: python app.py")
else:
    print("❌ Some dependencies are missing. Install them then re-run.")