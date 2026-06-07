# main.py  —  Environment sanity check
import sys
import os
import torch
import numpy as np
import cv2
from PIL import Image

# Fix: helpers.py lives in the same directory, not utils/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_environment():
    print("=" * 45)
    print("  Dew-Based Federated Learning — Env Check")
    print("=" * 45)
    print(f"  Python      : {sys.version.split()[0]}")
    print(f"  PyTorch     : {torch.__version__}")
    print(f"  CUDA        : {'Available ✅ (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'Not available (CPU only)'}")
    print(f"  NumPy       : {np.__version__}")
    print(f"  OpenCV      : {cv2.__version__}")
    print(f"  Pillow      : {Image.__version__}")
    print("=" * 45)
    print("  Environment is ready! ✅")
    print("=" * 45)


if __name__ == "__main__":
    test_environment()

    # Fixed import — helpers.py is in the project root, not utils/
    from helpers import greet
    greet()
