"""Environment Verification Script for Aerial-Ground Person ReID.

This script verifies:
1. Python interpreter and version.
2. Core dependencies (PyTorch, torchvision, NumPy, OpenCV, etc.).
3. Apple Silicon Metal Performance Shaders (MPS) hardware acceleration.
4. Basic CPU and MPS tensor operations.
"""

import os
import sys

# Set matplotlib cache directory to /tmp if default is not writable
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def main() -> int:
    print("=" * 60)
    print("Environment Verification: Aerial-Ground Person ReID")
    print("=" * 60)

    # 1. Check Python version
    print(f"\n[1/6] Python Version: {sys.version}")
    if sys.version_info < (3, 9):
        print("ERROR: Python 3.9 or higher is required.")
        return 1

    # 2. Check Core Imports
    print("\n[2/6] Checking required library imports...")
    try:
        import numpy as np
        print(f"  ✓ NumPy: {np.__version__}")
    except ImportError as e:
        print(f"  ✗ Failed to import NumPy: {e}")
        return 1

    try:
        import PIL
        print(f"  ✓ Pillow (PIL): {PIL.__version__}")
    except ImportError as e:
        print(f"  ✗ Failed to import Pillow: {e}")
        return 1

    try:
        import cv2
        print(f"  ✓ OpenCV (cv2): {cv2.__version__}")
    except ImportError as e:
        print(f"  ✗ Failed to import OpenCV: {e}")
        return 1

    try:
        import torch
        print(f"  ✓ PyTorch: {torch.__version__}")
    except ImportError as e:
        print(f"  ✗ Failed to import PyTorch: {e}")
        return 1

    try:
        import torchvision
        print(f"  ✓ Torchvision: {torchvision.__version__}")
    except ImportError as e:
        print(f"  ✗ Failed to import Torchvision: {e}")
        return 1

    try:
        import sklearn
        print(f"  ✓ scikit-learn: {sklearn.__version__}")
    except ImportError as e:
        print(f"  ✗ Failed to import scikit-learn: {e}")
        return 1

    try:
        import matplotlib
        print(f"  ✓ Matplotlib: {matplotlib.__version__}")
    except ImportError as e:
        print(f"  ✗ Failed to import Matplotlib: {e}")
        return 1

    try:
        import yaml
        print(f"  ✓ PyYAML: {yaml.__version__}")
    except ImportError as e:
        print(f"  ✗ Failed to import PyYAML: {e}")
        return 1

    try:
        import tqdm
        print(f"  ✓ tqdm: {tqdm.__version__}")
    except ImportError as e:
        print(f"  ✗ Failed to import tqdm: {e}")
        return 1

    # 3. Check Hardware Acceleration (MPS and CUDA)
    print("\n[3/6] Checking Hardware Acceleration Devices...")

    # CUDA Check (Expected false on Mac, which is completely normal)
    cuda_available = torch.cuda.is_available()
    print(f"  - CUDA Available: {cuda_available} (Mac uses MPS instead of CUDA)")

    # MPS Check (Apple Silicon Metal Performance Shaders)
    mps_built = torch.backends.mps.is_built()
    mps_available = torch.backends.mps.is_available()
    print(f"  - PyTorch MPS Backend Built: {mps_built}")
    print(f"  - PyTorch MPS Available: {mps_available}")

    # 4. Perform Basic CPU Tensor Computation
    print("\n[4/6] Testing CPU Tensor Math...")
    try:
        x_cpu = torch.randn(3, 3)
        y_cpu = torch.matmul(x_cpu, x_cpu)
        print(f"  ✓ CPU Tensor Multiplication succeeded (output shape: {list(y_cpu.shape)})")
    except Exception as e:
        print(f"  ✗ CPU Tensor operation failed: {e}")
        return 1

    # 5. Perform MPS Hardware Tensor Computation (if supported)
    print("\n[5/6] Testing MPS (Apple Silicon GPU) Tensor Math...")
    if mps_available:
        try:
            device = torch.device("mps")
            x_mps = torch.randn(4, 4, device=device)
            y_mps = torch.matmul(x_mps, x_mps)
            # Ensure synchronization / completion
            torch.mps.synchronize()
            print(f"  ✓ MPS Tensor Multiplication succeeded on device '{device}' (shape: {list(y_mps.shape)})")
        except Exception as e:
            print(f"  ✗ MPS Tensor operation failed: {e}")
            return 1
    else:
        print("  - MPS not available or not supported on this device/configuration.")
        print("    PyTorch operations will run using CPU.")

    # 6. Overall Summary
    print("\n[6/6] Summary:")
    print("  ✓ All required libraries are installed and importable.")
    print(f"  ✓ Primary PyTorch device: {'mps (Apple Silicon GPU)' if mps_available else 'cpu'}")
    print("\nEnvironment is ready for Step 2.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
