"""Synthetic Image Processing Test.

Validates that NumPy, OpenCV, and Pillow can manipulate, process,
and save images locally without downloading any external or real person data.
"""

import os
import sys
import numpy as np
from PIL import Image
import cv2


def run_test() -> bool:
    print("=" * 60)
    print("Running Synthetic Image Processing Test")
    print("=" * 60)

    # 1. Create a synthetic RGB image with NumPy (128x128 pixels, 3 channels)
    height, width = 128, 128
    print(f"\n[1/5] Creating synthetic RGB array ({width}x{height} pixels)...")

    # Generate a smooth gradient pattern with colored blocks
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)
    xx, yy = np.meshgrid(x, y)

    # Combine channels: R = horizontal gradient, G = vertical gradient, B = diagonal blend
    r_channel = xx
    g_channel = yy
    b_channel = ((xx.astype(np.uint16) + yy.astype(np.uint16)) // 2).astype(np.uint8)
    synthetic_rgb = np.stack([r_channel, g_channel, b_channel], axis=-1)

    # 2. Verify dimensions and data type
    print("\n[2/5] Verifying synthetic image array properties...")
    assert synthetic_rgb.shape == (128, 128, 3), f"Unexpected shape: {synthetic_rgb.shape}"
    assert synthetic_rgb.dtype == np.uint8, f"Unexpected dtype: {synthetic_rgb.dtype}"
    print(f"  ✓ Shape: {synthetic_rgb.shape}, Dtype: {synthetic_rgb.dtype}")

    # 3. Apply a basic image-processing operation using OpenCV (Gaussian Blur + Edge detection or Color Conversion)
    print("\n[3/5] Applying image processing operations (Gaussian Blur & BGR conversion)...")
    # OpenCV expects BGR order for disk saving or conversions
    synthetic_bgr = cv2.cvtColor(synthetic_rgb, cv2.COLOR_RGB2BGR)
    blurred_bgr = cv2.GaussianBlur(synthetic_bgr, (7, 7), sigmaX=1.5)
    # Draw a simple test rectangle in the center to test drawing primitives
    cv2.rectangle(blurred_bgr, (32, 32), (96, 96), (255, 255, 255), 2)
    print("  ✓ OpenCV transformations applied successfully.")

    # 4. Save output to results/environment_test.png
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, "environment_test.png")

    print(f"\n[4/5] Saving processed image to:\n  {output_path}")
    save_success = cv2.imwrite(output_path, blurred_bgr)
    if not save_success:
        print("  ✗ cv2.imwrite failed to save the image.")
        return False
    print("  ✓ Image file written to disk.")

    # 5. Verify the saved file exists, has non-zero size, and can be reopened with Pillow
    print("\n[5/5] Re-opening saved image to verify file integrity...")
    if not os.path.exists(output_path):
        print(f"  ✗ Output file does not exist at {output_path}")
        return False

    file_size_bytes = os.path.getsize(output_path)
    if file_size_bytes == 0:
        print("  ✗ Output file is empty (0 bytes).")
        return False
    print(f"  ✓ File size: {file_size_bytes} bytes")

    with Image.open(output_path) as img:
        img_format = img.format
        img_size = img.size
        img_mode = img.mode
        print(f"  ✓ Image opened successfully with Pillow: Format={img_format}, Size={img_size}, Mode={img_mode}")

    print("\n" + "=" * 60)
    print("Image Processing Test PASSED successfully!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)

