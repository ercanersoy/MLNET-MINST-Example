# convert.py - A Python script to convert YOLO model to ONNX format for Object Detector
#
# Copyright (c) 2026 Ercan Ersoy.
# This file is licensed under the MIT License.
# Written by Ercan Ersoy helped by GitHub Copilot and Claude Haiku 4.5.

import os
import urllib.request
import urllib.error

from pathlib import Path
from ultralytics import YOLO


def download_model(url, output_path):
    """
    Downloads a model from the given URL.

    Args:
        url (str): The download URL of the model
        output_path (str): The path where the downloaded file will be saved

    Returns:
        bool: True if download is successful, False otherwise
    """

    # Check if file already exists
    if os.path.exists(output_path):
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✓ File already exists: {output_path} ({file_size_mb:.2f} MB)")
        return True

    # Download file
    print(f"Downloading model: {url}")
    try:
        # Download progress function
        def download_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 / total_size, 100)
            msg = (f"\rDownload: {percent:.1f}% "
                   f"({downloaded / (1024*1024):.1f} MB / "
                   f"{total_size / (1024*1024):.1f} MB)")
            print(msg, end="")

        urllib.request.urlretrieve(url, output_path,
                                   reporthook=download_progress)
        print()  # New line

        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✓ Successfully downloaded: {output_path} "
              f"({file_size_mb:.2f} MB)")
        return True

    except urllib.error.URLError as e:
        print(f"\n✗ Download error: {str(e)}")
        return False
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        return False


def convert_yolo_to_onnx(model_path, output_dir=None):
    """
    Converts a YOLO model to ONNX format using Ultralytics export.

    Args:
        model_path (str): Path to the input .pt file
        output_dir (str, optional): Directory where the ONNX file will be
                                    saved. If not specified, defaults to
                                    ../bin directory.
    """

    # Default output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "..", "bin")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load model using ultralytics
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    # Convert to ONNX format
    print(f"Converting to ONNX format...")
    try:
        export_result = model.export(format="onnx", imgsz=640,
                                     project=output_dir, name="")
        print(f"✓ Successfully converted: {export_result}")

        # Display file size
        if os.path.exists(export_result):
            size_mb = os.path.getsize(export_result) / (1024 * 1024)
            print(f"File size: {size_mb:.2f} MB")

    except Exception as e:
        print(f"✗ Conversion error: {str(e)}")
        raise


if __name__ == "__main__":
    # Model filename and URL
    model_filename = "yolo26x-cls.pt"
    model_url = ("https://github.com/ultralytics/assets/releases/download/"
                 "v8.4.0/yolo26x-cls.pt")

    # Full path to model file
    current_dir = os.path.join(os.getcwd(), "binaries")
    model_path = os.path.join(current_dir, model_filename)

    # Download model
    print("=" * 50)
    print("YOLO Model Download and Conversion")
    print("=" * 50)

    if download_model(model_url, model_path):
        print("\n" + "=" * 50)
        print("ONNX Format Conversion")
        print("=" * 50)

        # Set output directory to ../binaries
        bin_dir = os.path.join(os.getcwd(), "binaries")
        convert_yolo_to_onnx(
            model_path=model_path,
            output_dir=bin_dir
        )

    else:
        print("Model could not be downloaded, conversion cancelled.")
