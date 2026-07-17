# yolo26-extralarge.py - A Python script to convert YOLO model to ONNX format
#                        for Object Detector
#
# Copyright (c) 2026 Ercan Ersoy.
# This file is licensed under the MIT License.
# Written by Ercan Ersoy helped by GitHub Copilot Claude Haiku 4.5 and
# Claude Opus 4.8.

import os
import shutil
import urllib.request
import urllib.error

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
                                    saved. If not specified, defaults to the
                                    ../binaries directory.
    """

    # Default output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "..", "binaries")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load model using ultralytics
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    # Convert to ONNX format
    print("Converting to ONNX format...")
    try:
        # Ultralytics writes the .onnx file next to the source .pt file and
        # returns its path, so it is moved into output_dir afterwards.
        export_result = str(model.export(format="onnx", imgsz=640))

        target_path = os.path.join(output_dir,
                                   os.path.basename(export_result))
        if os.path.abspath(export_result) != os.path.abspath(target_path):
            if os.path.exists(target_path):
                os.remove(target_path)
            shutil.move(export_result, target_path)

        print(f"✓ Successfully converted: {target_path}")

        # Display file size
        if os.path.exists(target_path):
            size_mb = os.path.getsize(target_path) / (1024 * 1024)
            print(f"File size: {size_mb:.2f} MB")

    except Exception as e:
        print(f"✗ Conversion error: {str(e)}")
        raise


if __name__ == "__main__":
    # Model filename and URL
    model_filename = "yolo26x.pt"
    model_url = ("https://github.com/ultralytics/assets/releases/download/"
                 "v8.4.0/yolo26x.pt")

    # Resolve directories relative to this script so the build works no matter
    # what the current working directory is. The .pt file is kept next to this
    # script in the model directory; the .onnx file is written to binaries.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bin_dir = os.path.join(script_dir, "..", "binaries")

    # Full path to model file
    model_path = os.path.join(script_dir, model_filename)

    # Download model
    print("=" * 50)
    print("YOLO Model Download and Conversion")
    print("=" * 50)

    if download_model(model_url, model_path):
        print("\n" + "=" * 50)
        print("ONNX Format Conversion")
        print("=" * 50)

        convert_yolo_to_onnx(
            model_path=model_path,
            output_dir=bin_dir
        )

    else:
        print("Model could not be downloaded, conversion cancelled.")
