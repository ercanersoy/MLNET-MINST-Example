# Object Detector

A Windows Forms object detection (image classification) application written in
C#. It runs the Ultralytics YOLO26 model (exported to ONNX) on a live webcam
feed using Microsoft ONNX Runtime, shows the detected object with a confidence
value, keeps a history list, and can save snapshots and record video.

Copyright (c) 2026 Ercan Ersoy (https://ercanersoy.net)

This project is licensed under MIT License.

## Features

* Live object detection from a webcam with the detected class shown in the main window
* Object history list
* Snapshot capture in BMP, JPEG, TIFF and PNG formats
* Video recording (see the note about video formats below)
* Settings window (language, camera, mirror, brightness, contrast, detection and output settings)
* About window
* Languages: System default, Turkish, English (United Kingdom) and English (United States)

## Requirements

* 64 bit x86 CPU
* 4 GB RAM
* 500 MB free space
* Microsoft Windows 10 or later
* Microsoft .NET Framework 4.8.1
* Python 3.8 or later (only to build the model)
* A video capture device (webcam)
* An Internet connection (to download the model and build dependencies)

## Build

The build has two steps and produces its output in the `binaries` directory:

1. Build the model: run `.\build.bat` in the project root. This downloads the
   YOLO26 model and exports it to ONNX (`binaries\yolo26x-cls.onnx`).
2. Build the application: run `.\sources\build.bat`. This downloads ONNX Runtime
   from nuget.org and compiles `binaries\ObjectDetector.exe` for x86-64 and
   Microsoft .NET Framework 4.8.1 using the C# compiler that ships with the
   framework.

Run `binaries\ObjectDetector.exe` to start the application.

To remove the downloaded and compiled files (the `binaries` directory and the
downloaded NuGet package cache under `sources`), run `.\clean.bat` in the
project root.

## Notes about video recording

Webcam capture uses the Video for Windows (avicap32) API and video recording is
implemented from scratch, without any external library. Because no external
codec library is used, only the **AVI** format (Motion-JPEG) can be produced
natively. The other formats (WMV, MOV, MKV, MPEG, MPEG-2, MPEG-4) are listed in
the settings but require an external codec that is not available under this
constraint; selecting them shows a message. Choose AVI to record.

## Notes

* Microsoft is registered trademark of Microsoft Corporation.
* Ultralytics and YOLO are registered trademarks of Ultralytics Incorporated.
* ONNX Runtime is a project of Microsoft, licensed under the MIT License. It is downloaded from nuget.org during the application build.
* This project is under development.
