REM build.bat - Build script for Object Detector
REM
REM Copyright (c) 2026, Ercan Ersoy.
REM This file is licensed under the MIT License.
REM Written by Ercan Ersoy helped by Claude Opus 4.8.

mkdir "binaries"

python "model\\yolo26-extralarge.py"

call "sources\build.bat"
