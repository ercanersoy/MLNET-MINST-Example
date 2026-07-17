@echo off
REM clean.bat - Clean script for Object Detector
REM
REM Copyright (c) 2026 Ercan Ersoy.
REM This file is licensed under the MIT License.
REM Written by Ercan Ersoy helped by Claude Opus 4.8.
REM
REM Removes everything the build steps produce: the binaries directory (the
REM compiled application, the downloaded ONNX Runtime assemblies and the
REM downloaded and exported model files) and the downloaded NuGet package cache
REM under sources. The tracked source files are left untouched.

setlocal

set "ROOT=%~dp0"
set "BINARIES=%ROOT%binaries"
set "PACKAGES=%ROOT%sources\packages"

REM --- Remove the build output and downloaded dependencies -------------------
if exist "%BINARIES%" (
    echo Removing "%BINARIES%"...
    rmdir /s /q "%BINARIES%"
) else (
    echo Not found, skipping: "%BINARIES%"
)

REM --- Remove the downloaded NuGet package cache -----------------------------
if exist "%PACKAGES%" (
    echo Removing "%PACKAGES%"...
    rmdir /s /q "%PACKAGES%"
) else (
    echo Not found, skipping: "%PACKAGES%"
)

REM --- Remove the downloaded model as .pt file -------------------------------
del model\yolo26x.pt

echo.
echo Clean complete.
endlocal
