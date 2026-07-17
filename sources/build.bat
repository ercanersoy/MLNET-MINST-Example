@echo off
REM build.bat - Build script for the Object Detector C# WinForms application
REM
REM Copyright (c) 2026 Ercan Ersoy.
REM This file is licensed under the MIT License.
REM Written by Ercan Ersoy helped by Claude Opus 4.8.
REM
REM Compiles the application for x86-64 and Microsoft .NET Framework 4.8.1 using
REM the C# compiler that ships with the framework, and writes the executable to
REM the binaries directory. ONNX Runtime and its support assemblies are fetched
REM from nuget.org into the same directory.

setlocal

set "SRC=%~dp0"
set "ROOT=%SRC%.."
set "OUT=%ROOT%\binaries"
set "PKG=%SRC%packages"
set "EXE=%OUT%\ObjectDetector.exe"

if not exist "%OUT%" mkdir "%OUT%"

REM --- Locate the .NET Framework C# compiler (64-bit) ------------------------
set "CSC=%WINDIR%\Microsoft.NET\Framework64\v4.0.30319\csc.exe"
if not exist "%CSC%" (
    echo Error: C# compiler not found at "%CSC%".
    echo Please install Microsoft .NET Framework 4.8.1.
    exit /b 1
)

REM --- Fetch native and managed dependencies --------------------------------
echo Fetching dependencies...
powershell -NoProfile -ExecutionPolicy Bypass -File "%SRC%fetch-deps.ps1" -OutDir "%OUT%" -CacheDir "%PKG%"
if errorlevel 1 (
    echo Error: failed to fetch dependencies.
    exit /b 1
)

REM --- Write the application configuration (assembly binding redirects) ------
echo Writing application configuration...
(
echo ^<?xml version="1.0" encoding="utf-8"?^>
echo ^<configuration^>
echo   ^<startup^>
echo     ^<supportedRuntime version="v4.0" sku=".NETFramework,Version=v4.8.1" /^>
echo   ^</startup^>
echo   ^<runtime^>
echo     ^<assemblyBinding xmlns="urn:schemas-microsoft-com:asm.v1"^>
echo       ^<dependentAssembly^>
echo         ^<assemblyIdentity name="System.Runtime.CompilerServices.Unsafe" publicKeyToken="b03f5f7f11d50a3a" culture="neutral" /^>
echo         ^<bindingRedirect oldVersion="0.0.0.0-6.0.0.0" newVersion="6.0.0.0" /^>
echo       ^</dependentAssembly^>
echo       ^<dependentAssembly^>
echo         ^<assemblyIdentity name="System.Memory" publicKeyToken="cc7b13ffcd2ddd51" culture="neutral" /^>
echo         ^<bindingRedirect oldVersion="0.0.0.0-4.0.1.2" newVersion="4.0.1.2" /^>
echo       ^</dependentAssembly^>
echo       ^<dependentAssembly^>
echo         ^<assemblyIdentity name="System.Buffers" publicKeyToken="cc7b13ffcd2ddd51" culture="neutral" /^>
echo         ^<bindingRedirect oldVersion="0.0.0.0-4.0.3.0" newVersion="4.0.3.0" /^>
echo       ^</dependentAssembly^>
echo       ^<dependentAssembly^>
echo         ^<assemblyIdentity name="System.Numerics.Vectors" publicKeyToken="b03f5f7f11d50a3a" culture="neutral" /^>
echo         ^<bindingRedirect oldVersion="0.0.0.0-4.1.4.0" newVersion="4.1.4.0" /^>
echo       ^</dependentAssembly^>
echo     ^</assemblyBinding^>
echo   ^</runtime^>
echo ^</configuration^>
) > "%EXE%.config"

REM --- Compile ---------------------------------------------------------------
echo Compiling...
"%CSC%" /nologo /target:winexe /platform:x64 /unsafe /optimize+ ^
    /out:"%EXE%" ^
    /reference:System.dll ^
    /reference:System.Core.dll ^
    /reference:System.Drawing.dll ^
    /reference:System.Windows.Forms.dll ^
    /reference:"%WINDIR%\Microsoft.NET\Framework64\v4.0.30319\netstandard.dll" ^
    /reference:"%OUT%\Microsoft.ML.OnnxRuntime.dll" ^
    /reference:"%OUT%\System.Memory.dll" ^
    /reference:"%OUT%\System.Numerics.Vectors.dll" ^
    "%SRC%Main.cs" ^
    "%SRC%AppSettings.cs" ^
    "%SRC%Localization.cs" ^
    "%SRC%OnnxClassifier.cs" ^
    "%SRC%WebcamCapture.cs" ^
    "%SRC%AviRecorder.cs" ^
    "%SRC%SettingsForm.cs" ^
    "%SRC%AboutForm.cs"

if errorlevel 1 (
    echo.
    echo Build failed.
    exit /b 1
)

echo.
echo Build succeeded: "%EXE%"
endlocal
