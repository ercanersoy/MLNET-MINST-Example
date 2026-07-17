# fetch-deps.ps1 - Downloads the managed and native dependencies required by
#                  Object Detector (ONNX Runtime and its support assemblies)
#                  from nuget.org and copies them next to the executable.
#
# Copyright (c) 2026 Ercan Ersoy.
# This file is licensed under the MIT License.
# Written by Ercan Ersoy helped by Claude Opus 4.8.

param(
    [Parameter(Mandatory = $true)] [string] $OutDir,
    [Parameter(Mandatory = $true)] [string] $CacheDir
)

$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

# Package id / version pairs to download.
$packages = @(
    @{ Id = 'Microsoft.ML.OnnxRuntime';                    Version = '1.20.1' },
    @{ Id = 'Microsoft.ML.OnnxRuntime.Managed';            Version = '1.20.1' },
    @{ Id = 'System.Memory';                               Version = '4.5.5'  },
    @{ Id = 'System.Buffers';                              Version = '4.5.1'  },
    @{ Id = 'System.Numerics.Vectors';                     Version = '4.5.0'  },
    @{ Id = 'System.Runtime.CompilerServices.Unsafe';      Version = '6.0.0'  }
)

# DLLs to place into the output directory. For each, an ordered list of
# preferred path fragments is used to pick the right target framework build.
$wanted = @(
    @{ Name = 'onnxruntime.dll';                             Prefer = @('runtimes/win-x64/native') },
    @{ Name = 'Microsoft.ML.OnnxRuntime.dll';                Prefer = @('netstandard2.0', 'netstandard2.1', 'netstandard1.1') },
    @{ Name = 'System.Memory.dll';                           Prefer = @('netstandard2.0', 'net461') },
    @{ Name = 'System.Buffers.dll';                          Prefer = @('netstandard2.0', 'net461') },
    @{ Name = 'System.Numerics.Vectors.dll';                 Prefer = @('netstandard2.0', 'net46') },
    @{ Name = 'System.Runtime.CompilerServices.Unsafe.dll';  Prefer = @('netstandard2.0', 'net461') }
)

New-Item -ItemType Directory -Force -Path $OutDir  | Out-Null
New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null

$extractRoot = Join-Path $CacheDir 'extracted'
New-Item -ItemType Directory -Force -Path $extractRoot | Out-Null

foreach ($pkg in $packages) {
    $idLower = $pkg.Id.ToLowerInvariant()
    $nupkgName = "$idLower.$($pkg.Version).nupkg"
    $nupkgPath = Join-Path $CacheDir $nupkgName
    $destDir = Join-Path $extractRoot "$idLower.$($pkg.Version)"

    if (-not (Test-Path $nupkgPath)) {
        $url = "https://api.nuget.org/v3-flatcontainer/$idLower/$($pkg.Version)/$nupkgName"
        Write-Host "Downloading $($pkg.Id) $($pkg.Version)..."
        Invoke-WebRequest -Uri $url -OutFile $nupkgPath -UseBasicParsing
    }

    if (Test-Path $destDir) {
        Remove-Item -Recurse -Force $destDir
    }
    # A .nupkg is a zip archive.
    $zipCopy = "$nupkgPath.zip"
    Copy-Item -Force $nupkgPath $zipCopy
    Expand-Archive -Path $zipCopy -DestinationPath $destDir -Force
    Remove-Item -Force $zipCopy
}

# Selects the best matching file from all extracted packages.
function Copy-Best($name, $preferFragments) {
    $matches = Get-ChildItem -Path $extractRoot -Recurse -Filter $name -ErrorAction SilentlyContinue
    if (-not $matches) {
        throw "Could not locate $name in the downloaded packages."
    }

    $chosen = $null
    foreach ($fragment in $preferFragments) {
        $frag = $fragment.Replace('/', [IO.Path]::DirectorySeparatorChar)
        $chosen = $matches | Where-Object { $_.FullName -like "*$frag*" } | Select-Object -First 1
        if ($chosen) { break }
    }
    if (-not $chosen) {
        $chosen = $matches | Select-Object -First 1
    }

    Copy-Item -Force $chosen.FullName (Join-Path $OutDir $name)
    Write-Host "  -> $name"
}

Write-Host 'Copying dependencies to the output directory...'
foreach ($item in $wanted) {
    Copy-Best $item.Name $item.Prefer
}

Write-Host 'Dependencies ready.'
