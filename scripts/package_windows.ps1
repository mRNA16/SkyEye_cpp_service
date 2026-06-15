param(
    [string]$OutputDir = "dist/pilot_windows",
    [string]$PreferredBuildDir = "",
    [string]$VcpkgRoot = "D:/vcpkg/installed/x64-windows",
    [string]$OrtDir = "D:/onnxruntime-win-x64-gpu-1.24.1",
    [string]$CudaRoot = $env:CUDA_PATH,
    [string]$CudnnRoot = "",
    [string]$MsvcRedistRoot = "",
    [string]$CmakeBuildPreset = "pilot-release",
    [switch]$BuildRelease,
    [switch]$AllowDebugBuild,
    [switch]$SkipMsvcRuntime,
    [switch]$SkipCudaRuntime
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$dist = Join-Path $root $OutputDir

if ($BuildRelease) {
    Write-Host "Building Release target with CMake preset: $CmakeBuildPreset"
    Push-Location $root
    try {
        & cmake --build --preset $CmakeBuildPreset
        if ($LASTEXITCODE -ne 0) {
            throw "CMake build failed with exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }
}

function Get-ExeRoot {
    param([string[]]$Candidates)
    foreach ($candidate in $Candidates) {
        if ([string]::IsNullOrWhiteSpace($candidate)) { continue }
        $path = Join-Path $root $candidate
        if (Test-Path (Join-Path $path "pilot.exe")) {
            return $path
        }
    }
    return $null
}

function Test-IsDebugDllName {
    param([string]$Name)

    return $Name -match "(?i)(vcruntime.*d\.dll$|msvcp.*d\.dll$|ucrtbased\.dll$|opencv_.*d\.dll$|python.*_d\.dll$|.*-d\.dll$|.*debug.*\.dll$)"
}

function Get-DebugDlls {
    param([string[]]$Directories)

    $matches = @()
    foreach ($directory in $Directories) {
        if (-not (Test-Path $directory)) { continue }
        $matches += Get-ChildItem -Path $directory -Filter *.dll -File -ErrorAction SilentlyContinue |
            Where-Object { Test-IsDebugDllName -Name $_.Name } |
            Select-Object -ExpandProperty FullName
    }

    return $matches | Sort-Object -Unique
}

$candidateRoots = @()
if (-not [string]::IsNullOrWhiteSpace($PreferredBuildDir)) {
    $candidateRoots += $PreferredBuildDir
}
$candidateRoots += @(
    "out\build\windows-release",
    "pilot\x64\Release",
    "out\build\windows-base"
)

if ($AllowDebugBuild) {
    $candidateRoots += "pilot\x64\Debug"
}

$buildRoot = Get-ExeRoot -Candidates $candidateRoots
if (-not $buildRoot) {
    throw "pilot.exe not found in known build folders. Use -PreferredBuildDir to point at the folder containing pilot.exe."
}

$debugDlls = Get-DebugDlls -Directories @($buildRoot) |
    Select-Object -First 8 |
    ForEach-Object { Split-Path $_ -Leaf }

if (($buildRoot -match "(?i)debug" -or $debugDlls) -and -not $AllowDebugBuild) {
    $debugHint = if ($debugDlls) { " Debug DLLs found: $($debugDlls -join ', ')." } else { "" }
    throw "Refusing to package a Debug build from '$buildRoot'.$debugHint Build Release first, or pass -AllowDebugBuild only for local development."
}

if ($AllowDebugBuild -and ($buildRoot -match "(?i)debug" -or $debugDlls)) {
    Write-Warning "Packaging a Debug build. Debug MSVC runtime DLLs are not redistributable; use a Release or RelWithDebInfo build for target machines without Visual Studio."
}

if (Test-Path $dist) {
    Remove-Item -Recurse -Force $dist
}

$dirs = @(
    $dist,
    (Join-Path $dist "bin"),
    (Join-Path $dist "runtime"),
    (Join-Path $dist "runtime\ffmpeg"),
    (Join-Path $dist "client"),
    (Join-Path $dist "models"),
    (Join-Path $dist "config"),
    (Join-Path $dist "output"),
    (Join-Path $dist "temp")
)

foreach ($dir in $dirs) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

Copy-Item -Force (Join-Path $buildRoot "pilot.exe") (Join-Path $dist "bin\pilot.exe")

function Copy-DllToPackage {
    param(
        [Parameter(Mandatory = $true)]
        [System.IO.FileInfo]$DllFile
    )

    $targets = @(
        (Join-Path $dist "runtime\$($DllFile.Name)"),
        (Join-Path $dist "bin\$($DllFile.Name)")
    )

    foreach ($target in $targets) {
        Copy-Item -Force $DllFile.FullName $target
    }
}

function Copy-DllsFromDirectory {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Directory,
        [string[]]$Patterns = @("*.dll")
    )

    if (-not (Test-Path $Directory)) {
        return 0
    }

    $seen = @{}
    $count = 0
    foreach ($pattern in $Patterns) {
        foreach ($dll in Get-ChildItem -Path $Directory -Filter $pattern -File -ErrorAction SilentlyContinue) {
            if ($seen.ContainsKey($dll.FullName)) { continue }
            $seen[$dll.FullName] = $true
            Copy-DllToPackage -DllFile $dll
            $count++
        }
    }
    return $count
}

function Get-VisualStudioInstallPath {
    $programFilesX86 = [Environment]::GetFolderPath("ProgramFilesX86")
    $programFiles = [Environment]::GetFolderPath("ProgramFiles")
    $vswhereCandidates = @(
        (Join-Path $programFilesX86 "Microsoft Visual Studio\Installer\vswhere.exe"),
        (Join-Path $programFiles "Microsoft Visual Studio\Installer\vswhere.exe")
    )

    foreach ($vswhere in $vswhereCandidates) {
        if (-not (Test-Path $vswhere)) { continue }
        $installPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if (-not [string]::IsNullOrWhiteSpace($installPath) -and (Test-Path $installPath)) {
            return $installPath.Trim()
        }
    }

    return $null
}

function Get-MsvcRuntimeDirectories {
    param([string]$RedistRoot)

    $roots = @()
    if (-not [string]::IsNullOrWhiteSpace($RedistRoot)) {
        $roots += $RedistRoot
    }
    if (-not [string]::IsNullOrWhiteSpace($env:VCToolsRedistDir)) {
        $roots += $env:VCToolsRedistDir
    }

    $vsInstallPath = Get-VisualStudioInstallPath
    if ($vsInstallPath) {
        $roots += (Join-Path $vsInstallPath "VC\Redist\MSVC")
    }

    $runtimeDirs = @()
    foreach ($root in ($roots | Where-Object { $_ } | Sort-Object -Unique)) {
        if (-not (Test-Path $root)) { continue }

        if (Test-Path (Join-Path $root "vcruntime140.dll")) {
            $runtimeDirs += $root
            continue
        }

        $crtDir = Get-ChildItem -Path $root -Directory -Recurse -ErrorAction SilentlyContinue |
            Where-Object {
                $_.FullName -match "\\x64\\Microsoft\.VC\d+\.CRT$" -and
                $_.FullName -notmatch "debug_nonredist" -and
                $_.FullName -notmatch "\\onecore\\"
            } |
            Sort-Object FullName -Descending |
            Select-Object -First 1

        if ($crtDir) {
            $archDir = Split-Path $crtDir.FullName -Parent
            $runtimeDirs += Get-ChildItem -Path $archDir -Directory -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -match "^Microsoft\.VC\d+\.(CRT|OpenMP|CXXAMP)$" } |
                Select-Object -ExpandProperty FullName
        }
    }

    return $runtimeDirs | Where-Object { $_ } | Sort-Object -Unique
}

foreach ($dll in Get-ChildItem -Path $buildRoot -Filter *.dll -File -ErrorAction SilentlyContinue) {
    Copy-DllToPackage -DllFile $dll
}

$extraDllRoots = @(
    (Join-Path $VcpkgRoot "bin"),
    (Join-Path $OrtDir "lib")
)

if ($AllowDebugBuild) {
    $extraDllRoots += (Join-Path $VcpkgRoot "debug\bin")
}
foreach ($extraRoot in $extraDllRoots) {
    if (-not (Test-Path $extraRoot)) { continue }
    foreach ($dll in Get-ChildItem -Path $extraRoot -Filter *.dll -File -ErrorAction SilentlyContinue) {
        $runtimeDest = Join-Path $dist "runtime\$($dll.Name)"
        $binDest = Join-Path $dist "bin\$($dll.Name)"
        if (-not (Test-Path $runtimeDest) -or -not (Test-Path $binDest)) {
            Copy-DllToPackage -DllFile $dll
        }
    }
}

if (-not $SkipMsvcRuntime) {
    $msvcDllCount = 0
    foreach ($runtimeDir in Get-MsvcRuntimeDirectories -RedistRoot $MsvcRedistRoot) {
        $msvcDllCount += Copy-DllsFromDirectory -Directory $runtimeDir
    }
    if ($msvcDllCount -gt 0) {
        Write-Host "MSVC runtime DLLs copied: $msvcDllCount"
    } else {
        Write-Warning "MSVC runtime DLLs were not copied. Pass -MsvcRedistRoot or install VC++ Redistributable on the target machine."
    }
}

if (-not $SkipCudaRuntime) {
    $cudaDllPatterns = @(
        "cudart64_*.dll",
        "cublas64_*.dll",
        "cublasLt64_*.dll",
        "cufft64_*.dll",
        "curand64_*.dll",
        "cusolver64_*.dll",
        "cusparse64_*.dll",
        "nvJitLink_*.dll",
        "nvrtc64_*.dll",
        "nvrtc-builtins64_*.dll",
        "nvjpeg64_*.dll",
        "cudnn*.dll",
        "zlibwapi.dll"
    )

    $cudaRoots = @()
    if (-not [string]::IsNullOrWhiteSpace($CudaRoot)) {
        $cudaRoots += (Join-Path $CudaRoot "bin")
    }
    if (-not [string]::IsNullOrWhiteSpace($env:CUDA_PATH)) {
        $cudaRoots += (Join-Path $env:CUDA_PATH "bin")
    }

    $cudaDllCount = 0
    foreach ($cudaBin in ($cudaRoots | Where-Object { $_ } | Sort-Object -Unique)) {
        $cudaDllCount += Copy-DllsFromDirectory -Directory $cudaBin -Patterns $cudaDllPatterns
    }

    if (-not (Test-Path (Join-Path $dist "bin\cudnn*.dll"))) {
        $cudnnRoots = @()
        if (-not [string]::IsNullOrWhiteSpace($CudnnRoot)) {
            $cudnnRoots += $CudnnRoot
        }
        $programFiles = [Environment]::GetFolderPath("ProgramFiles")
        $commonCudnnRoot = Join-Path $programFiles "NVIDIA\CUDNN"
        if (Test-Path $commonCudnnRoot) {
            $cudnnRoots += Get-ChildItem -Path $commonCudnnRoot -Directory -Recurse -ErrorAction SilentlyContinue |
                Where-Object { $_.FullName -match "\\bin\\[^\\]+\\x64$" } |
                Sort-Object FullName -Descending |
                Select-Object -ExpandProperty FullName
        }

        foreach ($cudnnBin in ($cudnnRoots | Where-Object { $_ } | Sort-Object -Unique)) {
            $cudaDllCount += Copy-DllsFromDirectory -Directory $cudnnBin -Patterns @("cudnn*.dll")
            if (Test-Path (Join-Path $dist "bin\cudnn*.dll")) { break }
        }
    }

    if ($cudaDllCount -gt 0) {
        Write-Host "CUDA/cuDNN runtime DLLs copied: $cudaDllCount"
    } else {
        Write-Warning "CUDA/cuDNN runtime DLLs were not copied. Pass -CudaRoot/-CudnnRoot or install compatible CUDA/cuDNN runtime on the target machine."
    }
}

$models = @(
    [PSCustomObject]@{
        Name = "I3D feature extractor"
        Source = "i3d\models\a320_new_full.onnx"
        Target = "models\a320_new_full.onnx"
        Note = "Updated from final_models\特征提取"
    },
    [PSCustomObject]@{
        Name = "TriDet temporal detector"
        Source = "algos\tridet_a320.onnx"
        Target = "models\tridet_a320.onnx"
        Note = "Updated from final_models\时序检测, 25 classes with per-class thresholds in service"
    },
    [PSCustomObject]@{
        Name = "YOLO object detector"
        Source = "yolo\config\best.onnx"
        Target = "models\best.onnx"
        Note = "Kept unchanged by request"
    }
)

$modelManifest = @(
    "Pilot model manifest",
    "Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')",
    "Package: $dist",
    ""
)

foreach ($model in $models) {
    $src = Join-Path $root $model.Source
    $dst = Join-Path $dist $model.Target
    if (-not (Test-Path $src)) {
        throw "Model not found: $src"
    }
    Copy-Item -Force $src $dst

    $item = Get-Item $src
    $modelManifest += @(
        "[$($model.Name)]",
        "source=$($model.Source)",
        "target=$($model.Target)",
        "bytes=$($item.Length)",
        "last_write_time=$($item.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))",
        "note=$($model.Note)",
        ""
    )
    Write-Host "Model copied: $($model.Source) -> $($model.Target) ($($item.Length) bytes)"
}

Copy-Item -Force (Join-Path $root "client\index.html") (Join-Path $dist "client\index.html")
Copy-Item -Force (Join-Path $root "config\pilot_deploy.properties") (Join-Path $dist "config\pilot_deploy.properties")

$ffmpegPath = $null
foreach ($name in @("ffmpeg.exe", "ffmpeg")) {
    $cmd = Get-Command $name -ErrorAction SilentlyContinue
    if ($cmd) {
        $ffmpegPath = $cmd.Source
        break
    }
}

if ($ffmpegPath) {
    Copy-Item -Force $ffmpegPath (Join-Path $dist "runtime\ffmpeg\ffmpeg.exe")
} else {
    Write-Warning "ffmpeg.exe not found in PATH. Put it into runtime\ffmpeg\ffmpeg.exe manually."
}

if (-not $AllowDebugBuild) {
    $packagedDebugDlls = Get-DebugDlls -Directories @(
        (Join-Path $dist "bin"),
        (Join-Path $dist "runtime")
    )

    if ($packagedDebugDlls) {
        $preview = $packagedDebugDlls |
            Select-Object -First 12 |
            ForEach-Object { $_.Substring($dist.Length + 1) }
        throw "Debug DLLs were packaged: $($preview -join ', '). Build/package Release dependencies only, or pass -AllowDebugBuild only for local development."
    }
}

$note = @"
Pilot Windows deployment package

Run:
  bin\pilot.exe

Open:
  http://127.0.0.1:8080/

Target machine requirements:
- Windows x64 with desktop session
- NVIDIA driver
- MSVC runtime DLLs are copied into bin/runtime when found on the build machine
- CUDA / cuDNN runtime DLLs are copied into bin/runtime when found on the build machine
- ffmpeg.exe in runtime\ffmpeg or on PATH
- Launch-time DLLs are copied into both bin\ and runtime\; start the app from bin\pilot.exe
"@

Set-Content -Path (Join-Path $dist "README_DEPLOY.txt") -Value $note -Encoding UTF8
Set-Content -Path (Join-Path $dist "MODEL_MANIFEST.txt") -Value ($modelManifest -join [Environment]::NewLine) -Encoding UTF8

Write-Host "Deployment package created at: $dist"
Write-Host "Build root used: $buildRoot"
