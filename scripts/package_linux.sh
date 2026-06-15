#!/usr/bin/env bash
set -Eeuo pipefail

OUTPUT_DIR="dist/pilot_linux"
PREFERRED_BUILD_DIR=""
VCPKG_ROOT="${VCPKG_ROOT:-/opt/vcpkg}"
VCPKG_INSTALLED=""
ORT_DIR="${ORT_DIR:-/opt/onnxruntime}"
LIBTORCH_DIR="${LIBTORCH_DIR:-/opt/libtorch}"
CUDA_ROOT="${CUDA_HOME:-${CUDA_PATH:-/usr/local/cuda}}"
CUDNN_ROOT="${CUDNN_ROOT:-}"
FFMPEG_PATH="${FFMPEG_PATH:-}"
ALLOW_DEBUG_BUILD=0
SKIP_SO_RUNTIME=0
SKIP_CUDA_RUNTIME=0
SKIP_LDD_RUNTIME=0

usage() {
  cat <<'EOF'
Usage:
  scripts/package_linux.sh [options]

Options:
  --output-dir DIR             Output package directory. Default: dist/pilot_linux
  --preferred-build-dir DIR    Build directory containing the pilot executable
  --vcpkg-root DIR             vcpkg root. Default: $VCPKG_ROOT or /opt/vcpkg
  --vcpkg-installed DIR        vcpkg installed triplet root. Default: <vcpkg-root>/installed/x64-linux
  --ort-dir DIR                ONNX Runtime root. Default: $ORT_DIR or /opt/onnxruntime
  --libtorch-dir DIR           LibTorch root. Default: $LIBTORCH_DIR or /opt/libtorch
  --cuda-root DIR              CUDA root. Default: $CUDA_HOME, $CUDA_PATH, or /usr/local/cuda
  --cudnn-root DIR             cuDNN root containing lib/ or lib64/
  --ffmpeg-path FILE           Explicit ffmpeg executable to package
  --allow-debug-build          Allow packaging a build path containing "debug"
  --skip-so-runtime            Do not copy dependency .so files into runtime/lib
  --skip-cuda-runtime          Do not copy CUDA/cuDNN .so files into runtime/lib
  --skip-ldd-runtime           Do not copy non-system libraries discovered by ldd
  -h, --help                   Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --preferred-build-dir) PREFERRED_BUILD_DIR="$2"; shift 2 ;;
    --vcpkg-root) VCPKG_ROOT="$2"; shift 2 ;;
    --vcpkg-installed) VCPKG_INSTALLED="$2"; shift 2 ;;
    --ort-dir) ORT_DIR="$2"; shift 2 ;;
    --libtorch-dir) LIBTORCH_DIR="$2"; shift 2 ;;
    --cuda-root) CUDA_ROOT="$2"; shift 2 ;;
    --cudnn-root) CUDNN_ROOT="$2"; shift 2 ;;
    --ffmpeg-path) FFMPEG_PATH="$2"; shift 2 ;;
    --allow-debug-build) ALLOW_DEBUG_BUILD=1; shift ;;
    --skip-so-runtime) SKIP_SO_RUNTIME=1; shift ;;
    --skip-cuda-runtime) SKIP_CUDA_RUNTIME=1; shift ;;
    --skip-ldd-runtime) SKIP_LDD_RUNTIME=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "$VCPKG_INSTALLED" ]]; then
  VCPKG_INSTALLED="$VCPKG_ROOT/installed/x64-linux"
fi

if [[ "$OUTPUT_DIR" = /* ]]; then
  DIST="$OUTPUT_DIR"
else
  DIST="$ROOT/$OUTPUT_DIR"
fi

find_build_root() {
  local candidates=()
  if [[ -n "$PREFERRED_BUILD_DIR" ]]; then
    candidates+=("$PREFERRED_BUILD_DIR")
  fi
  candidates+=(
    "out/build/linux-release"
    "out/build/linux-base"
    "build"
  )

  local candidate path
  for candidate in "${candidates[@]}"; do
    [[ -z "$candidate" ]] && continue
    if [[ "$candidate" = /* ]]; then
      path="$candidate"
    else
      path="$ROOT/$candidate"
    fi
    if [[ -f "$path/pilot" ]]; then
      echo "$path"
      return 0
    fi
  done
  return 1
}

copy_file_required() {
  local src="$1"
  local dst="$2"
  if [[ ! -f "$src" ]]; then
    echo "Required file not found: $src" >&2
    exit 1
  fi
  cp -f "$src" "$dst"
}

copy_shared_objects_from_dir() {
  local dir="$1"
  local dest="$2"
  [[ -d "$dir" ]] || return 0

  find "$dir" -maxdepth 1 \( -type f -o -type l \) \( -name '*.so' -o -name '*.so.*' \) -print0 |
    while IFS= read -r -d '' file; do
      cp -a "$file" "$dest/"
    done
}

copy_cuda_runtime() {
  local dest="$1"
  local cuda_libs=()

  [[ -d "$CUDA_ROOT/lib64" ]] && cuda_libs+=("$CUDA_ROOT/lib64")
  [[ -d "$CUDA_ROOT/lib" ]] && cuda_libs+=("$CUDA_ROOT/lib")
  [[ -n "$CUDNN_ROOT" && -d "$CUDNN_ROOT/lib64" ]] && cuda_libs+=("$CUDNN_ROOT/lib64")
  [[ -n "$CUDNN_ROOT" && -d "$CUDNN_ROOT/lib" ]] && cuda_libs+=("$CUDNN_ROOT/lib")

  local dir pattern
  local patterns=(
    'libcudart.so*'
    'libcublas.so*'
    'libcublasLt.so*'
    'libcufft.so*'
    'libcurand.so*'
    'libcusolver.so*'
    'libcusparse.so*'
    'libnvJitLink.so*'
    'libnvrtc.so*'
    'libnvrtc-builtins.so*'
    'libnvjpeg.so*'
    'libcudnn*.so*'
  )

  for dir in "${cuda_libs[@]}"; do
    for pattern in "${patterns[@]}"; do
      find "$dir" -maxdepth 1 \( -type f -o -type l \) -name "$pattern" -print0 2>/dev/null |
        while IFS= read -r -d '' file; do
          cp -a "$file" "$dest/"
        done
    done
  done
}

copy_ldd_runtime() {
  local binary="$1"
  local dest="$2"
  command -v ldd >/dev/null 2>&1 || return 0
  [[ -f "$binary" ]] || return 0

  ldd "$binary" 2>/dev/null |
    awk '
      /=> \// { print $3 }
      /^[[:space:]]*\// { print $1 }
    ' |
    while IFS= read -r lib; do
      [[ -f "$lib" ]] || continue
      local name
      name="$(basename "$lib")"
      case "$name" in
        linux-vdso*|ld-linux*|libc.so*|libm.so*|libpthread.so*|libdl.so*|librt.so*|libresolv.so*|libnsl.so*|libutil.so*|libanl.so*|libcrypt.so*|libcuda.so*|libnvidia*)
          continue
          ;;
      esac
      cp -f "$lib" "$dest/"
    done
}

BUILD_ROOT="$(find_build_root || true)"
if [[ -z "$BUILD_ROOT" ]]; then
  echo "pilot executable not found. Build first or pass --preferred-build-dir." >&2
  exit 1
fi

if [[ "$ALLOW_DEBUG_BUILD" -ne 1 && "$BUILD_ROOT" =~ [Dd]ebug ]]; then
  echo "Refusing to package a Debug build from: $BUILD_ROOT" >&2
  echo "Build Release first, or pass --allow-debug-build for local development." >&2
  exit 1
fi

if [[ -e "$DIST" ]]; then
  rm -rf "$DIST"
fi

mkdir -p \
  "$DIST/bin" \
  "$DIST/runtime/lib" \
  "$DIST/runtime/ffmpeg" \
  "$DIST/client" \
  "$DIST/models" \
  "$DIST/config" \
  "$DIST/output" \
  "$DIST/temp"

cp -f "$BUILD_ROOT/pilot" "$DIST/bin/pilot"
chmod +x "$DIST/bin/pilot"

copy_file_required "$ROOT/i3d/models/a320_new_full.onnx" "$DIST/models/a320_new_full.onnx"
copy_file_required "$ROOT/algos/tridet_a320.onnx" "$DIST/models/tridet_a320.onnx"
copy_file_required "$ROOT/yolo/config/best.onnx" "$DIST/models/best.onnx"
copy_file_required "$ROOT/client/index.html" "$DIST/client/index.html"
copy_file_required "$ROOT/config/pilot_deploy.linux.properties" "$DIST/config/pilot_deploy.linux.properties"

write_model_manifest() {
  local manifest="$DIST/MODEL_MANIFEST.txt"
  {
    echo "Pilot model manifest"
    echo "Generated: $(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "Package: $DIST"
    echo

    echo "[I3D feature extractor]"
    echo "source=i3d/models/a320_new_full.onnx"
    echo "target=models/a320_new_full.onnx"
    echo "bytes=$(wc -c < "$ROOT/i3d/models/a320_new_full.onnx")"
    echo "last_write_time=$(date -r "$ROOT/i3d/models/a320_new_full.onnx" '+%Y-%m-%d %H:%M:%S')"
    echo "note=Updated from final_models/特征提取"
    echo

    echo "[TriDet temporal detector]"
    echo "source=algos/tridet_a320.onnx"
    echo "target=models/tridet_a320.onnx"
    echo "bytes=$(wc -c < "$ROOT/algos/tridet_a320.onnx")"
    echo "last_write_time=$(date -r "$ROOT/algos/tridet_a320.onnx" '+%Y-%m-%d %H:%M:%S')"
    echo "note=Updated from final_models/时序检测, 25 classes with per-class thresholds in service"
    echo

    echo "[YOLO object detector]"
    echo "source=yolo/config/best.onnx"
    echo "target=models/best.onnx"
    echo "bytes=$(wc -c < "$ROOT/yolo/config/best.onnx")"
    echo "last_write_time=$(date -r "$ROOT/yolo/config/best.onnx" '+%Y-%m-%d %H:%M:%S')"
    echo "note=Kept unchanged by request"
  } > "$manifest"
}

write_model_manifest

if [[ -z "$FFMPEG_PATH" ]]; then
  FFMPEG_PATH="$(command -v ffmpeg || true)"
fi

if [[ -n "$FFMPEG_PATH" && -x "$FFMPEG_PATH" ]]; then
  cp -f "$FFMPEG_PATH" "$DIST/runtime/ffmpeg/ffmpeg"
  chmod +x "$DIST/runtime/ffmpeg/ffmpeg"
else
  echo "Warning: ffmpeg was not found. Put it into runtime/ffmpeg/ffmpeg or set ffmpeg_path=/usr/bin/ffmpeg." >&2
fi

if [[ "$SKIP_SO_RUNTIME" -ne 1 ]]; then
  copy_shared_objects_from_dir "$BUILD_ROOT" "$DIST/runtime/lib"
  copy_shared_objects_from_dir "$ORT_DIR/lib" "$DIST/runtime/lib"
  copy_shared_objects_from_dir "$LIBTORCH_DIR/lib" "$DIST/runtime/lib"
  copy_shared_objects_from_dir "$VCPKG_INSTALLED/lib" "$DIST/runtime/lib"
fi

if [[ "$SKIP_LDD_RUNTIME" -ne 1 ]]; then
  copy_ldd_runtime "$BUILD_ROOT/pilot" "$DIST/runtime/lib"
fi

if [[ "$SKIP_CUDA_RUNTIME" -ne 1 ]]; then
  copy_cuda_runtime "$DIST/runtime/lib"
fi

cat > "$DIST/run.sh" <<'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="$ROOT/runtime/lib:${LD_LIBRARY_PATH:-}"

cd "$ROOT/bin"
exec ./pilot "$@"
EOF
chmod +x "$DIST/run.sh"

cat > "$DIST/README_DEPLOY.txt" <<'EOF'
Pilot Linux deployment package

Run:
  ./run.sh

Open:
  http://127.0.0.1:8080/

Directory layout:
  bin/pilot
  client/index.html
  config/pilot_deploy.linux.properties
  models/*.onnx
  runtime/ffmpeg/ffmpeg
  runtime/lib/*.so*
  output/
  temp/

Target machine requirements:
- Linux x86_64
- NVIDIA driver if GPU inference is enabled
- Compatible CUDA/cuDNN runtime if not fully packaged into runtime/lib
- Port 8080 available, or change config/pilot_deploy.linux.properties

If packaged shared libraries are incomplete, set LD_LIBRARY_PATH before running:
  export LD_LIBRARY_PATH=/path/to/onnxruntime/lib:/path/to/libtorch/lib:/path/to/vcpkg/installed/x64-linux/lib:$LD_LIBRARY_PATH
EOF

echo "Deployment package created at: $DIST"
echo "Build root used: $BUILD_ROOT"
