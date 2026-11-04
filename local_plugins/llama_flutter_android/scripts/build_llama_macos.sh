#!/usr/bin/env bash
# =============================================================================
# build_llama_macos.sh
# Compiles the bundled llama.cpp source into a universal macOS XCFramework
# (arm64 + x86_64) with Metal GPU and ggml-rpc cluster support.
#
# Run automatically by CocoaPods during `pod install` (via script_phase) or
# manually from the plugin root:
#   bash scripts/build_llama_macos.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

LLAMA_SRC="$PLUGIN_DIR/android/src/main/cpp/llama.cpp"
BUILD_ROOT="$PLUGIN_DIR/build/macos_llama"
XCFRAMEWORK_OUT="$PLUGIN_DIR/macos/Frameworks/llama.xcframework"

if [ -d "$XCFRAMEWORK_OUT" ] && [ "${FORCE_REBUILD:-0}" != "1" ]; then
  echo "✅ $XCFRAMEWORK_OUT already exists (set FORCE_REBUILD=1 to rebuild)."
  exit 0
fi

echo "🔨 Building llama.cpp for macOS from: $LLAMA_SRC"

if [ ! -f "$LLAMA_SRC/CMakeLists.txt" ]; then
  echo "❌ Error: llama.cpp source not found at $LLAMA_SRC"
  exit 1
fi

for tool in cmake xcodebuild libtool; do
  if ! command -v "$tool" &>/dev/null; then
    echo "❌ Required tool not found: $tool"
    exit 1
  fi
done

COMMON_CMAKE_FLAGS=(
  -DBUILD_SHARED_LIBS=OFF
  -DLLAMA_BUILD_TESTS=OFF
  -DLLAMA_BUILD_EXAMPLES=OFF
  -DLLAMA_BUILD_SERVER=OFF
  -DLLAMA_BUILD_TOOLS=OFF
  -DLLAMA_BUILD_COMMON=OFF
  -DLLAMA_OPENSSL=OFF
  -DGGML_OPENMP=OFF
  -DGGML_LLAMAFILE=OFF
  -DGGML_METAL=ON
  -DGGML_METAL_EMBED_LIBRARY=ON
  -DGGML_BLAS=ON
  -DGGML_BLAS_VENDOR=Apple
  -DGGML_RPC=ON
  -DLLAMA_BUILD_MTMD=ON
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0
)

NCPU=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)

BUILD_MACOS="$BUILD_ROOT/macos"
cmake -S "$LLAMA_SRC" -B "$BUILD_MACOS" \
  "${COMMON_CMAKE_FLAGS[@]}" \
  -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64"

cmake --build "$BUILD_MACOS" --config Release -j "$NCPU"

merge_archives() {
  local build_dir="$1"
  local output_lib="$2"
  local archives=()
  while IFS= read -r -d '' f; do
    archives+=("$f")
  done < <(find "$build_dir" -name "*.a" -print0)

  if [ ${#archives[@]} -eq 0 ]; then
    echo "❌ No .a files found in $build_dir"
    exit 1
  fi
  libtool -static -o "$output_lib" "${archives[@]}"
}

merge_archives "$BUILD_MACOS" "$BUILD_MACOS/libllama_merged.a"

HEADERS_STAGE="$BUILD_ROOT/headers"
rm -rf "$HEADERS_STAGE"
mkdir -p "$HEADERS_STAGE"

cp "$LLAMA_SRC/include/"*.h "$HEADERS_STAGE/"
cp "$LLAMA_SRC/ggml/include/"*.h "$HEADERS_STAGE/"
if [ -d "$LLAMA_SRC/tools/mtmd" ]; then
  cp "$LLAMA_SRC/tools/mtmd/"*.h "$HEADERS_STAGE/" 2>/dev/null || true
fi

rm -rf "$XCFRAMEWORK_OUT"
mkdir -p "$(dirname "$XCFRAMEWORK_OUT")"

xcodebuild -create-xcframework \
  -library "$BUILD_MACOS/libllama_merged.a" -headers "$HEADERS_STAGE" \
  -output "$XCFRAMEWORK_OUT"

echo "✅ XCFramework created at: $XCFRAMEWORK_OUT"
