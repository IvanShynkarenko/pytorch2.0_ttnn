#!/usr/bin/env bash
set -euo pipefail

# 1) Parse build type
BUILD_TYPE=${1:-Release}
echo "> Build type: $BUILD_TYPE"

# 2) Locate this script’s directory
CUR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "> Current directory: $CUR_DIR"

# 3) Get the same ABI flags PyTorch was compiled with
TORCH_ABI_FLAGS=$(python3 "$CUR_DIR/utils/get_torch_abi_flags.py")
echo "> TORCH_ABI_FLAGS: $TORCH_ABI_FLAGS"

# 4) Ensure required tools exist
command -v cmake    >/dev/null || { echo "cmake not found"; exit 1; }
command -v ninja    >/dev/null || echo "warning: ninja not found, will fallback to Makefiles"
command -v gcc-12   >/dev/null || echo "warning: gcc-12 not found, using default gcc"
command -v g++-12   >/dev/null || echo "warning: g++-12 not found, using default g++"
command -v pip3     >/dev/null || { echo "pip3 not found"; exit 1; }

# 5) Configure TT-Metal
echo "> Configuring tt-metal"
GENERATOR="Ninja"
if ! command -v ninja &>/dev/null; then
  GENERATOR="Unix Makefiles"
  echo "> Ninja not found; falling back to $GENERATOR"
fi

cmake -B "$CUR_DIR/third-party/tt-metal/build" \
      -G "$GENERATOR" \
      -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
      -DCMAKE_INSTALL_PREFIX="$CUR_DIR/third-party/tt-metal/build" \
      -DCMAKE_DISABLE_PRECOMPILE_HEADERS=TRUE \
      -DENABLE_CCACHE=TRUE \
      -DTT_UNITY_BUILDS=ON \
      -DTT_ENABLE_LIGHT_METAL_TRACE=ON \
      -DWITH_PYTHON_BINDINGS=ON \
      -DCMAKE_TOOLCHAIN_FILE="$CUR_DIR/cmake/x86_64-linux-torch-toolchain.cmake" \
      -DCMAKE_CXX_FLAGS="${TORCH_ABI_FLAGS} -std=c++17 -O3" \
      -S "$CUR_DIR/third-party/tt-metal"

# 6) Build & install TT-Metal
echo "> Building tt-metal"
cmake --build "$CUR_DIR/third-party/tt-metal/build" --target install

# 7) Install the Python wheel for tt-metal
echo "> Installing tt-metal Python package"
pip3 install -e "$CUR_DIR/third-party/tt-metal"

# 8) Set TT_METAL_HOME for downstream builds
export TT_METAL_HOME="$CUR_DIR/third-party/tt-metal"
echo "> TT_METAL_HOME=$TT_METAL_HOME"

# 9) Build your PyTorch C++ extension in-place
echo "> Building torch-ttnn C++ extension"
# prefer inplace build so we don’t re-install every time
python3 setup.py build_ext --inplace

echo "✅ Build complete — you can now run your tests."
