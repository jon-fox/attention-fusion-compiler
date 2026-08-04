# Run Doc

## Prerequisites

MLIR/LLVM, CMake, and Ninja. On macOS:

```sh
brew install llvm cmake ninja
```

Build with Homebrew's clang, not Apple's — Apple clang fails to compile the MLIR headers.

## Build

```sh
cmake -S . -B build -G Ninja \
  -DMLIR_DIR=/opt/homebrew/opt/llvm/lib/cmake/mlir \
  -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++

cmake --build build
```

On Linux, point `MLIR_DIR` at your LLVM install's `lib/cmake/mlir` and drop the compiler flags.

## Test

`tests/build_graph` builds a `matmul -> scale -> softmax` graph in the `attn` dialect, verifies it, and prints it. Non-zero exit means the dialect is broken.

```sh
./build/tests/build_graph
```

Expected output:

```mlir
module {
  func.func @attention_scores(%arg0: tensor<128x64xf32>, %arg1: tensor<64x128xf32>) -> tensor<128x128xf32> {
    %0 = attn.matmul %arg0, %arg1 : (tensor<128x64xf32>, tensor<64x128xf32>) -> tensor<128x128xf32>
    %1 = attn.scale %0, 1.250000e-01 : tensor<128x128xf32>
    %2 = attn.softmax %1 {axis = 1 : i64} : tensor<128x128xf32>
    return %2 : tensor<128x128xf32>
  }
}
```

A `ld: warning: ignoring duplicate libraries` line during linking is harmless.

## Python side

Not yet used by the compiler — `main.py` is a stub. Set up when needed:

```sh
uv sync
uv run main.py
```

GPU lowering and benchmarking need a CUDA machine and are not part of this build.
