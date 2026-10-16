# Rasterizer

```sh
cargo run --release                       # list bundled scenes
cargo run --release -- living-room        # CPU rendering
cargo test --release
```

Settings are in `config.toml`; images are written to `output/`. `--camera-light`
adds an unshadowed light aligned with the camera.

## CUDA fragment shading

The optional `cuda` feature uses NVIDIA cuda-oxide, pinned in `Cargo.toml`.
Vertex processing, rasterization, shadow generation, and postprocessing stay on
the CPU. The shader, texture filters, BRDF, and shadow filters are shared Rust.

Select CUDA 13.4 explicitly (the default `nvcc` may be older). LLVM/Clang 22
must also be installed:

```sh
export CUDA_HOME=/usr/local/cuda-13.4
export CUDA_TOOLKIT_PATH="$CUDA_HOME"
export CUDA_OXIDE_LLC=/usr/bin/llc
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/nvvm/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Install the matching compiler CLI once, then build for the RTX 2000 Ada (`sm_89`):

```sh
rustup toolchain install nightly-2026-10-11 --component rust-src,rustc-dev,llvm-tools
cargo +nightly-2026-10-11 install --locked --git https://github.com/NVIDIA/cuda-rust.git \
  --rev e84e5b3d74014fcca36ffcf1b8d603c197cc7c8f cargo-oxide
cargo +nightly-2026-10-11 oxide build --arch sm_89 --no-fmad --features cuda \
  --cargo-target-dir target/cuda -- --release
target/cuda/release/rasterizer living-room --cuda
cargo +nightly-2026-10-11 oxide test --arch sm_89 --no-fmad \
  --cargo-target-dir target/cuda -- --release --features cuda -j1
```

CUDA tests use `-j1` because this compiler revision shares intermediate artifact
filenames between test targets.

Use `--no-fmad` for comparisons with CPU arithmetic: contraction can move shadow
filter results beyond the comparison tolerance. CUDA builds enable glam
`scalar-math` for CPU code too; normal builds retain CPU SIMD. Omit `--cuda` to
run the CPU fragment backend in the same scalar build.

The backend uploads textures and shadow depths directly from their existing
host slices, then retains scene resources and buffer capacity across renders.
Call `Pipeline::invalidate_resources()` (or the fragment stage's method directly)
after changing materials, textures, lights, or shadow depths, and when replacing a
scene. CPU stages need no refresh; the shared method is a no-op there. Camera and
fragment changes need no invalidation. Different material/light slice addresses
or lengths also trigger a refresh; pointer identity alone cannot detect in-place
edits or reused allocations.

Reported fragment wall time includes resource uploads on the first render and
after invalidation. Later renders transfer fragments and RGB results only. A fresh
CLI invocation still pays setup and scene uploads. The CUDA event interval measures
dispatch, including cold launch delay; it is not pure kernel execution or a
GPU-resident frame-time benchmark. `cuda::CudaFragment` implements `FragmentStage`.
