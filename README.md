# Rasterizer

```sh
cargo run --release                       # list bundled scenes
cargo run --release -- living-room        # CPU rendering
cargo test --release
```

Settings are in `config.toml`; images are written to `output/`. `--camera-light`
adds an unshadowed light aligned with the camera.

## CUDA backends

The optional `cuda` feature uses NVIDIA cuda-oxide, pinned in `Cargo.toml`.
Vertex transforms, clipping, rasterization math, fragment shading, texture filters,
BRDF, and shadow filters are shared Rust. Postprocessing stays on the CPU.

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
target/cuda/release/rasterizer living-room --cuda                # GPU fragments
target/cuda/release/rasterizer living-room --cuda-vertex         # GPU vertices
target/cuda/release/rasterizer living-room --cuda-vertex --cuda  # CPU rasterization
target/cuda/release/rasterizer living-room --cuda-rasterization # resident GPU pipeline
cargo +nightly-2026-10-11 oxide test --arch sm_89 --no-fmad \
  --cargo-target-dir target/cuda -- --release --features cuda -j1
```

CUDA tests use `-j1` because this compiler revision shares intermediate artifact
filenames between test targets.

Use `--no-fmad` for comparisons with CPU arithmetic: contraction can move shadow
filter results beyond the comparison tolerance. CUDA builds enable glam
`scalar-math` for CPU code too; normal builds retain CPU SIMD. All stages default
to CPU. `--cuda` and `--cuda-vertex` select individual stages;
`--cuda-rasterization` selects all three CUDA stages, including shadow rendering.

The backends retain uploaded geometry, materials, textures, shadows, and buffer
capacity across calls. CUDA stages share one `CudaDevice` (context, stream and
kernel module). Call `Pipeline::invalidate_resources()` after editing resources
in place or replacing a scene; pointer/length checks cannot detect all changes.
Camera movement alone needs no invalidation.

`CudaVertex` and `CudaFragment` implement the existing host stage interfaces for
mixed CPU/CUDA rendering. Their `timings()` separate packing, allocation/upload,
CUDA dispatch and readback. These paths return vertices or colors to the CPU.

For resident rendering, construct `Pipeline::new(CudaVertex::with_device(...),
CudaRasterization::with_device(...), CudaFragment::with_device(...))` with the same
device. Call `prepare_cuda_lights(scene, materials, lights, shadow_config,
unshadowed_lights)` once and again after invalidation, then call
`render_cuda(scene, view_projection, materials, width, height, camera_position)`
for each frame. Light discovery and shadow fitting use prepared host geometry.
Transformed vertices, visibility, fragment attributes and shadow depths stay on
the GPU; only final RGB/depth and small error/statistics counters return to the
CPU. Dimensions must be nonzero and their product must fit `u32`.

The resident rasterizer handles small triangles per thread and distributes large
triangles through a bounded work queue. Atomic depth/owner selection preserves
stable material/triangle order for equal depths. A per-pixel pass interpolates
only final winners. Clipping, coverage, depth and interpolation reuse the CPU
math. Storage includes dense fragment slots (about 80.6 MB at 800×600) and an
8-byte queue entry per input triangle; no global clipped-geometry pool is needed.

The CLI reports wall times for host stage interfaces and CUDA event intervals
for resident stages. Resident setup/readback is the remaining frame wall time;
shadow preparation includes its uploads and GPU passes. Event intervals can
include host launch delays and are not pure kernel execution time. A fresh CLI
invocation pays context setup, uploads and shadow generation; use repeated
renders in one process to measure cached performance. PNG encoding and writing
are reported separately.
