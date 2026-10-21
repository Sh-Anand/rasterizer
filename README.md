# Rasterizer

```sh
./run.sh                                 # list bundled scenes
./run.sh living-room                     # all GPU (default)
./run.sh gpu living-room                 # all GPU explicitly
./run.sh cpu living-room                 # all CPU
./run.sh cpu living-room --trace         # also write output/living-room.trace
cargo test --release
```

Settings are in `config.toml`; images are written to `output/`. `--camera-light`
adds an unshadowed light aligned with the camera.
The runner builds in release mode, with separate `target/cpu` and `target/gpu`
directories. GPU mode requires the one-time CUDA setup below; it selects CUDA
13.4 and the pinned nightly automatically. Postprocessing stays on the CPU.
Plain `cargo run --release -- living-room` still uses CPU rendering.

`--trace` swaps the CPU fragment stage for one that also records the shared
shader's operations per 32-lane warp (an 8x4 pixel block). The binary format is
documented on `trace::TraceFragment`; record site numbers follow `fragment::Site`.

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
target/cuda/release/rasterizer living-room --cuda
cargo +nightly-2026-10-11 oxide test --arch sm_89 --no-fmad \
  --cargo-target-dir target/cuda -- --release --features cuda -j1
```

CUDA tests use `-j1` because this compiler revision shares intermediate artifact
filenames between test targets.

Use `--no-fmad` for comparisons with CPU arithmetic: contraction can move shadow
filter results beyond the comparison tolerance. CUDA builds enable glam
`scalar-math` for CPU code too; normal builds retain CPU SIMD. The renderer binary
defaults to CPU; `--cuda` runs every stage, including shadow rendering, on CUDA.
CPU and CUDA stages cannot be mixed.

`prepare_cuda_lights` uploads geometry, materials, texels, lights and shadow maps;
renders reuse them and the buffer capacity. The CUDA stages share one `CudaDevice`
(context, stream and kernel module). Prepare again after editing resources in place
or replacing a scene; pointer/length checks cannot detect all changes, so call
`Pipeline::invalidate_resources()` to make renders fail until then. Camera movement
alone needs no preparation.

`bounce = true` in `config.toml` adds a ray-traced diffuse bounce (CUDA only), averaging
`BOUNCE_RAYS` rays per pixel; the count is fixed at build time (`BOUNCE_RAYS=16 ./run.sh
sponza`) and defaults to 1. Each bounce render writes `output/<scene>.hits`: the render width,
height and rays per pixel as u32, then per pixel the primary material and each ray's hit
material (`u32::MAX` for none; the scene's material count selects the default material).

Construct `Pipeline::new(CudaVertex::with_device(...),
CudaRasterization::with_device(...), CudaFragment::with_device(...))` with the same
device. Call `prepare_cuda_lights(scene, materials, texels, lights, shadow_config,
unshadowed_lights, bounce)` once and again after scene changes, then call
`render_cuda(scene, view_projection, materials, width, height, camera_position)`
for each frame. Light discovery and shadow fitting use prepared host geometry.
Transformed vertices, visibility, fragment attributes and shadow depths stay on
the GPU; only final RGB/depth and small error/statistics counters return to the
CPU. Dimensions must be nonzero and their product must fit `u32`.

The resident rasterizer handles small triangles per thread and distributes large
triangles through a bounded work queue. Atomic depth/owner selection preserves
stable material/triangle order for equal depths. A per-pixel pass interpolates
only final winners. Clipping, coverage, depth and interpolation reuse the CPU
math. Storage includes dense fragment slots (about 34.6 MB at 800×600) and an
8-byte queue entry per input triangle; no global clipped-geometry pool is needed.

The CLI reports wall times for CPU stages and CUDA event intervals for CUDA
stages. Resident setup/readback is the remaining frame wall time;
shadow preparation includes its uploads and GPU passes. Event intervals can
include host launch delays and are not pure kernel execution time. A fresh CLI
invocation pays context setup, uploads and shadow generation; use repeated
renders in one process to measure cached performance. PNG encoding and writing
are reported separately.
