# Rasterizer

```sh
./run.sh                                 # list bundled scenes
./run.sh living-room                     # build and render output/living-room.png
BOUNCE_RAYS=16 ./run.sh sponza           # 16 bounce rays per pixel instead of 4
```

Settings are in `config.toml`; images are written to `output/`. The runner builds in
release mode into `target/gpu`; it requires the one-time CUDA setup below and selects
CUDA 13.4 and the pinned nightly automatically.

## CUDA

The renderer runs on CUDA through NVIDIA cuda-oxide, pinned in `Cargo.toml`. Vertex
transforms, clipping, rasterization math, fragment shading, texture filters, BRDF,
shadow filters and ray traversal are Rust shared between kernels and host preparation.
Postprocessing stays on the CPU.

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
cargo +nightly-2026-10-11 oxide build --arch sm_89 --no-fmad \
  --cargo-target-dir target/gpu -- --release
target/gpu/release/rasterizer living-room
```

`--no-fmad` keeps renders bit-identical to earlier builds; contraction changes shadow
filter results slightly.

`cuda::Renderer::new(device)` owns the stream, kernel module and every device buffer.
Call `prepare(scene, materials, texels, light_config, shadow_config, bounce)` once and
again after any scene or resource change, then `render(scene, view_projection, width,
height, camera_position)` for each frame. `prepare` uploads geometry, materials and
texels and renders the light's shadow map, fitted on the host to the prepared geometry;
renders reuse them and the buffer capacity. Camera movement alone needs no preparation.
Transformed vertices, visibility, fragment attributes and shadow depths stay on the
GPU; only final RGB/depth and small error/statistics counters return to the host.
Dimensions must be nonzero and their product must fit `u32`.

`bounce = true` in `config.toml` adds a ray-traced diffuse bounce, averaging
`BOUNCE_RAYS` rays per pixel; the count is fixed at build time and defaults to 4. Each
bounce render writes `output/<scene>.hits`: the render width, height and rays per pixel
as u32, then per pixel the primary material and each ray's hit material (`u32::MAX` for
none; the scene's material count selects the default material).

The resident rasterizer handles small triangles per thread and distributes large
triangles through a bounded work queue. Atomic depth/owner selection preserves
stable material/triangle order for equal depths. A per-pixel pass interpolates
only final winners. Storage includes dense fragment slots (about 56.6 MB at 1024×768)
and an 8-byte queue entry per input triangle; no global clipped-geometry pool is needed.

The CLI reports CUDA event intervals for each stage; setup/readback is the remaining
frame wall time, and shadow preparation includes its uploads and GPU passes. Event
intervals can include host launch delays and are not pure kernel execution time. A
fresh CLI invocation pays context setup, uploads and shadow generation; use repeated
renders in one process to measure cached performance. PNG encoding and writing are
reported separately.
