# Rasterizer

```sh
./run.sh                                 # list bundled scenes
./run.sh living-room                     # build and render output/living-room.png
BOUNCE_RAYS=16 ./run.sh sponza           # 16 bounce rays per pixel instead of 4
BOUNCES=2 ./run.sh sponza                # continue each bounce ray for a second bounce
TRACE=1 ./run.sh sponza                  # also write output/sponza.trace
NTC_DP4A=1 ./run.sh flight-helmet        # decode neural textures with dp4a, not tensor cores
python3 scopes.py output/sponza.trace    # whole-shader cycles per warp on tensor-core scopes
python3 standins.py sponza               # stand-in NTC sets for sponza's materials (--remove undoes)
python3 assets/scenes/chess-mosaic/build.py --materials 4 --panels 8  # regenerate chess-mosaic
```

`TRACE=1` builds record which shader regions every GPU lane executes, as warp-level
records with active masks and source-level operation estimates; the format is documented
on `trace::write`. Trace builds are for traces only; time normal builds.

`neural` in `config.toml` chooses what networks replace. With `"textures"`, a material named M
decodes the RTXNTC texture set `assets/scenes/<scene>/ntc/M.ntc` instead of its textures;
FlightHelmet ships NVIDIA's sets. With `"materials"`, every material is a neural material
(latents, sampler and decoder networks) with seeded stand-in weights, for traces: images are
wrong. `bounce = true` adds ray-traced diffuse light, averaging `BOUNCE_RAYS` paths per pixel of
`BOUNCES` bounces each, fixed at build time.

Settings are in `config.toml`; images are written to `output/`. The CLI prints each GPU stage's
CUDA event interval, which can include host launch delays.

## CUDA

The renderer runs on CUDA through NVIDIA cuda-oxide, pinned in `Cargo.toml`. `run.sh` selects
CUDA 13.4 and the pinned nightly and builds in release mode into `target/gpu` for `sm_89`, with
`--no-fmad`: contraction changes shadow filter results slightly. LLVM/Clang 22 must be
installed, and the matching compiler CLI once:

```sh
rustup toolchain install nightly-2026-10-11 --component rust-src,rustc-dev,llvm-tools
cargo +nightly-2026-10-11 install --locked --git https://github.com/NVIDIA/cuda-rust.git \
  --rev e84e5b3d74014fcca36ffcf1b8d603c197cc7c8f cargo-oxide
```
