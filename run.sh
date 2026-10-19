#!/usr/bin/env bash
set -euo pipefail

backend=gpu
case "${1:-}" in
    cpu|gpu) backend=$1; shift ;;
    -h|--help) echo "Usage: $0 [cpu|gpu] [scene] [--camera-light]"; exit 0 ;;
esac
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if [[ $backend == cpu ]]; then
    exec cargo run --release --manifest-path "$root/Cargo.toml" \
        --target-dir "$root/target/cpu" -- "$@"
fi

export CUDA_HOME=/usr/local/cuda-13.4
export CUDA_TOOLKIT_PATH="$CUDA_HOME"
export CUDA_OXIDE_LLC=/usr/bin/llc
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/nvvm/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

(
    cd -- "$root"
    cargo +nightly-2026-10-11 oxide build --arch sm_89 --no-fmad --features cuda \
        --cargo-target-dir target/gpu -- --release
)
if (( $# )); then
    set -- "$@" --cuda-rasterization
fi
exec "$root/target/gpu/release/rasterizer" "$@"
