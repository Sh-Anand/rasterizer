#!/usr/bin/env bash
set -euo pipefail

case "${1:-}" in
    -h|--help) echo "Usage: $0 [scene] [--camera-light]"; exit 0 ;;
esac
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export CUDA_HOME=/usr/local/cuda-13.4
export CUDA_TOOLKIT_PATH="$CUDA_HOME"
export CUDA_OXIDE_LLC=/usr/bin/llc
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/nvvm/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

(
    cd -- "$root"
    cargo +nightly-2026-10-11 oxide build --arch sm_89 --no-fmad \
        --cargo-target-dir target/gpu -- --release
)
exec "$root/target/gpu/release/rasterizer" "$@"
