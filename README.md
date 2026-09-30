# Rasterizer

A CPU rasterizer for learning graphics and studying conventional and neural
shading workloads. Planned trace generation is optional and separate from
rendering; hardware simulation consumes the traces in a separate project.
Trace generation does not require producing an image.

Currently a dependency-free Rust binary skeleton. No rendering or tracing is
implemented yet.

```sh
cargo run
```

Checks:

```sh
cargo fmt --check
cargo test
```
