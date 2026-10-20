pub mod config;
pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod framebuffer;
pub mod geometry;
pub mod lighting;
pub mod material;
pub mod pipeline;
pub mod shadow;
pub mod texture;
pub mod trace;

mod fragment;
mod rasterization;
