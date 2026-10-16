use std::{io, path::Path};

use glam::Vec3;

use crate::{config::OutputConfig, texture::encode_srgb};

pub struct RenderOutput {
    pub framebuffer: Framebuffer,
    pub covered_fragments: u64,
    pub screen_vertices: usize,
    pub screen_triangles: usize,
}

pub struct Framebuffer {
    /// Linear HDR RGB, before exposure and tone mapping.
    pub color: Vec<Vec3>,
    pub depth: DepthBuffer,
}

impl Framebuffer {
    pub fn new(width: u32, height: u32) -> io::Result<Self> {
        let depth = DepthBuffer::new(width, height)?;
        let mut color = Vec::new();
        color
            .try_reserve_exact(depth.values.len())
            .map_err(io::Error::other)?;
        color.resize(depth.values.len(), Vec3::ZERO);
        Ok(Self { color, depth })
    }

    pub fn save(
        &self,
        path: impl AsRef<Path>,
        output: &OutputConfig,
        supersampling: u32,
    ) -> image::ImageResult<()> {
        output.validate()?;
        if supersampling == 0
            || !self.depth.width.is_multiple_of(supersampling)
            || !self.depth.height.is_multiple_of(supersampling)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Supersampling must be nonzero and divide the framebuffer dimensions",
            )
            .into());
        }
        let width = self.depth.width / supersampling;
        let height = self.depth.height / supersampling;
        let samples = supersampling as f32 * supersampling as f32;
        let mut pixels = Vec::with_capacity(width as usize * height as usize);
        for y in 0..height {
            for x in 0..width {
                let mut color = Vec3::ZERO;
                for sy in y * supersampling..(y + 1) * supersampling {
                    let start =
                        sy as usize * self.depth.width as usize + (x * supersampling) as usize;
                    color += self.color[start..start + supersampling as usize]
                        .iter()
                        .copied()
                        .sum::<Vec3>();
                }
                color = (color / samples).max(Vec3::ZERO) * output.exposure;
                if output.tone_mapping {
                    color /= Vec3::ONE + color;
                }
                pixels.push(encode_srgb(color));
            }
        }
        image::save_buffer(
            path,
            pixels.as_flattened(),
            width,
            height,
            image::ColorType::Rgb8,
        )
    }
}

pub struct DepthBuffer {
    pub width: u32,
    pub height: u32,
    pub values: Vec<f32>,
}

impl DepthBuffer {
    pub fn new(width: u32, height: u32) -> io::Result<Self> {
        if width == 0 || height == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Depth buffer dimensions must be nonzero",
            ));
        }
        let len =
            usize::try_from(u64::from(width) * u64::from(height)).map_err(io::Error::other)?;
        let mut values = Vec::new();
        values.try_reserve_exact(len).map_err(io::Error::other)?;
        values.resize(len, f32::INFINITY);
        Ok(Self {
            width,
            height,
            values,
        })
    }

    pub fn save(&self, path: impl AsRef<Path>) -> image::ImageResult<()> {
        let pixels: Vec<u8> = self
            .values
            .iter()
            .map(|&depth| {
                if depth.is_finite() {
                    ((1.0 - depth.clamp(0.0, 1.0)) * 255.0).round() as u8
                } else {
                    0
                }
            })
            .collect();
        image::save_buffer(path, &pixels, self.width, self.height, image::ColorType::L8)
    }
}
