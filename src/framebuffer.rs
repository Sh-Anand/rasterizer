use std::{io, path::Path};

use glam::Vec3;

use crate::{config::OutputConfig, texture::encode_srgb};

pub struct RenderOutput {
    pub framebuffer: Framebuffer,
    pub covered_fragments: u64,
    /// CUDA does not count post-clipping vertices.
    pub screen_vertices: Option<usize>,
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

    pub fn to_image(
        &self,
        output: &OutputConfig,
        supersampling: u32,
    ) -> image::ImageResult<image::RgbImage> {
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
        let mut image = image::RgbImage::new(width, height);
        for (x, y, pixel) in image.enumerate_pixels_mut() {
            let mut color = Vec3::ZERO;
            for sy in y * supersampling..(y + 1) * supersampling {
                let start = sy as usize * self.depth.width as usize + (x * supersampling) as usize;
                color += self.color[start..start + supersampling as usize]
                    .iter()
                    .copied()
                    .sum::<Vec3>();
            }
            color = (color / samples).max(Vec3::ZERO) * output.exposure;
            if output.tone_mapping {
                color /= Vec3::ONE + color;
            }
            *pixel = image::Rgb(encode_srgb(color));
        }
        Ok(image)
    }
}

#[derive(Clone, Copy)]
pub struct DepthBuffer<P = Vec<f32>> {
    pub width: u32,
    pub height: u32,
    pub values: P,
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
