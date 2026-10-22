use std::io;

use glam::Vec3;

use crate::{config::OutputConfig, texture::encode_srgb};

pub struct RenderOutput {
    pub framebuffer: Framebuffer,
    pub covered_fragments: u64,
    pub screen_triangles: usize,
}

pub struct Framebuffer {
    /// Linear HDR RGB, before exposure and tone mapping.
    pub color: Vec<Vec3>,
    pub depth: DepthBuffer,
}

impl Framebuffer {
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

pub struct DepthBuffer {
    pub width: u32,
    pub height: u32,
    pub values: Vec<f32>,
}
