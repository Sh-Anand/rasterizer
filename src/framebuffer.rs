use std::{io, path::Path};

pub struct Framebuffer {
    pub width: u32,
    pub height: u32,
    pub color: Vec<[u8; 3]>,
    pub depth: DepthBuffer,
}

impl Framebuffer {
    pub fn new(width: u32, height: u32) -> io::Result<Self> {
        let depth = DepthBuffer::new(width, height)?;
        let mut color = Vec::new();
        color
            .try_reserve_exact(depth.values.len())
            .map_err(io::Error::other)?;
        color.resize(depth.values.len(), [0, 0, 0]);
        Ok(Self {
            width,
            height,
            color,
            depth,
        })
    }

    pub fn save(&self, path: impl AsRef<Path>) -> image::ImageResult<()> {
        image::save_buffer(
            path,
            self.color.as_flattened(),
            self.width,
            self.height,
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
