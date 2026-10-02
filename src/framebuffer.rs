use std::{io, path::Path};

pub struct Framebuffer {
    pub width: u32,
    pub height: u32,
    pub color: Vec<[u8; 3]>,
    pub depth: Vec<f32>,
}

impl Framebuffer {
    pub fn new(width: u32, height: u32) -> io::Result<Self> {
        if width == 0 || height == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Framebuffer dimensions must be nonzero",
            ));
        }
        let len =
            usize::try_from(u64::from(width) * u64::from(height)).map_err(io::Error::other)?;
        let mut color = Vec::new();
        color.try_reserve_exact(len).map_err(io::Error::other)?;
        color.resize(len, [0, 0, 0]);
        let mut depth = Vec::new();
        depth.try_reserve_exact(len).map_err(io::Error::other)?;
        depth.resize(len, f32::INFINITY);
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
