use std::io;

use glam::{Vec2, Vec3, Vec4};
use gltf::{
    image::{Data, Format},
    texture::{MagFilter, WrappingMode},
};

#[derive(Clone, Copy)]
pub enum ColorSpace {
    Srgb,
    Linear,
}

pub struct Texture<'a> {
    image: &'a Data,
    channels: usize,
    wrap_s: WrappingMode,
    wrap_t: WrappingMode,
    filter: MagFilter,
    color_space: ColorSpace,
}

impl<'a> Texture<'a> {
    pub fn new(
        texture: gltf::Texture<'_>,
        images: &'a [Data],
        color_space: ColorSpace,
    ) -> io::Result<Self> {
        let image = &images[texture.source().index()];
        let channels = match image.format {
            Format::R8 => 1,
            Format::R8G8 => 2,
            Format::R8G8B8 => 3,
            Format::R8G8B8A8 => 4,
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "Textures must use 8-bit channels",
                ));
            }
        };
        let expected =
            (u64::from(image.width) * u64::from(image.height)).checked_mul(channels as u64);
        if image.width == 0 || image.height == 0 || Some(image.pixels.len() as u64) != expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid texture image",
            ));
        }
        let sampler = texture.sampler();
        Ok(Self {
            image,
            channels,
            wrap_s: sampler.wrap_s(),
            wrap_t: sampler.wrap_t(),
            filter: sampler.mag_filter().unwrap_or(MagFilter::Linear),
            color_space,
        })
    }

    /// Sample level zero with the magnification filter, returning linear RGBA.
    pub fn sample(&self, uv: Vec2) -> Vec4 {
        let x = wrap_uv(f64::from(uv.x), self.wrap_s) * f64::from(self.image.width);
        let y = wrap_uv(f64::from(uv.y), self.wrap_t) * f64::from(self.image.height);
        if self.filter == MagFilter::Nearest {
            return self.texel(x.floor() as i64, y.floor() as i64);
        }
        let x = x - 0.5;
        let y = y - 0.5;
        let x0 = x.floor() as i64;
        let y0 = y.floor() as i64;
        let tx = (x - x.floor()) as f32;
        let ty = (y - y.floor()) as f32;
        let top = self.texel(x0, y0).lerp(self.texel(x0 + 1, y0), tx);
        let bottom = self.texel(x0, y0 + 1).lerp(self.texel(x0 + 1, y0 + 1), tx);
        top.lerp(bottom, ty)
    }

    fn texel(&self, x: i64, y: i64) -> Vec4 {
        let x = wrap_index(x, self.image.width, self.wrap_s);
        let y = wrap_index(y, self.image.height, self.wrap_t);
        let offset = (y * self.image.width as usize + x) * self.channels;
        let pixel = &self.image.pixels[offset..offset + self.channels];
        let rgba = match pixel {
            [gray] => [*gray, *gray, *gray, 255],
            [gray, alpha] => [*gray, *gray, *gray, *alpha],
            [r, g, b] => [*r, *g, *b, 255],
            [r, g, b, a] => [*r, *g, *b, *a],
            _ => unreachable!(),
        };
        let decode = |value| match self.color_space {
            ColorSpace::Srgb => decode_srgb(value),
            ColorSpace::Linear => f32::from(value) / 255.0,
        };
        Vec4::new(
            decode(rgba[0]),
            decode(rgba[1]),
            decode(rgba[2]),
            f32::from(rgba[3]) / 255.0,
        )
    }
}

fn wrap_uv(value: f64, mode: WrappingMode) -> f64 {
    match mode {
        WrappingMode::ClampToEdge => value.clamp(0.0, 1.0),
        WrappingMode::Repeat => value.rem_euclid(1.0),
        WrappingMode::MirroredRepeat => 1.0 - (value.rem_euclid(2.0) - 1.0).abs(),
    }
}

fn wrap_index(index: i64, size: u32, mode: WrappingMode) -> usize {
    let size = i64::from(size);
    (match mode {
        WrappingMode::ClampToEdge => index.clamp(0, size - 1),
        WrappingMode::Repeat => index.rem_euclid(size),
        WrappingMode::MirroredRepeat => {
            let index = index.rem_euclid(size * 2);
            if index < size {
                index
            } else {
                size * 2 - 1 - index
            }
        }
    }) as usize
}

fn decode_srgb(value: u8) -> f32 {
    let value = f32::from(value) / 255.0;
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

pub fn encode_srgb(color: Vec3) -> [u8; 3] {
    color.to_array().map(|value| {
        let value = value.clamp(0.0, 1.0);
        let encoded = if value <= 0.0031308 {
            value * 12.92
        } else {
            1.055 * value.powf(1.0 / 2.4) - 0.055
        };
        (encoded * 255.0).round() as u8
    })
}
