use std::{collections::HashMap, io, sync::Arc};

use glam::{DVec2, Vec2, Vec3, Vec4};
use gltf::{
    image::{Data, Format},
    texture::WrappingMode,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Sampling {
    None,
    Bilinear,
    #[default]
    Trilinear,
}

#[derive(Debug, Clone, Copy)]
pub struct UvSample {
    pub uv: Vec2,
    pub dx: Vec2,
    pub dy: Vec2,
}

impl UvSample {
    /// A point query has no pixel footprint and samples level zero.
    pub fn point(uv: Vec2) -> Self {
        Self {
            uv,
            dx: Vec2::ZERO,
            dy: Vec2::ZERO,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum ColorSpace {
    Srgb,
    Linear,
}

pub struct Texture {
    levels: Arc<[MipLevel]>,
    wrap_s: WrappingMode,
    wrap_t: WrappingMode,
    sampling: Sampling,
}

pub(crate) struct TextureLoader<'a> {
    images: &'a [Data],
    sampling: Sampling,
    levels: HashMap<(usize, ColorSpace), Arc<[MipLevel]>>,
}

impl<'a> TextureLoader<'a> {
    pub fn new(images: &'a [Data], sampling: Sampling) -> Self {
        Self {
            images,
            sampling,
            levels: HashMap::new(),
        }
    }

    pub fn load(
        &mut self,
        texture: gltf::Texture<'_>,
        color_space: ColorSpace,
    ) -> io::Result<Texture> {
        let key = (texture.source().index(), color_space);
        let levels = match self.levels.entry(key) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                let mut levels = vec![MipLevel::decode(&self.images[key.0], color_space)?];
                if self.sampling == Sampling::Trilinear {
                    while levels
                        .last()
                        .is_some_and(|level| level.width > 1 || level.height > 1)
                    {
                        levels.push(levels.last().unwrap().downsample());
                    }
                }
                entry.insert(levels.into())
            }
        };
        let sampler = texture.sampler();
        Ok(Texture {
            levels: Arc::clone(levels),
            wrap_s: sampler.wrap_s(),
            wrap_t: sampler.wrap_t(),
            sampling: self.sampling,
        })
    }
}

struct MipLevel {
    width: u32,
    height: u32,
    pixels: Vec<Vec4>,
}

impl MipLevel {
    fn decode(image: &Data, color_space: ColorSpace) -> io::Result<Self> {
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
        let decode = |value| match color_space {
            ColorSpace::Srgb => decode_srgb(value),
            ColorSpace::Linear => f32::from(value) / 255.0,
        };
        let pixels = image
            .pixels
            .chunks_exact(channels)
            .map(|pixel| {
                let [r, g, b, a] = match pixel {
                    [gray] => [*gray, *gray, *gray, 255],
                    [gray, alpha] => [*gray, *gray, *gray, *alpha],
                    [r, g, b] => [*r, *g, *b, 255],
                    [r, g, b, a] => [*r, *g, *b, *a],
                    _ => unreachable!(),
                };
                Vec4::new(decode(r), decode(g), decode(b), f32::from(a) / 255.0)
            })
            .collect();
        Ok(Self {
            width: image.width,
            height: image.height,
            pixels,
        })
    }

    fn downsample(&self) -> Self {
        let width = (self.width / 2).max(1);
        let height = (self.height / 2).max(1);
        let mut pixels = Vec::with_capacity(width as usize * height as usize);
        for y in 0..height {
            let y0 = f64::from(y) * f64::from(self.height) / f64::from(height);
            let y1 = f64::from(y + 1) * f64::from(self.height) / f64::from(height);
            for x in 0..width {
                let x0 = f64::from(x) * f64::from(self.width) / f64::from(width);
                let x1 = f64::from(x + 1) * f64::from(self.width) / f64::from(width);
                let mut sum = Vec4::ZERO;
                // Area weights include all texels, including odd-sized image edges.
                for sy in y0.floor() as u32..(y1.ceil() as u32).min(self.height) {
                    let wy = y1.min(f64::from(sy + 1)) - y0.max(f64::from(sy));
                    for sx in x0.floor() as u32..(x1.ceil() as u32).min(self.width) {
                        let wx = x1.min(f64::from(sx + 1)) - x0.max(f64::from(sx));
                        sum += self.pixels[sy as usize * self.width as usize + sx as usize]
                            * (wx * wy) as f32;
                    }
                }
                pixels.push(sum / ((x1 - x0) * (y1 - y0)) as f32);
            }
        }
        Self {
            width,
            height,
            pixels,
        }
    }
}

impl Texture {
    pub fn sample(&self, sample: UvSample) -> Vec4 {
        let lod = if self.sampling == Sampling::Trilinear {
            let size = DVec2::new(
                f64::from(self.levels[0].width),
                f64::from(self.levels[0].height),
            );
            let rho_squared = (sample.dx.as_dvec2() * size)
                .length_squared()
                .max((sample.dy.as_dvec2() * size).length_squared());
            (0.5 * rho_squared.log2()).max(0.0) as f32
        } else {
            0.0
        };
        self.sample_lod(sample.uv, lod)
    }

    pub fn sample_lod(&self, uv: Vec2, lod: f32) -> Vec4 {
        if self.sampling != Sampling::Trilinear {
            return self.sample_level(uv, 0);
        }
        let lod = lod.max(0.0).min((self.levels.len() - 1) as f32);
        let lower = lod.floor() as usize;
        let upper = (lower + 1).min(self.levels.len() - 1);
        let color = self.sample_level(uv, lower);
        if lod.fract() == 0.0 {
            return color;
        }
        color.lerp(self.sample_level(uv, upper), lod.fract())
    }

    fn sample_level(&self, uv: Vec2, level: usize) -> Vec4 {
        let level = &self.levels[level];
        let x = wrap_uv(f64::from(uv.x), self.wrap_s) * f64::from(level.width);
        let y = wrap_uv(f64::from(uv.y), self.wrap_t) * f64::from(level.height);
        let texel = |x, y| {
            let x = wrap_index(x, level.width, self.wrap_s);
            let y = wrap_index(y, level.height, self.wrap_t);
            level.pixels[y * level.width as usize + x]
        };
        if self.sampling == Sampling::None {
            return texel(x.floor() as i64, y.floor() as i64);
        }
        let x = x - 0.5;
        let y = y - 0.5;
        let x0 = x.floor() as i64;
        let y0 = y.floor() as i64;
        let tx = (x - x.floor()) as f32;
        let ty = (y - y.floor()) as f32;
        let top = texel(x0, y0).lerp(texel(x0 + 1, y0), tx);
        let bottom = texel(x0, y0 + 1).lerp(texel(x0 + 1, y0 + 1), tx);
        top.lerp(bottom, ty)
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
