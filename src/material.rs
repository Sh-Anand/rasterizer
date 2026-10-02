use std::io;

use glam::{Vec2, Vec4};

use crate::texture::Texture;

pub struct Material<'a> {
    pub base_color_factor: Vec4,
    pub base_color_texture: Option<Texture<'a>>,
    pub alpha_cutoff: Option<f32>,
}

impl Default for Material<'_> {
    fn default() -> Self {
        Self {
            base_color_factor: Vec4::ONE,
            base_color_texture: None,
            alpha_cutoff: None,
        }
    }
}

impl<'a> Material<'a> {
    pub fn new(material: gltf::Material<'_>, images: &'a [gltf::image::Data]) -> io::Result<Self> {
        let pbr = material.pbr_metallic_roughness();
        let base_color_texture = pbr
            .base_color_texture()
            .map(|info| {
                if info.extension_value("KHR_texture_transform").is_some() {
                    return Err(io::Error::new(
                        io::ErrorKind::Unsupported,
                        "Texture transforms are not implemented",
                    ));
                }
                Texture::new(info.texture(), images)
            })
            .transpose()?;
        Ok(Self {
            base_color_factor: Vec4::from(pbr.base_color_factor()),
            base_color_texture,
            alpha_cutoff: (material.alpha_mode() == gltf::material::AlphaMode::Mask)
                .then(|| material.alpha_cutoff().unwrap_or(0.5)),
        })
    }

    pub fn base_color(&self, uv: Vec2) -> Vec4 {
        self.base_color_factor
            * self
                .base_color_texture
                .as_ref()
                .map_or(Vec4::ONE, |texture| texture.sample(uv))
    }
}
