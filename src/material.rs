use std::io;

use glam::{Vec2, Vec3, Vec4};

use crate::texture::Texture;

pub struct Material<'a> {
    pub base_color_factor: Vec4,
    pub base_color_texture: Option<Texture<'a>>,
    pub emissive_factor: Vec3,
    pub emissive_texture: Option<Texture<'a>>,
    pub alpha_cutoff: Option<f32>,
    pub double_sided: bool,
}

impl Default for Material<'_> {
    fn default() -> Self {
        Self {
            base_color_factor: Vec4::ONE,
            base_color_texture: None,
            emissive_factor: Vec3::ZERO,
            emissive_texture: None,
            alpha_cutoff: None,
            double_sided: false,
        }
    }
}

impl<'a> Material<'a> {
    pub fn new(material: gltf::Material<'_>, images: &'a [gltf::image::Data]) -> io::Result<Self> {
        let pbr = material.pbr_metallic_roughness();
        Ok(Self {
            base_color_factor: Vec4::from(pbr.base_color_factor()),
            base_color_texture: load_texture(pbr.base_color_texture(), images)?,
            emissive_factor: Vec3::from(material.emissive_factor()),
            emissive_texture: load_texture(material.emissive_texture(), images)?,
            alpha_cutoff: (material.alpha_mode() == gltf::material::AlphaMode::Mask)
                .then(|| material.alpha_cutoff().unwrap_or(0.5)),
            double_sided: material.double_sided(),
        })
    }

    pub fn base_color(&self, uv: Vec2) -> Vec4 {
        self.base_color_factor
            * self
                .base_color_texture
                .as_ref()
                .map_or(Vec4::ONE, |texture| texture.sample(uv))
    }

    pub fn emission(&self, uv: Vec2) -> Vec3 {
        self.emissive_factor
            * self
                .emissive_texture
                .as_ref()
                .map_or(Vec3::ONE, |texture| texture.sample(uv).truncate())
    }
}

fn load_texture<'a>(
    info: Option<gltf::texture::Info<'_>>,
    images: &'a [gltf::image::Data],
) -> io::Result<Option<Texture<'a>>> {
    info.map(|info| {
        if info.extension_value("KHR_texture_transform").is_some() {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "Texture transforms are not implemented",
            ));
        }
        Texture::new(info.texture(), images)
    })
    .transpose()
}
