use std::io;

use glam::{Vec2, Vec3, Vec4};

use crate::{
    asset::GltfAsset,
    texture::{ColorSpace, Texture},
};

pub fn load_materials(asset: &GltfAsset) -> io::Result<Vec<Material<'_>>> {
    asset
        .document
        .materials()
        .map(|material| Material::new(material, &asset.images))
        .collect()
}

pub struct Material<'a> {
    pub base_color_factor: Vec4,
    pub base_color_texture: Option<Texture<'a>>,
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub metallic_roughness_texture: Option<Texture<'a>>,
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
            metallic_factor: 1.0,
            roughness_factor: 1.0,
            metallic_roughness_texture: None,
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
            base_color_texture: load_texture(pbr.base_color_texture(), images, ColorSpace::Srgb)?,
            metallic_factor: pbr.metallic_factor(),
            roughness_factor: pbr.roughness_factor(),
            metallic_roughness_texture: load_texture(
                pbr.metallic_roughness_texture(),
                images,
                ColorSpace::Linear,
            )?,
            emissive_factor: Vec3::from(material.emissive_factor()),
            emissive_texture: load_texture(material.emissive_texture(), images, ColorSpace::Srgb)?,
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

    pub fn metallic(&self, uv: Vec2) -> f32 {
        self.metallic_roughness(uv).0
    }

    pub fn metallic_roughness(&self, uv: Vec2) -> (f32, f32) {
        let sample = self
            .metallic_roughness_texture
            .as_ref()
            .map_or(Vec4::ONE, |texture| texture.sample(uv));
        (
            self.metallic_factor * sample.z,
            self.roughness_factor * sample.y,
        )
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
    color_space: ColorSpace,
) -> io::Result<Option<Texture<'a>>> {
    info.map(|info| {
        if info.extension_value("KHR_texture_transform").is_some() {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "Texture transforms are not implemented",
            ));
        }
        Texture::new(info.texture(), images, color_space)
    })
    .transpose()
}
