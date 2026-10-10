use std::io;

use glam::{Vec3, Vec4};

use crate::{
    asset::GltfAsset,
    texture::{ColorSpace, Sampling, Texture, TextureLoader, UvSample},
};

pub fn load_materials(asset: &GltfAsset, sampling: Sampling) -> io::Result<Vec<Material>> {
    let mut textures = TextureLoader::new(&asset.images, sampling);
    asset
        .document
        .materials()
        .map(|material| Material::new(material, &mut textures))
        .collect()
}

pub struct Material {
    pub base_color_factor: Vec4,
    pub base_color_texture: Option<Texture>,
    pub normal_scale: f32,
    pub normal_texture: Option<Texture>,
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub metallic_roughness_texture: Option<Texture>,
    pub occlusion_strength: f32,
    pub occlusion_texture: Option<Texture>,
    pub emissive_factor: Vec3,
    pub emissive_texture: Option<Texture>,
    pub alpha_cutoff: Option<f32>,
    pub double_sided: bool,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            base_color_factor: Vec4::ONE,
            base_color_texture: None,
            normal_scale: 1.0,
            normal_texture: None,
            metallic_factor: 1.0,
            roughness_factor: 1.0,
            metallic_roughness_texture: None,
            occlusion_strength: 1.0,
            occlusion_texture: None,
            emissive_factor: Vec3::ZERO,
            emissive_texture: None,
            alpha_cutoff: None,
            double_sided: false,
        }
    }
}

impl Material {
    fn new(material: gltf::Material<'_>, textures: &mut TextureLoader<'_>) -> io::Result<Self> {
        let pbr = material.pbr_metallic_roughness();
        let normal = material.normal_texture();
        let occlusion = material.occlusion_texture();
        Ok(Self {
            base_color_factor: Vec4::from(pbr.base_color_factor()),
            base_color_texture: load_texture(pbr.base_color_texture(), textures, ColorSpace::Srgb)?,
            normal_scale: normal.as_ref().map_or(1.0, |info| info.scale()),
            normal_texture: normal
                .map(|info| {
                    check_texture_transform(
                        info.extension_value("KHR_texture_transform").is_some(),
                    )?;
                    textures.load(info.texture(), ColorSpace::Linear)
                })
                .transpose()?,
            metallic_factor: pbr.metallic_factor(),
            roughness_factor: pbr.roughness_factor(),
            metallic_roughness_texture: load_texture(
                pbr.metallic_roughness_texture(),
                textures,
                ColorSpace::Linear,
            )?,
            occlusion_strength: occlusion.as_ref().map_or(1.0, |info| info.strength()),
            occlusion_texture: occlusion
                .map(|info| {
                    check_texture_transform(
                        info.extension_value("KHR_texture_transform").is_some(),
                    )?;
                    textures.load(info.texture(), ColorSpace::Linear)
                })
                .transpose()?,
            emissive_factor: Vec3::from(material.emissive_factor()),
            emissive_texture: load_texture(
                material.emissive_texture(),
                textures,
                ColorSpace::Srgb,
            )?,
            alpha_cutoff: (material.alpha_mode() == gltf::material::AlphaMode::Mask)
                .then(|| material.alpha_cutoff().unwrap_or(0.5)),
            double_sided: material.double_sided(),
        })
    }

    pub fn base_color(&self, uv: UvSample) -> Vec4 {
        self.base_color_factor
            * self
                .base_color_texture
                .as_ref()
                .map_or(Vec4::ONE, |texture| texture.sample(uv))
    }

    pub fn metallic(&self, uv: UvSample) -> f32 {
        self.metallic_roughness(uv).0
    }

    pub fn shading_normal(&self, uv: UvSample, normal: Vec3, tangent: Vec4) -> Vec3 {
        let Some(texture) = &self.normal_texture else {
            return normal;
        };
        if self.normal_scale == 0.0 || tangent.w == 0.0 {
            return normal;
        }
        let Some(t) =
            (tangent.truncate() - normal * normal.dot(tangent.truncate())).try_normalize()
        else {
            return normal;
        };
        let b = normal.cross(t) * tangent.w.signum();
        let mut sampled = texture.sample(uv).truncate() * 2.0 - Vec3::ONE;
        sampled.x *= self.normal_scale;
        sampled.y *= self.normal_scale;
        (t * sampled.x + b * sampled.y + normal * sampled.z)
            .try_normalize()
            .unwrap_or(normal)
    }

    pub fn metallic_roughness(&self, uv: UvSample) -> (f32, f32) {
        let sample = self
            .metallic_roughness_texture
            .as_ref()
            .map_or(Vec4::ONE, |texture| texture.sample(uv));
        (
            self.metallic_factor * sample.z,
            self.roughness_factor * sample.y,
        )
    }

    pub fn occlusion(&self, uv: UvSample) -> f32 {
        self.occlusion_texture.as_ref().map_or(1.0, |texture| {
            1.0 + self.occlusion_strength * (texture.sample(uv).x - 1.0)
        })
    }

    pub fn emission(&self, uv: UvSample) -> Vec3 {
        self.emissive_factor
            * self
                .emissive_texture
                .as_ref()
                .map_or(Vec3::ONE, |texture| texture.sample(uv).truncate())
    }
}

fn load_texture(
    info: Option<gltf::texture::Info<'_>>,
    textures: &mut TextureLoader<'_>,
    color_space: ColorSpace,
) -> io::Result<Option<Texture>> {
    info.map(|info| {
        check_texture_transform(info.extension_value("KHR_texture_transform").is_some())?;
        textures.load(info.texture(), color_space)
    })
    .transpose()
}

fn check_texture_transform(present: bool) -> io::Result<()> {
    if present {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "Texture transforms are not implemented",
        ));
    }
    Ok(())
}
