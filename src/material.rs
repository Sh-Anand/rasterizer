use std::io;

use glam::{Vec3, Vec4};

use crate::{
    geometry::GltfAsset,
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

#[derive(Clone, Copy)]
pub struct Material<T = Texture> {
    pub base_color_factor: Vec3,
    pub base_color_texture: Option<T>,
    pub normal_scale: f32,
    pub normal_texture: Option<T>,
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub metallic_roughness_texture: Option<T>,
    pub emissive_factor: Vec3,
    pub emissive_texture: Option<T>,
    pub double_sided: bool,
}

impl<T> Default for Material<T> {
    fn default() -> Self {
        Self {
            base_color_factor: Vec3::ONE,
            base_color_texture: None,
            normal_scale: 1.0,
            normal_texture: None,
            metallic_factor: 1.0,
            roughness_factor: 1.0,
            metallic_roughness_texture: None,
            emissive_factor: Vec3::ZERO,
            emissive_texture: None,
            double_sided: false,
        }
    }
}

impl Material {
    fn new(material: gltf::Material<'_>, textures: &mut TextureLoader<'_>) -> io::Result<Self> {
        let pbr = material.pbr_metallic_roughness();
        let normal = material.normal_texture();
        Ok(Self {
            base_color_factor: Vec4::from(pbr.base_color_factor()).truncate(),
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
            emissive_factor: Vec3::from(material.emissive_factor()),
            emissive_texture: load_texture(
                material.emissive_texture(),
                textures,
                ColorSpace::Srgb,
            )?,
            double_sided: material.double_sided(),
        })
    }
}

impl<P: AsRef<[Vec4]>> Material<Texture<P>> {
    pub fn base_color(&self, uv: UvSample) -> Vec3 {
        self.base_color_factor * sample_texture(&self.base_color_texture, uv).truncate()
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
        let sample = sample_texture(&self.metallic_roughness_texture, uv);
        (
            self.metallic_factor * sample.z,
            self.roughness_factor * sample.y,
        )
    }

    pub fn emission(&self, uv: UvSample) -> Vec3 {
        self.emissive_factor * sample_texture(&self.emissive_texture, uv).truncate()
    }
}

fn sample_texture<P: AsRef<[Vec4]>>(texture: &Option<Texture<P>>, uv: UvSample) -> Vec4 {
    texture
        .as_ref()
        .map_or(Vec4::ONE, |texture| texture.sample(uv))
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

#[cfg(feature = "cuda")]
impl<T> Material<T> {
    pub(crate) fn map_textures<'a, U>(&'a self, mut map: impl FnMut(&'a T) -> U) -> Material<U> {
        Material {
            base_color_factor: self.base_color_factor,
            base_color_texture: self.base_color_texture.as_ref().map(&mut map),
            normal_scale: self.normal_scale,
            normal_texture: self.normal_texture.as_ref().map(&mut map),
            metallic_factor: self.metallic_factor,
            roughness_factor: self.roughness_factor,
            metallic_roughness_texture: self.metallic_roughness_texture.as_ref().map(&mut map),
            emissive_factor: self.emissive_factor,
            emissive_texture: self.emissive_texture.as_ref().map(&mut map),
            double_sided: self.double_sided,
        }
    }
}
