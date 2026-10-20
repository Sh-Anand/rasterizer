use std::io;

use glam::{Vec3, Vec4};

use crate::{
    geometry::GltfAsset,
    texture::{ColorSpace, Sampling, Texel, Texture, TextureLoader, UvSample},
};

/// Returns the materials and the texel array their textures index.
pub fn load_materials(
    asset: &GltfAsset,
    sampling: Sampling,
) -> io::Result<(Vec<Material>, Vec<Texel>)> {
    let mut textures = TextureLoader::new(&asset.images, sampling);
    let materials = asset
        .document
        .materials()
        .map(|material| Material::new(material, &mut textures))
        .collect::<io::Result<_>>()?;
    Ok((materials, textures.texels))
}

/// Factor-scaled material values at one UV.
pub struct MaterialSample {
    pub base_color: Vec3,
    /// Scaled tangent-space normal; meaningful only with a normal texture.
    pub normal: Vec3,
    pub metallic: f32,
    pub roughness: f32,
    pub emission: Vec3,
}

#[derive(Clone, Copy)]
pub struct Material {
    pub base_color_factor: Vec3,
    pub base_color_texture: Option<Texture>,
    pub normal_scale: f32,
    pub normal_texture: Option<Texture>,
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub metallic_roughness_texture: Option<Texture>,
    pub emissive_factor: Vec3,
    pub emissive_texture: Option<Texture>,
    pub double_sided: bool,
}

impl Default for Material {
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

    /// Reads every texture at one UV, as a texture-set decoder produces all channels at once.
    #[inline(always)]
    pub fn sample(&self, texels: &[Texel], uv: UvSample) -> MaterialSample {
        let base_color = sample_texture(&self.base_color_texture, texels, uv);
        let mut normal =
            sample_texture(&self.normal_texture, texels, uv).truncate() * 2.0 - Vec3::ONE;
        normal.x *= self.normal_scale;
        normal.y *= self.normal_scale;
        let metallic_roughness = sample_texture(&self.metallic_roughness_texture, texels, uv);
        MaterialSample {
            base_color: self.base_color_factor * base_color.truncate(),
            normal,
            metallic: self.metallic_factor * metallic_roughness.z,
            roughness: self.roughness_factor * metallic_roughness.y,
            emission: self.emission(texels, uv),
        }
    }

    /// Applies a sampled tangent-space normal.
    pub fn shading_normal(&self, sampled: Vec3, normal: Vec3, tangent: Vec4) -> Vec3 {
        if self.normal_texture.is_none() || self.normal_scale == 0.0 || tangent.w == 0.0 {
            return normal;
        }
        let Some(t) =
            (tangent.truncate() - normal * normal.dot(tangent.truncate())).try_normalize()
        else {
            return normal;
        };
        let b = normal.cross(t) * tangent.w.signum();
        (t * sampled.x + b * sampled.y + normal * sampled.z)
            .try_normalize()
            .unwrap_or(normal)
    }

    #[inline(always)]
    pub fn emission(&self, texels: &[Texel], uv: UvSample) -> Vec3 {
        self.emissive_factor * sample_texture(&self.emissive_texture, texels, uv).truncate()
    }
}

#[inline(always)]
fn sample_texture(texture: &Option<Texture>, texels: &[Texel], uv: UvSample) -> Vec4 {
    match texture {
        Some(texture) => texture.sample(texels, uv),
        None => Vec4::ONE,
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
