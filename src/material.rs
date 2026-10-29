use std::{collections::HashMap, io, path::Path};

use glam::{Vec3, Vec4};

use crate::{
    appearance::{self, Appearance},
    geometry::GltfAsset,
    ntc::{self, NeuralSet},
    texture::{ColorSpace, Sampling, Texel, Texture, TextureLoader, UvSample},
    trace::{Site, TRACE, mark},
};

/// Returns the materials and the texel array their textures index. With `ntc`, a material
/// named M (or numbered M, unnamed) with a texture set `ntc/M.ntc` decodes it instead of its base
/// color, normal and metallic-roughness textures; materials sharing a file share one set. With
/// `neural`, every material is a neural material with stand-in weights instead, and only
/// emission stays conventional.
pub fn load_materials(
    asset: &GltfAsset,
    sampling: Sampling,
    ntc: Option<&Path>,
    neural: bool,
) -> io::Result<(Vec<Material>, Vec<Texel>)> {
    let mut textures = TextureLoader::new(&asset.images, sampling);
    let mut sets = HashMap::new();
    let latents = neural.then(|| appearance::load_latents(&mut textures.texels));
    let materials = asset
        .document
        .materials()
        .enumerate()
        .map(|(index, material)| {
            let name = material
                .name()
                .map_or_else(|| index.to_string(), str::to_owned);
            let appearance = latents
                .map(|latents| Appearance::load(index as u32, latents, &mut textures.texels));
            let neural = ntc
                .filter(|_| appearance.is_none())
                .map(|dir| dir.join(format!("{name}.ntc")))
                .filter(|path| path.is_file())
                .map(|path| -> io::Result<NeuralSet> {
                    let key = path.canonicalize()?;
                    if let Some(&set) = sets.get(&key) {
                        return Ok(set);
                    }
                    let set = ntc::load(&path, &mut textures.texels)?;
                    sets.insert(key, set);
                    Ok(set)
                })
                .transpose()?;
            Material::new(material, &mut textures, neural, appearance)
        })
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
    /// Replaces the base color, normal and metallic-roughness textures.
    pub neural: Option<NeuralSet>,
    /// Replaces those textures and the BRDF.
    pub appearance: Option<Appearance>,
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
            neural: None,
            appearance: None,
            double_sided: false,
        }
    }
}

impl Material {
    fn new(
        material: gltf::Material<'_>,
        textures: &mut TextureLoader<'_>,
        neural: Option<NeuralSet>,
        appearance: Option<Appearance>,
    ) -> io::Result<Self> {
        let pbr = material.pbr_metallic_roughness();
        let normal = material.normal_texture();
        let conventional = neural.is_none() && appearance.is_none();
        Ok(Self {
            base_color_factor: Vec4::from(pbr.base_color_factor()).truncate(),
            base_color_texture: load_texture(
                pbr.base_color_texture().filter(|_| conventional),
                textures,
                ColorSpace::Srgb,
            )?,
            normal_scale: normal.as_ref().map_or(1.0, |info| info.scale()),
            normal_texture: normal
                .filter(|_| conventional)
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
                pbr.metallic_roughness_texture().filter(|_| conventional),
                textures,
                ColorSpace::Linear,
            )?,
            emissive_factor: Vec3::from(material.emissive_factor())
                * material.emissive_strength().unwrap_or(1.0),
            emissive_texture: load_texture(
                material.emissive_texture(),
                textures,
                ColorSpace::Srgb,
            )?,
            neural,
            appearance,
            double_sided: material.double_sided(),
        })
    }

    /// Reads every texture at one UV, as a texture-set decoder produces all channels at once.
    /// Every lane of a warp must call this together: the tensor-core decoder is warp-wide.
    #[inline(always)]
    pub(crate) fn sample(&self, texels: &[Texel], uv: UvSample, sites: &mut u32) -> MaterialSample {
        trace_texture(
            sites,
            &self.base_color_texture,
            uv,
            Site::BaseColor,
            Site::BaseColorLevel,
        );
        trace_texture(
            sites,
            &self.normal_texture,
            uv,
            Site::Normal,
            Site::NormalLevel,
        );
        trace_texture(
            sites,
            &self.metallic_roughness_texture,
            uv,
            Site::MetallicRoughness,
            Site::MetallicRoughnessLevel,
        );
        trace_texture(
            sites,
            &self.emissive_texture,
            uv,
            Site::Emissive,
            Site::EmissiveLevel,
        );
        let c = ntc::sample(texels, self.neural.as_ref(), uv);
        // ORM channels put roughness and metallic where metallic-roughness textures do.
        let (base_color, normal, metallic_roughness) = match &self.neural {
            Some(_) => {
                mark(sites, Site::Latents);
                mark(sites, Site::Network);
                (
                    Vec3::new(c[0], c[1], c[2]),
                    Vec3::new(c[6], c[7], c[8]),
                    Vec3::new(c[3], c[4], c[5]),
                )
            }
            None => (
                sample_texture(&self.base_color_texture, texels, uv).truncate(),
                sample_texture(&self.normal_texture, texels, uv).truncate(),
                sample_texture(&self.metallic_roughness_texture, texels, uv).truncate(),
            ),
        };
        let mut normal = normal * 2.0 - Vec3::ONE;
        normal.x *= self.normal_scale;
        normal.y *= self.normal_scale;
        MaterialSample {
            base_color: self.base_color_factor * base_color,
            normal,
            metallic: self.metallic_factor * metallic_roughness.z,
            roughness: self.roughness_factor * metallic_roughness.y,
            emission: self.emissive_factor
                * sample_texture(&self.emissive_texture, texels, uv).truncate(),
        }
    }

    pub fn has_normal_map(&self) -> bool {
        self.normal_texture.is_some() || self.neural.is_some()
    }

    /// Applies a sampled tangent-space normal.
    pub fn shading_normal(&self, sampled: Vec3, normal: Vec3, tangent: Vec4) -> Vec3 {
        if !self.has_normal_map() || self.normal_scale == 0.0 || tangent.w == 0.0 {
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
}

/// Marks a texture's first mip level, then the second a trilinear sample may read.
#[inline(always)]
fn trace_texture(
    sites: &mut u32,
    texture: &Option<Texture>,
    uv: UvSample,
    site: Site,
    level: Site,
) {
    if !TRACE {
        return;
    }
    if let Some(texture) = texture {
        mark(sites, site);
        if texture.reads_second_level(uv) {
            mark(sites, level);
        }
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
