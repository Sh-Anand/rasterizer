use std::{collections::HashMap, io, path::Path};

use glam::{Vec3, Vec4};

use crate::{
    appearance::{self, Appearance},
    config::Neural,
    geometry::GltfAsset,
    ntc::{self, NeuralSet},
    texture::{ColorSpace, Sampling, Texel, Texture, TextureLoader, UvSample},
    trace::{Site, TRACE, mark},
};

/// Returns the materials and the texel array their textures index. With `Neural::Textures`, a
/// material named M (or numbered M, unnamed) with a texture set `ntc/M.ntc` decodes it instead of
/// its base color, normal and metallic-roughness textures; materials sharing a file share one
/// set. With `Neural::Materials`, every material is a neural material with stand-in weights, and
/// only emission stays conventional.
pub fn load_materials(
    asset: &GltfAsset,
    sampling: Sampling,
    neural: Neural,
    ntc: &Path,
) -> io::Result<(Vec<Material>, Vec<Texel>)> {
    let mut textures = TextureLoader::new(&asset.images, sampling);
    let mut sets = HashMap::new();
    let latents =
        (neural == Neural::Materials).then(|| appearance::load_latents(&mut textures.texels));
    let materials = asset
        .document
        .materials()
        .enumerate()
        .map(|(index, material)| {
            let path = ntc.join(format!(
                "{}.ntc",
                material
                    .name()
                    .map_or_else(|| index.to_string(), str::to_owned)
            ));
            let network = match (neural, latents) {
                (Neural::Materials, Some(latents)) => Network::Material(Appearance::load(
                    index as u32,
                    latents,
                    &mut textures.texels,
                )),
                (Neural::Textures, _) if path.is_file() => {
                    let key = path.canonicalize()?;
                    Network::Textures(match sets.get(&key) {
                        Some(&set) => set,
                        None => {
                            let set = ntc::load(&path, &mut textures.texels)?;
                            sets.insert(key, set);
                            set
                        }
                    })
                }
                _ => Network::None,
            };
            Material::new(material, &mut textures, network)
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

/// The network a material evaluates, if any.
#[derive(Clone, Copy, PartialEq)]
pub enum Network {
    None,
    /// Replaces the base color, normal and metallic-roughness textures.
    Textures(NeuralSet),
    /// Replaces those textures and the BRDF.
    Material(Appearance),
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
    pub network: Network,
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
            network: Network::None,
            double_sided: false,
        }
    }
}

impl Material {
    fn new(
        material: gltf::Material<'_>,
        textures: &mut TextureLoader<'_>,
        network: Network,
    ) -> io::Result<Self> {
        let pbr = material.pbr_metallic_roughness();
        let normal = material.normal_texture();
        let conventional = network == Network::None;
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
            network,
            double_sided: material.double_sided(),
        })
    }

    /// Reads every texture at one UV, as a texture-set decoder produces all channels at once.
    /// Every lane of a warp must call this together: the tensor-core decoder is warp-wide.
    #[inline(always)]
    pub(crate) fn sample(&self, texels: &[Texel], uv: UvSample, sites: &mut u32) -> MaterialSample {
        let set = match &self.network {
            Network::Textures(set) => Some(set),
            _ => None,
        };
        let c = ntc::sample(texels, set, uv);
        // ORM channels put roughness and metallic where metallic-roughness textures do.
        let (base_color, normal, metallic_roughness) = if set.is_some() {
            mark(sites, Site::Latents);
            mark(sites, Site::Network);
            (
                Vec3::new(c[0], c[1], c[2]),
                Vec3::new(c[6], c[7], c[8]),
                Vec3::new(c[3], c[4], c[5]),
            )
        } else {
            (
                sample_texture(
                    &self.base_color_texture,
                    texels,
                    uv,
                    sites,
                    [Site::BaseColor, Site::BaseColorLevel],
                ),
                sample_texture(
                    &self.normal_texture,
                    texels,
                    uv,
                    sites,
                    [Site::Normal, Site::NormalLevel],
                ),
                sample_texture(
                    &self.metallic_roughness_texture,
                    texels,
                    uv,
                    sites,
                    [Site::MetallicRoughness, Site::MetallicRoughnessLevel],
                ),
            )
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
                * sample_texture(
                    &self.emissive_texture,
                    texels,
                    uv,
                    sites,
                    [Site::Emissive, Site::EmissiveLevel],
                ),
        }
    }

    pub fn has_normal_map(&self) -> bool {
        self.normal_texture.is_some() || matches!(self.network, Network::Textures(_))
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

/// Samples a texture, marking its region and, if a trilinear sample reads a second mip level,
/// that level's region.
#[inline(always)]
fn sample_texture(
    texture: &Option<Texture>,
    texels: &[Texel],
    uv: UvSample,
    sites: &mut u32,
    [site, level]: [Site; 2],
) -> Vec3 {
    let Some(texture) = texture else {
        return Vec3::ONE;
    };
    mark(sites, site);
    if TRACE && texture.reads_second_level(uv) {
        mark(sites, level);
    }
    texture.sample(texels, uv).truncate()
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
