use std::f32::consts::PI;

use glam::Vec3;

use crate::{
    lighting::Light,
    material::Material,
    pipeline::{FragmentAttributes, Shading},
    shadow::LightShadow,
    texture::{Sampling, Texture, UvSample},
};

/// Straight-line shader regions in program order; trace records store their numbers.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Site {
    Input,
    BaseColor,
    BaseColorLevel,
    Normal,
    NormalLevel,
    MetallicRoughness,
    MetallicRoughnessLevel,
    Emissive,
    EmissiveLevel,
    NormalMap,
    Surface,
    Light,
    Shadow,
    Brdf,
    Output,
}

/// One lane's memory operations and arithmetic in a region.
#[derive(Clone, Copy)]
pub(crate) struct Ops {
    pub loads: u16,
    pub stores: u16,
    pub fp: u16,
    pub sfu: u16,
    pub int: u16,
}

pub(crate) const fn ops(loads: u16, stores: u16, fp: u16, sfu: u16, int: u16) -> Ops {
    Ops {
        loads,
        stores,
        fp,
        sfu,
        int,
    }
}

/// Receives each region a lane executes; `()` records nothing.
pub(crate) trait Trace {
    /// False skips computing what regions read, which CUDA cannot always eliminate.
    const ENABLED: bool = true;

    fn op(&mut self, site: Site, light: u32, ops: Ops);
}

impl Trace for () {
    const ENABLED: bool = false;

    fn op(&mut self, _: Site, _: u32, _: Ops) {}
}

// Source-level estimates of the work below, not compiled instruction counts.
const NEAREST: Ops = ops(1, 0, 29, 1, 30);
const BILINEAR: Ops = ops(4, 0, 52, 0, 40);
const TRILINEAR: Ops = ops(4, 0, 69, 1, 57);
const SECOND_LEVEL: Ops = ops(4, 0, 64, 0, 52);
const NORMAL_MAP: Ops = ops(0, 0, 67, 4, 0);
const SURFACE: Ops = ops(0, 0, 47, 2, 0);
const DIRECTIONAL_LIGHT: Ops = ops(1, 0, 6, 0, 0);
const AREA_LIGHT: Ops = ops(1, 0, 30, 3, 0);
const DIRECTIONAL_SHADOW: Ops = ops(5, 0, 76, 2, 35);
const AREA_SHADOW: Ops = ops(5, 0, 102, 2, 50);
const BRDF: Ops = ops(0, 0, 112, 7, 0);

// CUDA shading must inline this to address texels and shadow depths as global memory.
#[inline(always)]
pub(crate) fn shade(
    shading: &Shading<'_>,
    material: &Material,
    attributes: &FragmentAttributes,
    front_facing: bool,
    trace: &mut impl Trace,
) -> Vec3 {
    let position = attributes.world_position;
    let uv = attributes.uv;
    trace_texture(trace, Site::BaseColor, &material.base_color_texture, uv);
    trace_texture(trace, Site::Normal, &material.normal_texture, uv);
    trace_texture(
        trace,
        Site::MetallicRoughness,
        &material.metallic_roughness_texture,
        uv,
    );
    trace_texture(trace, Site::Emissive, &material.emissive_texture, uv);
    let surface = material.sample(shading.texels, uv);
    if material.normal_texture.is_some() {
        trace.op(Site::NormalMap, 0, NORMAL_MAP);
    }
    let normal = material.shading_normal(surface.normal, attributes.normal, attributes.tangent);
    let normal = if front_facing { normal } else { -normal };
    let view = (shading.camera_position - position).normalize_or_zero();
    trace.op(Site::Surface, 0, SURFACE);
    let brdf = Brdf::new(surface.base_color, surface.metallic, surface.roughness);
    let mut reflected = Vec3::ZERO;
    // A wrapping count avoids enumerate's overflow check, which CUDA builds keep.
    let mut next_index = 0u32;
    for light in shading.lights {
        let index = next_index;
        next_index = next_index.wrapping_add(1);
        let light_ops = match light.light {
            Light::Directional(_) => DIRECTIONAL_LIGHT,
            Light::Area(_) => AREA_LIGHT,
        };
        trace.op(Site::Light, index, light_ops);
        let Some(sample) = light.light.sample(position) else {
            continue;
        };
        let cosine = normal.dot(sample.direction);
        // The BRDF is zero here, so the light contributes nothing and needs no shadow test.
        if cosine <= 0.0 {
            continue;
        }
        let visibility = match &light.shadow {
            Some(shadow) => {
                let shadow_ops = match shadow {
                    LightShadow::Directional(_) => DIRECTIONAL_SHADOW,
                    LightShadow::Area { .. } => AREA_SHADOW,
                };
                trace.op(Site::Shadow, index, shadow_ops);
                shadow.visibility(shading.maps, shading.depths, position)
            }
            None => 1.0,
        };
        if visibility > 0.0 {
            trace.op(Site::Brdf, index, BRDF);
            reflected += brdf.evaluate(normal, view, sample.direction)
                * (sample.irradiance * visibility)
                * cosine.max(0.0);
        }
    }
    reflected + surface.emission
}

/// Record a texture sample, then the second mip level a trilinear sample may read.
#[inline(always)]
fn trace_texture<T: Trace>(trace: &mut T, site: Site, texture: &Option<Texture>, uv: UvSample) {
    if !T::ENABLED {
        return;
    }
    if let Some(texture) = texture {
        let (sampling, second_level) = texture.reads(uv);
        let first = match sampling {
            Sampling::None => NEAREST,
            Sampling::Bilinear => BILINEAR,
            Sampling::Trilinear => TRILINEAR,
        };
        trace.op(site, 0, first);
        if second_level {
            let level = match site {
                Site::BaseColor => Site::BaseColorLevel,
                Site::Normal => Site::NormalLevel,
                Site::MetallicRoughness => Site::MetallicRoughnessLevel,
                _ => Site::EmissiveLevel,
            };
            trace.op(level, 0, SECOND_LEVEL);
        }
    }
}

const DIELECTRIC_F0: f32 = 0.04;

struct Brdf {
    diffuse: Vec3,
    f0: Vec3,
    alpha_squared: f32,
}

impl Brdf {
    fn new(base_color: Vec3, metallic: f32, roughness: f32) -> Self {
        Self {
            diffuse: base_color * (1.0 - metallic),
            f0: Vec3::splat(DIELECTRIC_F0).lerp(base_color, metallic),
            alpha_squared: roughness.clamp(0.045, 1.0).powi(4),
        }
    }

    fn evaluate(&self, normal: Vec3, view: Vec3, light: Vec3) -> Vec3 {
        let no_v = normal.dot(view).clamp(0.0, 1.0);
        let no_l = normal.dot(light).clamp(0.0, 1.0);
        if no_v == 0.0 || no_l == 0.0 {
            return Vec3::ZERO;
        }
        let Some(half) = (view + light).try_normalize() else {
            return Vec3::ZERO;
        };
        let no_h = normal.dot(half).clamp(0.0, 1.0);
        let vo_h = view.dot(half).clamp(0.0, 1.0);
        let a2 = self.alpha_squared;
        // Cross product avoids cancellation near the GGX peak.
        let denominator = normal.cross(half).length_squared() + a2 * no_h * no_h;
        let distribution = a2 / (PI * denominator * denominator);
        let visibility = 0.5
            / (no_l * (no_v * no_v * (1.0 - a2) + a2).sqrt()
                + no_v * (no_l * no_l * (1.0 - a2) + a2).sqrt());
        let fresnel_weight = (1.0 - vo_h).powi(5);
        let fresnel = self.f0 + (Vec3::ONE - self.f0) * fresnel_weight;
        let dielectric_fresnel = DIELECTRIC_F0 + (1.0 - DIELECTRIC_F0) * fresnel_weight;
        self.diffuse * ((1.0 - dielectric_fresnel) / PI) + fresnel * (distribution * visibility)
    }
}
