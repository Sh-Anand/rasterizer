use std::f32::consts::PI;

use glam::Vec3;

use crate::{
    lighting::Light,
    material::{Material, MaterialSample, Network},
    rasterization::FragmentAttributes,
    texture::Texel,
    trace::{Site, mark},
};

/// Lighting resources for shading.
#[derive(Clone, Copy)]
pub struct Shading<'a> {
    pub light: Light,
    /// The light's shadow-map depths.
    pub depths: &'a [f32],
    pub camera_position: Vec3,
    /// Neural materials' weights and latents.
    pub texels: &'a [Texel],
}

/// Lights a surface whose material the caller sampled once at its UV, returning its light and
/// the weight of light it scatters from a cosine-weighted direction (its diffuse albedo). A
/// neural material samples its own direction and decodes the BSDF there and toward the light.
// CUDA shading must inline this to address shadow depths as global memory.
#[inline(always)]
pub(crate) fn shade(
    shading: &Shading<'_>,
    material: &Material,
    surface: &MaterialSample,
    attributes: &FragmentAttributes,
    front_facing: bool,
    sites: &mut u32,
) -> (Vec3, Vec3) {
    let position = attributes.world_position;
    let view = (shading.camera_position - position).normalize_or_zero();
    let light = shading.light.direction;
    if let Network::Material(appearance) = &material.network {
        let normal = if front_facing {
            attributes.normal
        } else {
            -attributes.normal
        };
        let point = appearance.sample(shading.texels, attributes, normal, view, sites);
        let mut reflected = Vec3::ZERO;
        if let Some((_, irradiance)) = lit(shading, position, normal, sites) {
            reflected += appearance.eval(shading.texels, &point, light, sites) * irradiance;
        }
        return (reflected + surface.emission, point.weight);
    }
    if material.has_normal_map() {
        mark(sites, Site::NormalMap);
    }
    mark(sites, Site::Surface);
    let normal = material.shading_normal(surface.normal, attributes.normal, attributes.tangent);
    let normal = if front_facing { normal } else { -normal };
    let brdf = Brdf::new(surface.base_color, surface.metallic, surface.roughness);
    let mut reflected = Vec3::ZERO;
    if let Some((cosine, irradiance)) = lit(shading, position, normal, sites) {
        mark(sites, Site::Brdf);
        reflected += brdf.evaluate(normal, view, light) * irradiance * cosine.max(0.0);
    }
    (
        reflected + surface.emission,
        surface.base_color * (1.0 - surface.metallic),
    )
}

/// The light's cosine and shadowed irradiance at a surface it lights.
#[inline(always)]
fn lit(
    shading: &Shading<'_>,
    position: Vec3,
    normal: Vec3,
    sites: &mut u32,
) -> Option<(f32, Vec3)> {
    mark(sites, Site::Light);
    let light = &shading.light;
    let cosine = normal.dot(light.direction);
    // At or below the horizon the BRDF is zero, so the light needs no shadow test.
    if !(cosine > 0.0) {
        return None;
    }
    if light.shadow.is_some() {
        mark(sites, Site::Shadow);
    }
    let visibility = light.visibility(shading.depths, position);
    (visibility > 0.0).then_some((cosine, light.irradiance * visibility))
}

/// A PCG hash step as a uniform float in [0, 1).
#[inline(always)]
pub(crate) fn uniform(state: &mut u32) -> f32 {
    let x = state.wrapping_mul(747796405).wrapping_add(2891336453);
    let word = ((x >> ((x >> 28) + 4)) ^ x).wrapping_mul(277803737);
    *state = (word >> 22) ^ word;
    (*state >> 8) as f32 / 16777216.0
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
