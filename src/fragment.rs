use std::f32::consts::PI;

use glam::Vec3;

use crate::{
    lighting::RenderLight, material::Material, rasterization::FragmentAttributes,
    shadow::ShadowMap, texture::Texel,
};

/// Fragment-shading resources other than the material table.
#[derive(Clone, Copy)]
pub struct Shading<'a> {
    /// Every material texture's mip levels.
    pub texels: &'a [Texel],
    pub lights: &'a [RenderLight],
    pub maps: &'a [ShadowMap],
    /// Every shadow map's depths.
    pub depths: &'a [f32],
    pub camera_position: Vec3,
}

// CUDA shading must inline this to address texels and shadow depths as global memory.
#[inline(always)]
pub(crate) fn shade(
    shading: &Shading<'_>,
    material: &Material,
    attributes: &FragmentAttributes,
    front_facing: bool,
) -> Vec3 {
    let position = attributes.world_position;
    let surface = material.sample(shading.texels, attributes.uv);
    let normal = material.shading_normal(surface.normal, attributes.normal, attributes.tangent);
    let normal = if front_facing { normal } else { -normal };
    let view = (shading.camera_position - position).normalize_or_zero();
    let brdf = Brdf::new(surface.base_color, surface.metallic, surface.roughness);
    let mut reflected = Vec3::ZERO;
    for light in shading.lights {
        let Some(sample) = light.light.sample(position) else {
            continue;
        };
        let cosine = normal.dot(sample.direction);
        // The BRDF is zero here, so the light contributes nothing and needs no shadow test.
        if cosine <= 0.0 {
            continue;
        }
        let visibility = match &light.shadow {
            Some(shadow) => shadow.visibility(shading.maps, shading.depths, position),
            None => 1.0,
        };
        if visibility > 0.0 {
            reflected += brdf.evaluate(normal, view, sample.direction)
                * (sample.irradiance * visibility)
                * cosine.max(0.0);
        }
    }
    reflected + surface.emission
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
