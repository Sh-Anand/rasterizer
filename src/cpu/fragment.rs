use std::{f32::consts::PI, io};

use glam::Vec3;

use crate::{
    lighting::{Lighting, visible_lights},
    material::Material,
    pipeline::{Fragment, FragmentAttributes, FragmentStage},
};

pub struct CpuFragment;

impl FragmentStage for CpuFragment {
    fn shade(
        &mut self,
        fragments: &[Fragment],
        material: &Material,
        lighting: &Lighting<'_>,
    ) -> io::Result<Vec<Vec3>> {
        Ok(fragments
            .iter()
            .map(|fragment| {
                shade(
                    lighting,
                    material,
                    &fragment.attributes,
                    fragment.front_facing,
                )
            })
            .collect())
    }
}

pub(super) fn shade(
    lighting: &Lighting<'_>,
    material: &Material,
    attributes: &FragmentAttributes,
    front_facing: bool,
) -> Vec3 {
    let base_color = material.base_color(attributes.uv).truncate();
    let normal =
        material.shading_normal(attributes.normal_uv, attributes.normal, attributes.tangent);
    let normal = if front_facing { normal } else { -normal };
    let view = (lighting.camera_position - attributes.world_position).normalize_or_zero();
    let (metallic, roughness) = material.metallic_roughness(attributes.metallic_roughness_uv);
    let brdf = Brdf::new(base_color, metallic, roughness);
    let mut reflected = lighting.indirect.map_or(Vec3::ZERO, |map| {
        brdf.diffuse
            * map.sample(attributes.lightmap_uv, !front_facing)
            * material.occlusion(attributes.occlusion_uv)
    });
    for sample in visible_lights(lighting.lights, attributes.world_position) {
        reflected += brdf.evaluate(normal, view, sample.direction)
            * sample.irradiance
            * normal.dot(sample.direction).max(0.0);
    }
    reflected + material.emission(attributes.emissive_uv)
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
