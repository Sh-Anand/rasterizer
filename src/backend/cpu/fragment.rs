use std::io;

use glam::Vec3;

use crate::{
    brdf::{Brdf, diffuse_color},
    lighting::{Lighting, visible_lights},
    lightmap::Lightmap,
    material::Material,
    pipeline::{Fragment, FragmentStage},
    viewport::FragmentAttributes,
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

pub(super) fn shade_capture(
    lightmap: &Lightmap,
    material: &Material,
    attributes: &FragmentAttributes,
    front_facing: bool,
) -> Vec3 {
    // Single-sided backs block light, but don't reflect it.
    if front_facing || material.double_sided {
        diffuse_color(
            material.base_color(attributes.uv).truncate(),
            material.metallic(attributes.metallic_roughness_uv),
        ) * lightmap.sample(attributes.lightmap_uv, !front_facing)
    } else {
        Vec3::ZERO
    }
}
