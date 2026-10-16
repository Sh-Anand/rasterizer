use std::io;

use glam::Vec3;
use rayon::prelude::*;

mod rasterization;
mod vertex;

pub use rasterization::CpuRasterization;
pub use vertex::CpuVertex;

use crate::{
    fragment::shade,
    framebuffer::{Framebuffer, RenderOutput},
    geometry::ClipScene,
    lighting::{Lighting, visible_lights},
    material::Material,
    pipeline::{Fragment, FragmentStage},
    shadow::LightShadow,
};
use rasterization::{DepthSlice, project, rasterize};

pub struct CpuFragment;

impl FragmentStage for CpuFragment {
    fn shade(
        &mut self,
        fragments: &[Fragment],
        materials: &[Material],
        lighting: &Lighting<'_>,
    ) -> io::Result<Vec<Vec3>> {
        let default_material = Material::default();
        Ok(fragments
            .par_iter()
            .map(|fragment| {
                let material = fragment
                    .material_index
                    .map_or(&default_material, |index| &materials[index]);
                shade(
                    lighting.camera_position,
                    material,
                    &fragment.attributes,
                    fragment.front_facing,
                    visible_lights(
                        lighting.lights,
                        fragment.attributes.world_position,
                        LightShadow::visibility,
                    ),
                )
            })
            .collect())
    }
}

/// Immediate shading in scene order, independent of fragment collection.
pub fn render_reference(
    scene: &ClipScene,
    materials: &[Material],
    width: u32,
    height: u32,
    lighting: Lighting<'_>,
) -> io::Result<RenderOutput> {
    if !lighting.camera_position.is_finite() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Non-finite camera position",
        ));
    }
    let mut output = RenderOutput {
        framebuffer: Framebuffer::new(width, height)?,
        covered_fragments: 0,
        screen_vertices: 0,
        screen_triangles: 0,
    };
    let default_material = Material::default();
    for primitive in &scene.primitives {
        let material = primitive
            .source
            .material_index
            .map_or(&default_material, |i| &materials[i]);
        let screen = project(primitive, width, height)?;
        output.screen_vertices += screen.vertices.len();
        output.screen_triangles += screen.triangles.len();
        output.covered_fragments += rasterize(
            &screen,
            &screen.triangles,
            material.double_sided,
            DepthSlice {
                width,
                y_start: 0,
                values: &mut output.framebuffer.depth.values,
            },
            None,
            |index, attributes, front_facing| {
                output.framebuffer.color[index] = shade(
                    lighting.camera_position,
                    material,
                    attributes,
                    front_facing,
                    visible_lights(
                        lighting.lights,
                        attributes.world_position,
                        LightShadow::visibility,
                    ),
                );
            },
        );
    }
    Ok(output)
}
