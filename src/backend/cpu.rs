use std::io;

mod capture;
mod fragment;
mod rasterization;
mod vertex;

pub use vertex::{reproject_scene, transform_scene};

pub(crate) use capture::CaptureScene;
use rasterization::{RasterPass, rasterize};

use crate::{
    clip::clip_scene,
    config::ShadowConfig,
    framebuffer::{Framebuffer, RenderOutput},
    lighting::{Light, Lighting, RenderLight},
    material::Material,
    shadow::{LightShadow, ShadowMap},
    vertex::ClipScene,
    viewport::{ScreenScene, project_scene},
};

pub fn render(
    scene: &ScreenScene,
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
    let mut framebuffer = Framebuffer::new(width, height)?;
    let covered_fragments = rasterize(
        &scene.primitives,
        materials,
        &mut framebuffer.depth,
        RasterPass::Camera,
        |index, material, attributes, base_color, front_facing| {
            framebuffer.color[index] =
                fragment::shade(&lighting, material, attributes, base_color, front_facing);
        },
    );
    Ok(RenderOutput {
        framebuffer,
        covered_fragments,
    })
}

pub fn prepare_lights(
    lights: Vec<Light>,
    scene: &ClipScene,
    materials: &[Material],
    config: &ShadowConfig,
) -> io::Result<Vec<RenderLight>> {
    lights
        .into_iter()
        .map(|light| {
            let shadow = if config.enabled {
                let mut shadow = LightShadow::new(scene, &light, config)?;
                for map in shadow.maps_mut() {
                    draw_shadow_map(scene, materials, map)?;
                }
                Some(shadow)
            } else {
                None
            };
            Ok(RenderLight { light, shadow })
        })
        .collect()
}

fn draw_shadow_map(
    scene: &ClipScene,
    materials: &[Material],
    shadow: &mut ShadowMap,
) -> io::Result<()> {
    let screen = {
        let clipped = clip_scene(&reproject_scene(scene, shadow.view_projection)?)?;
        project_scene(&clipped, shadow.depth.width, shadow.depth.height)?
    };
    rasterize(
        &screen.primitives,
        materials,
        &mut shadow.depth,
        RasterPass::Shadow(shadow.projection),
        |_, _, _, _, _| {},
    );
    Ok(())
}
