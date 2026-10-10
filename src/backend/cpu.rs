use std::io;

use glam::Vec3;

mod capture;
pub(crate) use capture::CaptureScene;

use crate::{
    brdf::{Brdf, diffuse_color},
    clip::clip_scene,
    config::ShadowConfig,
    coverage::rasterize_triangle,
    framebuffer::{DepthBuffer, Framebuffer},
    lighting::{Light, LightSample},
    lightmap::Lightmap,
    material::Material,
    shadow::{LightShadow, ShadowMap, ShadowProjection},
    vertex::{ClipScene, reproject_scene},
    viewport::{
        FragmentAttributes, ScreenPrimitive, ScreenScene, interpolate_attributes, project_scene,
    },
};

pub struct RenderOutput {
    pub framebuffer: Framebuffer,
    pub covered_fragments: u64,
}

pub struct RenderLight {
    pub light: Light,
    pub shadow: Option<LightShadow>,
}

pub struct Lighting<'a> {
    pub indirect: Option<&'a Lightmap>,
    pub lights: &'a [RenderLight],
    pub camera_position: Vec3,
}

impl Lighting<'_> {
    fn shade(
        &self,
        material: &Material<'_>,
        attributes: &FragmentAttributes,
        base_color: Vec3,
        front_facing: bool,
    ) -> Vec3 {
        let normal =
            material.shading_normal(attributes.normal_uv, attributes.normal, attributes.tangent);
        let normal = if front_facing { normal } else { -normal };
        let view = (self.camera_position - attributes.world_position).normalize_or_zero();
        let (metallic, roughness) = material.metallic_roughness(attributes.metallic_roughness_uv);
        let brdf = Brdf::new(base_color, metallic, roughness);
        let mut reflected = self.indirect.map_or(Vec3::ZERO, |map| {
            brdf.diffuse
                * map.sample(attributes.lightmap_uv, !front_facing)
                * material.occlusion(attributes.occlusion_uv)
        });
        for sample in visible_lights(self.lights, attributes.world_position) {
            reflected += brdf.evaluate(normal, view, sample.direction)
                * sample.irradiance
                * normal.dot(sample.direction).max(0.0);
        }
        reflected + material.emission(attributes.emissive_uv)
    }
}

pub(crate) fn lambertian_lighting(lights: &[RenderLight], position: Vec3, normal: Vec3) -> Vec3 {
    visible_lights(lights, position)
        .map(|sample| sample.diffuse(normal))
        .sum()
}

fn visible_lights(lights: &[RenderLight], position: Vec3) -> impl Iterator<Item = LightSample> {
    lights
        .iter()
        .filter(move |light| {
            !light
                .shadow
                .as_ref()
                .is_some_and(|shadow| shadow.is_shadowed(position))
        })
        .filter_map(move |light| light.light.sample(position))
}

enum Pass<'a> {
    Shaded {
        color: &'a mut [Vec3],
        lighting: Lighting<'a>,
    },
    Capture {
        color: &'a mut [Vec3],
        lightmap: &'a Lightmap,
    },
    Shadow(ShadowProjection),
}

pub fn render(
    scene: &ScreenScene,
    materials: &[Material<'_>],
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
    let covered_fragments = draw(
        &scene.primitives,
        materials,
        &mut framebuffer.depth,
        Pass::Shaded {
            color: &mut framebuffer.color,
            lighting,
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
    materials: &[Material<'_>],
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
    materials: &[Material<'_>],
    shadow: &mut ShadowMap,
) -> io::Result<()> {
    let screen = {
        let clipped = clip_scene(&reproject_scene(scene, shadow.view_projection)?)?;
        project_scene(&clipped, shadow.depth.width, shadow.depth.height)?
    };
    draw(
        &screen.primitives,
        materials,
        &mut shadow.depth,
        Pass::Shadow(shadow.projection),
    );
    Ok(())
}

fn draw(
    primitives: &[ScreenPrimitive],
    materials: &[Material<'_>],
    depth_buffer: &mut DepthBuffer,
    mut pass: Pass<'_>,
) -> u64 {
    let linear_far = match &pass {
        Pass::Shadow(ShadowProjection::Perspective { far }) => Some(*far),
        _ => None,
    };
    let default_material = Material::default();
    let (width, height) = (depth_buffer.width, depth_buffer.height);
    let mut covered_fragments = 0;
    for primitive in primitives {
        let material = primitive
            .source
            .material_index
            .map_or(&default_material, |index| &materials[index]);
        for triangle in &primitive.triangles {
            let vertices = triangle
                .indices
                .map(|index| primitive.vertices[index as usize]);
            let [a, b, c] = vertices.map(|v| v.position.as_dvec3());
            let ab = b - a;
            let ac = c - a;
            let area = ab.truncate().perp_dot(ac.truncate());
            // The viewport's downward Y axis reverses the projected winding.
            let front_facing = (area < 0.0) != primitive.mirrored;
            if area == 0.0
                || (!front_facing
                    && !material.double_sided
                    && !matches!(pass, Pass::Capture { .. }))
            {
                continue;
            }
            let depths = vertices.map(|v| {
                if linear_far.is_some() {
                    v.inv_w
                } else {
                    v.position.z
                }
            });
            // Both projected depth and reciprocal light-view depth are affine in screen space.
            let depth_bias = if matches!(pass, Pass::Shadow(_)) {
                let db = f64::from(depths[1]) - f64::from(depths[0]);
                let dc = f64::from(depths[2]) - f64::from(depths[0]);
                let dx = (db * ac.y - dc * ab.y) / area;
                let dy = (ab.x * dc - ac.x * db) / area;
                let bias = (0.5 * (dx.abs() + dy.abs())) as f32;
                if linear_far.is_some() { -bias } else { bias }
            } else {
                0.0
            };
            rasterize_triangle(
                vertices.map(|v| v.position.truncate()),
                width,
                height,
                |sample| {
                    covered_fragments += 1;
                    let depth = depths
                        .iter()
                        .zip(sample.barycentric)
                        .map(|(depth, weight)| depth * weight)
                        .sum::<f32>()
                        + depth_bias;
                    let depth = if let Some(far) = linear_far {
                        if depth <= 0.0 {
                            return;
                        }
                        depth.recip() / far
                    } else {
                        depth
                    };
                    let [x, y] = sample.pixel;
                    let index = y as usize * width as usize + x as usize;
                    if !depth.is_finite() || depth >= depth_buffer.values[index] {
                        return;
                    }
                    if matches!(pass, Pass::Shadow(_)) && material.alpha_cutoff.is_none() {
                        depth_buffer.values[index] = depth;
                        return;
                    }
                    let attributes = interpolate_attributes(vertices, sample.barycentric);
                    let base_color = material.base_color(attributes.uv);
                    if material
                        .alpha_cutoff
                        .is_some_and(|cutoff| base_color.w < cutoff)
                    {
                        return;
                    }
                    depth_buffer.values[index] = depth;
                    match &mut pass {
                        Pass::Shaded { color, lighting } => {
                            color[index] = lighting.shade(
                                material,
                                &attributes,
                                base_color.truncate(),
                                front_facing,
                            );
                        }
                        Pass::Capture { color, lightmap } => {
                            // Single-sided backs block light, but don't reflect it.
                            color[index] = if front_facing || material.double_sided {
                                diffuse_color(
                                    base_color.truncate(),
                                    material.metallic(attributes.metallic_roughness_uv),
                                ) * lightmap.sample(attributes.lightmap_uv, !front_facing)
                            } else {
                                Vec3::ZERO
                            };
                        }
                        Pass::Shadow(_) => {}
                    }
                },
            );
        }
    }
    covered_fragments
}
