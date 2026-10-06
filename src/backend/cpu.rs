use std::io;

use glam::Vec3;

use crate::{
    asset::GltfAsset,
    clip::clip_scene,
    config::ShadowConfig,
    coverage::rasterize_triangle,
    framebuffer::{DepthBuffer, Framebuffer},
    lighting::Light,
    material::{Material, load_materials},
    shadow::{LightShadow, ShadowMap, ShadowProjection},
    texture::encode_srgb,
    vertex::{ClipScene, reproject_scene},
    viewport::{ScreenScene, interpolate_attributes, project_scene},
};

pub struct RenderOutput {
    pub framebuffer: Framebuffer,
    pub covered_fragments: u64,
}

pub struct RenderLight {
    pub light: Light,
    pub shadow: Option<LightShadow>,
}

enum Pass<'a> {
    Shaded {
        color: &'a mut [[u8; 3]],
        lights: &'a [RenderLight],
    },
    Shadow(ShadowProjection),
}

pub fn render(
    scene: &ScreenScene,
    asset: &GltfAsset,
    width: u32,
    height: u32,
    lights: &[RenderLight],
) -> io::Result<RenderOutput> {
    let mut framebuffer = Framebuffer::new(width, height)?;
    let covered_fragments = draw(
        scene,
        &load_materials(asset)?,
        &mut framebuffer.depth,
        Pass::Shaded {
            color: &mut framebuffer.color,
            lights,
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
    asset: &GltfAsset,
    config: &ShadowConfig,
) -> io::Result<Vec<RenderLight>> {
    let materials = load_materials(asset)?;
    lights
        .into_iter()
        .map(|light| {
            let shadow = if config.enabled {
                let mut shadow = LightShadow::new(scene, &light, config)?;
                for map in shadow.maps_mut() {
                    draw_shadow_map(scene, &materials, map)?;
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
        &screen,
        materials,
        &mut shadow.depth,
        Pass::Shadow(shadow.projection),
    );
    Ok(())
}

fn draw(
    scene: &ScreenScene,
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
    for primitive in &scene.primitives {
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
            if area == 0.0 || (!front_facing && !material.double_sided) {
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
                    if let Pass::Shaded { color, lights } = &mut pass {
                        let normal = if front_facing {
                            attributes.normal
                        } else {
                            -attributes.normal
                        };
                        let mut diffuse = Vec3::ZERO;
                        for light in *lights {
                            if light
                                .shadow
                                .as_ref()
                                .is_some_and(|map| map.is_shadowed(attributes.world_position))
                            {
                                continue;
                            }
                            diffuse += light.light.shade(
                                base_color.truncate(),
                                normal,
                                attributes.world_position,
                            );
                        }
                        color[index] =
                            encode_srgb(diffuse + material.emission(attributes.emissive_uv));
                    }
                },
            );
        }
    }
    covered_fragments
}
