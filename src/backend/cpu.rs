use std::io;

use glam::Vec3;

use crate::{
    asset::GltfAsset,
    clip::clip_scene,
    config::ShadowConfig,
    coverage::rasterize_triangle,
    framebuffer::{DepthBuffer, Framebuffer},
    lighting::DirectionalLight,
    material::Material,
    shadow::ShadowMap,
    texture::encode_srgb,
    vertex::{ClipScene, reproject_scene},
    viewport::{ScreenScene, interpolate_attributes, project_scene},
};

pub struct RenderOutput {
    pub framebuffer: Framebuffer,
    pub covered_fragments: u64,
}

struct Shading<'a> {
    color: &'a mut [[u8; 3]],
    light: &'a DirectionalLight,
    shadow: Option<&'a ShadowMap>,
}

pub fn render(
    scene: &ScreenScene,
    asset: &GltfAsset,
    width: u32,
    height: u32,
    light: &DirectionalLight,
    shadow: Option<&ShadowMap>,
) -> io::Result<RenderOutput> {
    let mut framebuffer = Framebuffer::new(width, height)?;
    let covered_fragments = draw(
        scene,
        asset,
        &mut framebuffer.depth,
        Some(Shading {
            color: &mut framebuffer.color,
            light,
            shadow,
        }),
    )?;
    Ok(RenderOutput {
        framebuffer,
        covered_fragments,
    })
}

pub fn render_shadow_map(
    scene: &ClipScene,
    asset: &GltfAsset,
    light: &DirectionalLight,
    config: &ShadowConfig,
) -> io::Result<ShadowMap> {
    let mut shadow = ShadowMap::new(scene, light, config)?;
    let screen = {
        let clipped = clip_scene(&reproject_scene(scene, shadow.view_projection)?)?;
        project_scene(&clipped, shadow.depth.width, shadow.depth.height)?
    };
    draw(&screen, asset, &mut shadow.depth, None)?;
    Ok(shadow)
}

fn draw(
    scene: &ScreenScene,
    asset: &GltfAsset,
    depth_buffer: &mut DepthBuffer,
    mut shading: Option<Shading<'_>>,
) -> io::Result<u64> {
    let materials = asset
        .document
        .materials()
        .map(|material| Material::new(material, &asset.images))
        .collect::<io::Result<Vec<_>>>()?;
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
            // Bound the depth change between a sample and its texel center.
            let depth_bias = if shading.is_none() {
                let dx = (ab.z * ac.y - ac.z * ab.y) / area;
                let dy = (ab.x * ac.z - ac.x * ab.z) / area;
                (0.5 * (dx.abs() + dy.abs())) as f32
            } else {
                0.0
            };
            rasterize_triangle(
                vertices.map(|v| v.position.truncate()),
                width,
                height,
                |sample| {
                    covered_fragments += 1;
                    let depth = vertices
                        .iter()
                        .zip(sample.barycentric)
                        .map(|(vertex, weight)| vertex.position.z * weight)
                        .sum::<f32>()
                        + depth_bias;
                    let [x, y] = sample.pixel;
                    let index = y as usize * width as usize + x as usize;
                    if !depth.is_finite() || depth >= depth_buffer.values[index] {
                        return;
                    }
                    if shading.is_none() && material.alpha_cutoff.is_none() {
                        depth_buffer.values[index] = depth;
                        return;
                    }
                    let attributes = interpolate_attributes(vertices, sample.barycentric);
                    let color = material.base_color(attributes.uv);
                    if material.alpha_cutoff.is_some_and(|cutoff| color.w < cutoff) {
                        return;
                    }
                    depth_buffer.values[index] = depth;
                    if let Some(shading) = &mut shading {
                        let normal = if front_facing {
                            attributes.normal
                        } else {
                            -attributes.normal
                        };
                        let diffuse = if shading
                            .shadow
                            .is_some_and(|map| map.is_shadowed(attributes.world_position))
                        {
                            Vec3::ZERO
                        } else {
                            shading.light.shade(color.truncate(), normal)
                        };
                        shading.color[index] =
                            encode_srgb(diffuse + material.emission(attributes.emissive_uv));
                    }
                },
            );
        }
    }
    Ok(covered_fragments)
}
