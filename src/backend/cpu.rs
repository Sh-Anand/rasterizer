use std::io;

use crate::{
    asset::GltfAsset,
    coverage::rasterize_triangle,
    framebuffer::Framebuffer,
    material::Material,
    texture::encode_srgb,
    viewport::{ScreenScene, interpolate_uv},
};

pub struct RenderOutput {
    pub framebuffer: Framebuffer,
    pub covered_fragments: u64,
}

pub fn render(
    scene: &ScreenScene,
    asset: &GltfAsset,
    width: u32,
    height: u32,
) -> io::Result<RenderOutput> {
    let materials = asset
        .document
        .materials()
        .map(|material| Material::new(material, &asset.images))
        .collect::<io::Result<Vec<_>>>()?;
    let default_material = Material::default();
    let mut framebuffer = Framebuffer::new(width, height)?;
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
                        .sum::<f32>();
                    let [x, y] = sample.pixel;
                    let index = y as usize * width as usize + x as usize;
                    if depth < framebuffer.depth[index] {
                        let uv = interpolate_uv(vertices, sample.barycentric);
                        let color = material.base_color(uv);
                        if material.alpha_cutoff.is_some_and(|cutoff| color.w < cutoff) {
                            return;
                        }
                        framebuffer.depth[index] = depth;
                        framebuffer.color[index] = encode_srgb(color.truncate());
                    }
                },
            );
        }
    }
    Ok(RenderOutput {
        framebuffer,
        covered_fragments,
    })
}
