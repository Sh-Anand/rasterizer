use std::io;

use crate::{
    coverage::rasterize_triangle, framebuffer::Framebuffer, geometry::PrimitiveSource,
    viewport::ScreenScene,
};

pub struct RenderOutput {
    pub framebuffer: Framebuffer,
    pub covered_fragments: u64,
}

pub fn render(scene: &ScreenScene, width: u32, height: u32) -> io::Result<RenderOutput> {
    let mut framebuffer = Framebuffer::new(width, height)?;
    let mut covered_fragments = 0;
    for primitive in &scene.primitives {
        for triangle in &primitive.triangles {
            let vertices = triangle
                .indices
                .map(|index| primitive.vertices[index as usize].position);
            let color = triangle_color(primitive.source, triangle.source_index);
            rasterize_triangle(vertices.map(|v| v.truncate()), width, height, |sample| {
                covered_fragments += 1;
                let depth = vertices
                    .iter()
                    .zip(sample.barycentric)
                    .map(|(vertex, weight)| vertex.z * weight)
                    .sum::<f32>();
                let [x, y] = sample.pixel;
                let index = y as usize * width as usize + x as usize;
                if depth < framebuffer.depth[index] {
                    framebuffer.depth[index] = depth;
                    framebuffer.color[index] = color;
                }
            });
        }
    }
    Ok(RenderOutput {
        framebuffer,
        covered_fragments,
    })
}

fn triangle_color(source: PrimitiveSource, triangle_index: usize) -> [u8; 3] {
    let mut hash = 2166136261u32;
    for index in [
        source.node_index,
        source.mesh_index,
        source.primitive_index,
        triangle_index,
    ] {
        hash = (hash ^ index as u32).wrapping_mul(16777619);
    }
    hash ^= hash >> 16;
    [0, 8, 16].map(|shift| 64 + ((hash >> shift) & 127) as u8)
}
