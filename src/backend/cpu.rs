use crate::{coverage::rasterize_triangle, viewport::ScreenScene};

pub fn render(scene: &ScreenScene, width: u32, height: u32) -> u64 {
    let mut fragments = 0;
    for primitive in &scene.primitives {
        for triangle in &primitive.triangles {
            let vertices = triangle
                .indices
                .map(|index| primitive.vertices[index as usize].position.truncate());
            rasterize_triangle(vertices, width, height, |_| fragments += 1);
        }
    }
    fragments
}
