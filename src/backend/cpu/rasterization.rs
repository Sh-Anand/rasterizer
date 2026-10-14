use glam::Vec3;

use crate::{
    coverage::rasterize_triangle,
    framebuffer::DepthBuffer,
    material::Material,
    shadow::ShadowProjection,
    viewport::{FragmentAttributes, ScreenPrimitive, UvGradients, interpolate_attributes},
};

pub(super) enum RasterPass {
    Camera,
    Capture,
    Shadow(ShadowProjection),
}

// Emit each camera/capture fragment immediately after it passes depth and alpha tests.
// A later fragment may overwrite the same sample; this is not a final-winner buffer.
// Shadow passes only update depth and never invoke the callback.
pub(super) fn rasterize(
    primitives: &[ScreenPrimitive],
    materials: &[Material],
    depth_buffer: &mut DepthBuffer,
    pass: RasterPass,
    mut emit: impl FnMut(usize, &Material, &FragmentAttributes, Vec3, bool),
) -> u64 {
    let linear_far = match pass {
        RasterPass::Shadow(ShadowProjection::Perspective { far }) => Some(far),
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
                || (!front_facing && !material.double_sided && !matches!(pass, RasterPass::Capture))
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
            let depth_bias = if matches!(pass, RasterPass::Shadow(_)) {
                let db = f64::from(depths[1]) - f64::from(depths[0]);
                let dc = f64::from(depths[2]) - f64::from(depths[0]);
                let dx = (db * ac.y - dc * ab.y) / area;
                let dy = (ab.x * dc - ac.x * db) / area;
                // A bilinear PCF tap can be up to one texel from the receiver.
                let bias = (dx.abs() + dy.abs()) as f32;
                if linear_far.is_some() { -bias } else { bias }
            } else {
                0.0
            };
            let gradients = UvGradients::new(vertices);
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
                    if matches!(pass, RasterPass::Shadow(_)) && material.alpha_cutoff.is_none() {
                        depth_buffer.values[index] = depth;
                        return;
                    }
                    let attributes =
                        interpolate_attributes(vertices, sample.barycentric, &gradients);
                    let base_color = material.base_color(attributes.uv);
                    if material
                        .alpha_cutoff
                        .is_some_and(|cutoff| base_color.w < cutoff)
                    {
                        return;
                    }
                    depth_buffer.values[index] = depth;
                    if !matches!(pass, RasterPass::Shadow(_)) {
                        emit(
                            index,
                            material,
                            &attributes,
                            base_color.truncate(),
                            front_facing,
                        );
                    }
                },
            );
        }
    }
    covered_fragments
}
