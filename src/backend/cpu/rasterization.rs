use std::io;

use crate::{
    clip::clip_primitive,
    coverage::rasterize_triangle,
    framebuffer::DepthBuffer,
    material::Material,
    pipeline::{Fragment, RasterizationStage, RasterizedBatch},
    shadow::ShadowProjection,
    vertex::ClipPrimitive,
    viewport::{
        FragmentAttributes, ScreenPrimitive, UvGradients, interpolate_attributes, project_primitive,
    },
};

#[derive(Clone, Copy)]
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
    mut emit: impl FnMut(usize, &Material, &FragmentAttributes, bool),
) -> u64 {
    let default_material = Material::default();
    primitives
        .iter()
        .map(|primitive| {
            let material = primitive
                .source
                .material_index
                .map_or(&default_material, |index| &materials[index]);
            rasterize_primitive(primitive, material, depth_buffer, pass, &mut emit)
        })
        .sum()
}

fn rasterize_primitive(
    primitive: &ScreenPrimitive,
    material: &Material,
    depth_buffer: &mut DepthBuffer,
    pass: RasterPass,
    mut emit: impl FnMut(usize, &Material, &FragmentAttributes, bool),
) -> u64 {
    let linear_far = match pass {
        RasterPass::Shadow(ShadowProjection::Perspective { far }) => Some(far),
        _ => None,
    };
    let (width, height) = (depth_buffer.width, depth_buffer.height);
    let mut covered_fragments = 0;
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
                let attributes = interpolate_attributes(vertices, sample.barycentric, &gradients);
                if material
                    .alpha_cutoff
                    .is_some_and(|cutoff| material.base_color(attributes.uv).w < cutoff)
                {
                    return;
                }
                depth_buffer.values[index] = depth;
                if !matches!(pass, RasterPass::Shadow(_)) {
                    emit(index, material, &attributes, front_facing);
                }
            },
        );
    }
    covered_fragments
}

pub struct CpuRasterization;

impl RasterizationStage for CpuRasterization {
    fn rasterize(
        &mut self,
        primitives: &[&ClipPrimitive],
        material: &Material,
        depth: &mut DepthBuffer,
    ) -> io::Result<RasterizedBatch> {
        let mut batch = RasterizedBatch::default();
        let mut slot_for_sample = Vec::new();
        slot_for_sample
            .try_reserve_exact(depth.values.len())
            .map_err(io::Error::other)?;
        slot_for_sample.resize(depth.values.len(), usize::MAX);
        for primitive in primitives {
            let clipped = clip_primitive(primitive)?;
            let screen = project_primitive(&clipped, depth.width, depth.height)?;
            batch.screen_vertices += screen.vertices.len();
            batch.screen_triangles += screen.triangles.len();
            batch.covered_fragments += rasterize_primitive(
                &screen,
                material,
                depth,
                RasterPass::Camera,
                |sample_index, _, attributes, front_facing| {
                    let fragment = Fragment {
                        sample_index,
                        attributes: *attributes,
                        front_facing,
                    };
                    // The callback runs only after depth and alpha tests succeed.
                    // Keep the latest winner in its first allocated slot, avoiding
                    // both stale fragments and a separate compaction pass.
                    let slot = &mut slot_for_sample[sample_index];
                    if *slot == usize::MAX {
                        *slot = batch.fragments.len();
                        batch.fragments.push(fragment);
                    } else {
                        batch.fragments[*slot] = fragment;
                    }
                },
            );
        }
        Ok(batch)
    }
}
