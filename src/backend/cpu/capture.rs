use std::io;

use glam::{Mat4, Vec3};

use super::{Pass, draw};
use crate::{
    clip::clip_primitive,
    framebuffer::DepthBuffer,
    geometry::Triangle,
    lightmap::Lightmap,
    material::Material,
    vertex::{ClipPrimitive, ClipScene},
    viewport::project_primitive,
};

struct CaptureBatch<'a> {
    primitive: &'a ClipPrimitive,
    triangles: &'a [Triangle],
    bounds: [Vec3; 2],
}

pub(crate) struct CaptureScene<'a> {
    batches: Vec<CaptureBatch<'a>>,
}

impl<'a> CaptureScene<'a> {
    pub fn new(scene: &'a ClipScene) -> Self {
        let batches = scene
            .primitives
            .iter()
            .flat_map(|primitive| {
                primitive.triangles.chunks(64).map(move |triangles| {
                    let bounds = triangles.iter().flat_map(|t| t.indices).fold(
                        [Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)],
                        |[min, max], i| {
                            let p = primitive.vertices[i as usize].world_position;
                            [min.min(p), max.max(p)]
                        },
                    );
                    CaptureBatch {
                        primitive,
                        triangles,
                        bounds,
                    }
                })
            })
            .collect();
        Self { batches }
    }

    pub fn render(
        &self,
        materials: &[Material<'_>],
        lightmap: &Lightmap,
        view_projection: Mat4,
        resolution: u32,
    ) -> io::Result<Vec<Vec3>> {
        let mut depth = DepthBuffer::new(resolution, resolution)?;
        let mut color = vec![Vec3::ZERO; depth.values.len()];
        for batch in &self.batches {
            let [min, max] = batch.bounds;
            let corners: [_; 8] = std::array::from_fn(|i| {
                view_projection
                    * Vec3::new(
                        if i & 1 == 0 { min.x } else { max.x },
                        if i & 2 == 0 { min.y } else { max.y },
                        if i & 4 == 0 { min.z } else { max.z },
                    )
                    .extend(1.0)
            });
            if (0..3).any(|axis| {
                [-1.0, 1.0]
                    .into_iter()
                    .any(|sign| corners.iter().all(|p| p.w + sign * p[axis] < 0.0))
            }) {
                continue;
            }
            // Reject batches whose screen-space bounds contain no pixel centers.
            if corners.iter().all(|p| p.w > 0.0)
                && (0..2).any(|axis| {
                    let (min, max) = corners
                        .iter()
                        .map(|p| (p[axis] / p.w + 1.0) * 0.5 * resolution as f32)
                        .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), x| {
                            (min.min(x), max.max(x))
                        });
                    (min - 0.5).ceil() > (max - 0.5).floor()
                })
            {
                continue;
            }
            let mut projected = ClipPrimitive {
                source: batch.primitive.source,
                mirrored: batch.primitive.mirrored,
                vertices: Vec::with_capacity(batch.triangles.len() * 3),
                triangles: Vec::with_capacity(batch.triangles.len()),
            };
            for triangle in batch.triangles {
                let base = projected.vertices.len() as u32;
                projected.vertices.extend(triangle.indices.map(|i| {
                    let mut v = batch.primitive.vertices[i as usize];
                    v.position = view_projection * v.world_position.extend(1.0);
                    v
                }));
                projected.triangles.push(Triangle {
                    indices: [base, base + 1, base + 2],
                    source_index: triangle.source_index,
                });
            }
            let clipped = clip_primitive(&projected)?;
            let screen = project_primitive(&clipped, resolution, resolution)?;
            draw(
                std::slice::from_ref(&screen),
                materials,
                &mut depth,
                Pass::Capture {
                    color: &mut color,
                    lightmap,
                },
            );
        }
        Ok(color)
    }
}
