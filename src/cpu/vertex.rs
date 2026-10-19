use std::io;

use glam::Mat4;
use rayon::prelude::*;

use crate::{
    geometry::{
        ClipPrimitive, ClipScene, PreparedPrimitive, PreparedScene, VertexTransform,
        check_view_projection,
    },
    pipeline::VertexStage,
};

pub struct CpuVertex;

impl VertexStage for CpuVertex {
    fn transform(&mut self, scene: &PreparedScene, view_projection: Mat4) -> io::Result<ClipScene> {
        check_view_projection(view_projection)?;
        Ok(ClipScene {
            scene_index: scene.scene_index,
            primitives: scene
                .primitives
                .par_iter()
                .map(|primitive| transform_primitive(primitive, view_projection))
                .collect::<io::Result<_>>()?,
        })
    }
}

fn transform_primitive(
    primitive: &PreparedPrimitive,
    view_projection: Mat4,
) -> io::Result<ClipPrimitive> {
    let transform = VertexTransform::new(primitive, view_projection);
    let vertices = primitive
        .vertices
        .par_iter()
        .map(|vertex| transform.apply(vertex).checked(primitive))
        .collect::<io::Result<_>>()?;
    Ok(ClipPrimitive {
        source: primitive.source,
        mirrored: primitive.mirrored,
        vertices,
        triangles: primitive.triangles.clone(),
    })
}
