use std::io;

use glam::{Mat3, Mat4, Vec4};

use crate::{
    geometry::{PreparedPrimitive, PreparedScene},
    pipeline::VertexStage,
    vertex::{ClipPrimitive, ClipScene, ClipVertex},
};

pub struct CpuVertex;

impl VertexStage for CpuVertex {
    fn transform(&mut self, scene: &PreparedScene, view_projection: Mat4) -> io::Result<ClipScene> {
        transform_scene(scene, view_projection)
    }
}

pub fn transform_scene(scene: &PreparedScene, view_projection: Mat4) -> io::Result<ClipScene> {
    if !view_projection.is_finite() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Non-finite projection matrix",
        ));
    }
    Ok(ClipScene {
        scene_index: scene.scene_index,
        primitives: scene
            .primitives
            .iter()
            .map(|primitive| transform_primitive(primitive, view_projection))
            .collect::<io::Result<_>>()?,
    })
}

fn transform_primitive(
    primitive: &PreparedPrimitive,
    view_projection: Mat4,
) -> io::Result<ClipPrimitive> {
    let invalid = |message: &str| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Node {}, mesh {}, primitive {}: {message}",
                primitive.source.node_index,
                primitive.source.mesh_index,
                primitive.source.primitive_index,
            ),
        )
    };
    let mvp = view_projection * primitive.model;
    let linear = Mat3::from_mat4(primitive.model);
    let sign = linear.determinant().signum();
    let vertices = primitive
        .vertices
        .iter()
        .map(|vertex| {
            let position = mvp * vertex.position.extend(1.0);
            let world_position = primitive.model.transform_point3(vertex.position);
            if !position.is_finite() || !world_position.is_finite() {
                return Err(invalid("Non-finite vertex position"));
            }
            let normal = primitive.normal_matrix * vertex.normal;
            let normal = if primitive.flat_shaded {
                normal.normalize_or_zero()
            } else {
                normal
                    .try_normalize()
                    .ok_or_else(|| invalid("Invalid vertex normal"))?
            };
            let tangent = (linear * vertex.tangent.truncate())
                .try_normalize()
                .filter(|_| vertex.tangent.w.abs() == 1.0)
                .map_or(Vec4::ZERO, |t| t.extend(vertex.tangent.w * sign));
            Ok(ClipVertex {
                position,
                world_position,
                uv: vertex.uv,
                normal_uv: vertex.normal_uv,
                metallic_roughness_uv: vertex.metallic_roughness_uv,
                occlusion_uv: vertex.occlusion_uv,
                emissive_uv: vertex.emissive_uv,
                lightmap_uv: vertex.lightmap_uv,
                normal,
                tangent,
            })
        })
        .collect::<io::Result<_>>()?;
    Ok(ClipPrimitive {
        source: primitive.source,
        mirrored: primitive.mirrored,
        vertices,
        triangles: primitive.triangles.clone(),
    })
}

pub fn reproject_scene(scene: &ClipScene, view_projection: Mat4) -> io::Result<ClipScene> {
    if !view_projection.is_finite() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Non-finite projection matrix",
        ));
    }
    let mut scene = scene.clone();
    for primitive in &mut scene.primitives {
        for vertex in &mut primitive.vertices {
            vertex.position = view_projection * vertex.world_position.extend(1.0);
            if !vertex.position.is_finite() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Non-finite reprojected position",
                ));
            }
        }
    }
    Ok(scene)
}
