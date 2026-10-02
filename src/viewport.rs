use std::io;

use glam::{DVec2, DVec3, Vec2, Vec3};

use crate::{
    geometry::{PrimitiveSource, Triangle},
    vertex::{ClipScene, ClipVertex},
};

#[derive(Debug, Clone, Copy)]
pub struct ScreenVertex {
    pub position: Vec3,
    pub world_position: Vec3,
    pub inv_w: f32,
    pub uv: Vec2,
    pub emissive_uv: Vec2,
    /// World-space normal.
    pub normal: Vec3,
}

#[derive(Debug)]
pub struct ScreenPrimitive {
    pub source: PrimitiveSource,
    pub mirrored: bool,
    pub vertices: Vec<ScreenVertex>,
    pub triangles: Vec<Triangle>,
}

#[derive(Debug)]
pub struct ScreenScene {
    pub scene_index: usize,
    pub primitives: Vec<ScreenPrimitive>,
}

pub struct FragmentAttributes {
    pub world_position: Vec3,
    pub uv: Vec2,
    pub emissive_uv: Vec2,
    pub normal: Vec3,
}

pub fn project_scene(scene: &ClipScene, width: u32, height: u32) -> io::Result<ScreenScene> {
    if width == 0 || height == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Image width and height must be nonzero",
        ));
    }
    let primitives = scene
        .primitives
        .iter()
        .map(|primitive| {
            let vertices = primitive
                .vertices
                .iter()
                .map(|&vertex| project_vertex(vertex, width as f32, height as f32))
                .collect::<io::Result<_>>()?;
            Ok(ScreenPrimitive {
                source: primitive.source,
                mirrored: primitive.mirrored,
                vertices,
                triangles: primitive.triangles.clone(),
            })
        })
        .collect::<io::Result<_>>()?;
    Ok(ScreenScene {
        scene_index: scene.scene_index,
        primitives,
    })
}

fn project_vertex(vertex: ClipVertex, width: f32, height: f32) -> io::Result<ScreenVertex> {
    let clip = vertex.position;
    if !clip.is_finite() || clip.w <= 0.0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Perspective division requires a finite vertex with positive w",
        ));
    }
    let inv_w = clip.w.recip();
    let ndc = clip.truncate() / clip.w;
    let position = Vec3::new(
        (ndc.x + 1.0) * 0.5 * width,
        (1.0 - ndc.y) * 0.5 * height,
        (ndc.z + 1.0) * 0.5,
    );
    if !position.is_finite() || !inv_w.is_finite() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Perspective division produced a non-finite result",
        ));
    }
    Ok(ScreenVertex {
        position,
        world_position: vertex.world_position,
        inv_w,
        uv: vertex.uv,
        emissive_uv: vertex.emissive_uv,
        normal: vertex.normal,
    })
}

pub fn interpolate_attributes(
    vertices: [ScreenVertex; 3],
    barycentric: [f32; 3],
) -> FragmentAttributes {
    let weights: [f64; 3] =
        std::array::from_fn(|i| f64::from(barycentric[i]) * f64::from(vertices[i].inv_w));
    let world_position: DVec3 = vertices
        .iter()
        .zip(weights)
        .map(|(vertex, weight)| vertex.world_position.as_dvec3() * weight)
        .sum();
    let uv: DVec2 = vertices
        .iter()
        .zip(weights)
        .map(|(vertex, weight)| vertex.uv.as_dvec2() * weight)
        .sum();
    let emissive_uv: DVec2 = vertices
        .iter()
        .zip(weights)
        .map(|(vertex, weight)| vertex.emissive_uv.as_dvec2() * weight)
        .sum();
    let normal: DVec3 = vertices
        .iter()
        .zip(weights)
        .map(|(vertex, weight)| vertex.normal.as_dvec3() * weight)
        .sum();
    let sum = weights.iter().sum::<f64>();
    FragmentAttributes {
        world_position: (world_position / sum).as_vec3(),
        uv: (uv / sum).as_vec2(),
        emissive_uv: (emissive_uv / sum).as_vec2(),
        normal: (normal / sum).normalize_or_zero().as_vec3(),
    }
}
