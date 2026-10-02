use std::io;

use glam::{Vec3, Vec4};

use crate::{
    geometry::{PrimitiveSource, Triangle},
    vertex::ClipScene,
};

#[derive(Debug, Clone, Copy)]
pub struct ScreenVertex {
    pub position: Vec3,
    pub inv_w: f32,
}

#[derive(Debug)]
pub struct ScreenPrimitive {
    pub source: PrimitiveSource,
    pub vertices: Vec<ScreenVertex>,
    pub triangles: Vec<Triangle>,
}

#[derive(Debug)]
pub struct ScreenScene {
    pub scene_index: usize,
    pub primitives: Vec<ScreenPrimitive>,
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
                .positions
                .iter()
                .map(|&position| project_vertex(position, width as f32, height as f32))
                .collect::<io::Result<_>>()?;
            Ok(ScreenPrimitive {
                source: primitive.source,
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

fn project_vertex(clip: Vec4, width: f32, height: f32) -> io::Result<ScreenVertex> {
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
    Ok(ScreenVertex { position, inv_w })
}
