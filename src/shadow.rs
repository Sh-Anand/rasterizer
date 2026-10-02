use std::io;

use glam::{Mat4, Vec3, camera::rh};

use crate::{
    config::ShadowConfig, framebuffer::DepthBuffer, lighting::DirectionalLight, vertex::ClipScene,
};

pub struct ShadowMap {
    pub view_projection: Mat4,
    pub depth: DepthBuffer,
    bias: f32,
}

impl ShadowMap {
    /// Fit to the unclipped scene, including off-camera shadow casters.
    pub fn new(
        scene: &ClipScene,
        light: &DirectionalLight,
        config: &ShadowConfig,
    ) -> io::Result<Self> {
        if !(0.0..=1.0).contains(&config.bias) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Shadow bias must be finite and in 0..1",
            ));
        }
        Ok(Self {
            view_projection: light_matrix(scene, light.direction())?,
            depth: DepthBuffer::new(config.resolution, config.resolution)?,
            bias: config.bias,
        })
    }

    pub fn is_shadowed(&self, world_position: Vec3) -> bool {
        let clip = self.view_projection * world_position.extend(1.0);
        if !clip.is_finite() || clip.w <= 0.0 {
            return false;
        }
        let ndc = clip.truncate() / clip.w;
        if !ndc.is_finite() || ndc.abs().max_element() > 1.0 {
            return false;
        }
        let x = ((ndc.x + 1.0) * 0.5 * self.depth.width as f32) as u32;
        let y = ((1.0 - ndc.y) * 0.5 * self.depth.height as f32) as u32;
        let index = y.min(self.depth.height - 1) as usize * self.depth.width as usize
            + x.min(self.depth.width - 1) as usize;
        let depth = (ndc.z + 1.0) * 0.5;
        depth > self.depth.values[index] + self.bias
    }
}

fn light_matrix(scene: &ClipScene, direction: Vec3) -> io::Result<Mat4> {
    let positions = || {
        scene
            .primitives
            .iter()
            .flat_map(|p| p.vertices.iter().map(|v| v.world_position))
    };
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for position in positions() {
        if !position.is_finite() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Non-finite shadow caster position",
            ));
        }
        min = min.min(position);
        max = max.max(position);
    }
    if min.x == f32::INFINITY {
        return Ok(Mat4::IDENTITY);
    }
    let center = min * 0.5 + max * 0.5;
    let radius = (max - min).length() * 0.5;
    let padding = (radius * 0.02).max(1e-4);
    let eye = center + direction * (radius + 2.0 * padding);
    let up = if direction.y.abs() > 0.99 {
        Vec3::X
    } else {
        Vec3::Y
    };
    let view = rh::view::look_to_mat4(eye, -direction, up);
    min = Vec3::splat(f32::INFINITY);
    max = Vec3::splat(f32::NEG_INFINITY);
    for position in positions() {
        let position = view.transform_point3(position);
        min = min.min(position);
        max = max.max(position);
    }
    let projection = rh::proj::opengl::orthographic(
        min.x - padding,
        max.x + padding,
        min.y - padding,
        max.y + padding,
        (-max.z - padding).max(0.0),
        -min.z + padding,
    );
    let matrix = projection * view;
    if !matrix.is_finite() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Scene bounds produce a non-finite shadow matrix",
        ));
    }
    Ok(matrix)
}
