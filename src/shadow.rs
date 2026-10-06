use std::io;

use glam::{Mat3, Mat4, Vec3, camera::rh};

use crate::{
    config::ShadowConfig,
    framebuffer::DepthBuffer,
    lighting::{AreaLight, Light},
    vertex::ClipScene,
};

pub enum LightShadow {
    Directional(ShadowMap),
    Area {
        position: Vec3,
        basis: Mat3,
        faces: Vec<ShadowMap>,
    },
}

impl LightShadow {
    pub fn new(scene: &ClipScene, light: &Light, config: &ShadowConfig) -> io::Result<Self> {
        match light {
            Light::Directional(light) => Ok(Self::Directional(ShadowMap::new(
                light_matrix(scene, light.direction())?,
                ShadowProjection::Orthographic,
                config,
            )?)),
            Light::Area(light) => Self::area(scene, light, config),
        }
    }

    fn area(scene: &ClipScene, light: &AreaLight, config: &ShadowConfig) -> io::Result<Self> {
        let position = light.position();
        let normal = light.normal();
        let up = if normal.y.abs() > 0.99 {
            Vec3::X
        } else {
            Vec3::Y
        };
        let right = up.cross(normal).try_normalize().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "Area shadow needs a nondegenerate emitter",
            )
        })?;
        let up = normal.cross(right);
        let basis = Mat3::from_cols(right, up, normal);
        let mut radius = 0.0_f64;
        for vertex in scene.primitives.iter().flat_map(|p| &p.vertices) {
            let distance = (vertex.world_position.as_dvec3() - position.as_dvec3()).length();
            if !distance.is_finite() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Non-finite shadow caster position",
                ));
            }
            radius = radius.max(distance);
        }
        let far = (radius * 1.01).max(1e-3) as f32;
        let near = far * 1e-4;
        let projection = rh::proj::opengl::perspective(std::f32::consts::FRAC_PI_2, 1.0, near, far);
        // +X, -X, +Y, -Y, +Z, -Z in the emitter's local frame.
        let directions = [right, -right, up, -up, normal, -normal];
        let ups = [up, up, -normal, normal, up, up];
        let count = if light.double_sided() { 6 } else { 5 };
        let faces = directions
            .into_iter()
            .zip(ups)
            .take(count)
            .map(|(direction, up)| {
                let view = rh::view::look_to_mat4(position, direction, up);
                ShadowMap::new(
                    projection * view,
                    ShadowProjection::Perspective { far },
                    config,
                )
            })
            .collect::<io::Result<Vec<_>>>()?;
        Ok(Self::Area {
            position,
            basis,
            faces,
        })
    }

    pub fn maps(&self) -> &[ShadowMap] {
        match self {
            Self::Directional(map) => std::slice::from_ref(map),
            Self::Area { faces, .. } => faces,
        }
    }

    pub fn maps_mut(&mut self) -> &mut [ShadowMap] {
        match self {
            Self::Directional(map) => std::slice::from_mut(map),
            Self::Area { faces, .. } => faces,
        }
    }

    pub fn is_shadowed(&self, world_position: Vec3) -> bool {
        match self {
            Self::Directional(map) => map.is_shadowed(world_position),
            Self::Area {
                position,
                basis,
                faces,
            } => {
                let direction = basis.transpose() * (world_position - position);
                if !direction.is_finite() || (faces.len() == 5 && direction.z <= 0.0) {
                    return false;
                }
                let magnitude = direction.abs();
                let axis = if magnitude.x >= magnitude.y && magnitude.x >= magnitude.z {
                    0
                } else if magnitude.y >= magnitude.z {
                    1
                } else {
                    2
                };
                let face = 2 * axis + usize::from(direction[axis] < 0.0);
                faces[face].is_shadowed(world_position)
            }
        }
    }
}

#[derive(Clone, Copy)]
pub enum ShadowProjection {
    Orthographic,
    Perspective { far: f32 },
}

pub struct ShadowMap {
    pub view_projection: Mat4,
    pub depth: DepthBuffer,
    pub projection: ShadowProjection,
    bias: f32,
}

impl ShadowMap {
    fn new(
        view_projection: Mat4,
        projection: ShadowProjection,
        config: &ShadowConfig,
    ) -> io::Result<Self> {
        if !(0.0..=1.0).contains(&config.bias) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Shadow bias must be finite and in 0..1",
            ));
        }
        if !view_projection.is_finite() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Non-finite shadow matrix",
            ));
        }
        Ok(Self {
            view_projection,
            depth: DepthBuffer::new(config.resolution, config.resolution)?,
            projection,
            bias: config.bias,
        })
    }

    pub fn is_shadowed(&self, world_position: Vec3) -> bool {
        let clip = self.view_projection * world_position.extend(1.0);
        if !clip.is_finite() || clip.w <= 0.0 {
            return false;
        }
        let ndc = clip.truncate() / clip.w;
        // Adjacent cube faces can disagree by a few ulps at their shared edge.
        let xy_limit = match self.projection {
            ShadowProjection::Orthographic => 1.0,
            ShadowProjection::Perspective { .. } => 1.0 + 1e-5,
        };
        if !ndc.is_finite() || ndc.z.abs() > 1.0 || ndc.truncate().abs().max_element() > xy_limit {
            return false;
        }
        let x = ((ndc.x + 1.0) * 0.5 * self.depth.width as f32) as u32;
        let y = ((1.0 - ndc.y) * 0.5 * self.depth.height as f32) as u32;
        let index = y.min(self.depth.height - 1) as usize * self.depth.width as usize
            + x.min(self.depth.width - 1) as usize;
        let depth = match self.projection {
            ShadowProjection::Orthographic => (ndc.z + 1.0) * 0.5,
            ShadowProjection::Perspective { far } => clip.w / far,
        };
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
    Ok(projection * view)
}
