use std::io;

use glam::{Mat4, Vec2, Vec3, Vec4, camera::rh};

use crate::{
    config::Config,
    geometry::{PrimitiveSource, Triangle},
};

#[derive(Debug, Clone)]
pub struct ClipPrimitive {
    pub source: PrimitiveSource,
    pub mirrored: bool,
    pub vertices: Vec<ClipVertex>,
    pub triangles: Vec<Triangle>,
}

#[derive(Debug, Clone, Copy)]
pub struct ClipVertex {
    pub position: Vec4,
    pub world_position: Vec3,
    pub uv: Vec2,
    pub normal_uv: Vec2,
    pub metallic_roughness_uv: Vec2,
    pub occlusion_uv: Vec2,
    pub emissive_uv: Vec2,
    pub lightmap_uv: Vec2,
    /// World-space normal.
    pub normal: Vec3,
    /// World-space tangent; W is the bitangent sign (zero for an unusable frame).
    pub tangent: Vec4,
}

#[derive(Debug, Clone)]
pub struct ClipScene {
    pub scene_index: usize,
    pub primitives: Vec<ClipPrimitive>,
}

pub fn camera_matrix(config: &Config) -> io::Result<Mat4> {
    let camera = &config.camera;
    if config.width == 0 || config.height == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Image width and height must be nonzero",
        ));
    }
    if !camera.fov_y_degrees.is_finite()
        || camera.fov_y_degrees <= 0.0
        || camera.fov_y_degrees >= 180.0
        || !camera.near.is_finite()
        || !camera.far.is_finite()
        || camera.near <= 0.0
        || camera.far <= camera.near
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Camera needs 0 < fov_y_degrees < 180 and finite 0 < near < far",
        ));
    }

    let position = Vec3::from_array(camera.position);
    let target = Vec3::from_array(camera.target);
    let forward = (target - position).try_normalize();
    let up = Vec3::from_array(camera.up).try_normalize();
    let (Some(forward), Some(up)) = (forward, up) else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Camera needs a finite, nonzero viewing direction and up vector",
        ));
    };
    if !position.is_finite() || forward.cross(up).length_squared() < 1e-12 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Camera position must be finite and up must not parallel the viewing direction",
        ));
    }

    let view = rh::view::look_to_mat4(position, forward, up);
    // Camera looks down -Z; clipping bounds are -w..w on all three axes.
    let projection = rh::proj::opengl::perspective(
        camera.fov_y_degrees.to_radians(),
        config.width as f32 / config.height as f32,
        camera.near,
        camera.far,
    );
    let view_projection = projection * view;
    if !view_projection.is_finite() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Camera settings produce a non-finite matrix",
        ));
    }
    Ok(view_projection)
}
