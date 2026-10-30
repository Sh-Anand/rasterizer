use std::{collections::HashMap, fs, io, path::Path};

use glam::{Mat4, Vec3, camera::rh};

use serde::Deserialize;

use crate::texture::Sampling;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub width: u32,
    pub height: u32,
    pub supersampling: u32,
    pub sampling: Sampling,
    pub neural: Neural,
    pub bounce: bool,
    pub camera: CameraConfig,
    scene_cameras: HashMap<String, CameraConfig>,
    pub light: LightConfig,
    pub shadow: ShadowConfig,
    pub output: OutputConfig,
}

/// What neural networks replace in every material.
#[derive(Clone, Copy, PartialEq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Neural {
    None,
    /// Base color, normal and metallic-roughness textures, by the scene's RTXNTC texture sets.
    Textures,
    /// Textures and BRDF, by neural materials with stand-in weights.
    Materials,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputConfig {
    pub exposure: f32,
    pub tone_mapping: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShadowConfig {
    pub enabled: bool,
    pub resolution: u32,
    pub bias: f32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LightConfig {
    pub direction: [f32; 3],
    pub color: [f32; 3],
    pub intensity: f32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CameraConfig {
    pub position: [f32; 3],
    pub target: [f32; 3],
    pub up: [f32; 3],
    pub fov_y_degrees: f32,
    pub near: f32,
    pub far: f32,
}

impl Config {
    /// Loads the configuration with `scene`'s camera, if it has one.
    pub fn load(path: impl AsRef<Path>, scene: &str) -> io::Result<Self> {
        let mut config: Self = toml::from_str(&fs::read_to_string(path)?)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        if let Some(camera) = config.scene_cameras.remove(scene) {
            config.camera = camera;
        }
        Ok(config)
    }

    pub fn camera_matrix(&self) -> Mat4 {
        let camera = &self.camera;
        let position = Vec3::from_array(camera.position);
        let forward = (Vec3::from_array(camera.target) - position).normalize();
        let view =
            rh::view::look_to_mat4(position, forward, Vec3::from_array(camera.up).normalize());
        // Camera looks down -Z; clipping bounds are -w..w on all three axes.
        let projection = rh::proj::opengl::perspective(
            camera.fov_y_degrees.to_radians(),
            self.width as f32 / self.height as f32,
            camera.near,
            camera.far,
        );
        projection * view
    }

    /// The supersampled render resolution.
    pub fn render_size(&self) -> (u32, u32) {
        (
            self.width * self.supersampling,
            self.height * self.supersampling,
        )
    }
}
