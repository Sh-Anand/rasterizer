use std::{collections::HashMap, fs, io, path::Path};

use glam::{Mat4, Vec3, camera::rh};

use serde::Deserialize;

use crate::texture::Sampling;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub width: u32,
    pub height: u32,
    #[serde(default = "default_supersampling")]
    pub supersampling: u32,
    #[serde(default)]
    pub sampling: Sampling,
    #[serde(default)]
    pub bounce: bool,
    pub camera: CameraConfig,
    #[serde(default)]
    scene_cameras: HashMap<String, CameraConfig>,
    pub light: LightConfig,
    pub shadow: ShadowConfig,
    #[serde(default)]
    pub output: OutputConfig,
}

#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OutputConfig {
    pub exposure: f32,
    pub tone_mapping: bool,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            exposure: 1.0,
            tone_mapping: true,
        }
    }
}

impl OutputConfig {
    pub(crate) fn validate(&self) -> io::Result<()> {
        if !self.exposure.is_finite() || self.exposure < 0.0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Output exposure must be finite and nonnegative",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShadowConfig {
    pub enabled: bool,
    pub resolution: u32,
    pub bias: f32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LightConfig {
    pub direction: [f32; 3],
    pub color: [f32; 3],
    pub intensity: f32,
}

#[derive(Debug, Deserialize)]
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
    pub fn camera_matrix(&self) -> io::Result<Mat4> {
        let camera = &self.camera;
        if self.width == 0 || self.height == 0 {
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
            self.width as f32 / self.height as f32,
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

    pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
        let text = fs::read_to_string(path)?;
        let config: Self = toml::from_str(&text)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        config.output.validate()?;
        config.render_size()?;
        Ok(config)
    }

    pub fn render_size(&self) -> io::Result<(u32, u32)> {
        match (
            self.width.checked_mul(self.supersampling),
            self.height.checked_mul(self.supersampling),
        ) {
            (Some(width), Some(height)) if width > 0 && height > 0 => Ok((width, height)),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Supersampled image dimensions must be nonzero and fit in 32 bits",
            )),
        }
    }

    pub fn load_scene(path: impl AsRef<Path>, scene: &str) -> io::Result<Self> {
        let mut config = Self::load(path)?;
        if let Some(camera) = config.scene_cameras.remove(scene) {
            config.camera = camera;
        }
        Ok(config)
    }
}

fn default_supersampling() -> u32 {
    1
}
