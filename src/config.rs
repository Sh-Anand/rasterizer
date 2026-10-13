use std::{collections::HashMap, fs, io, path::Path};

use serde::{Deserialize, Serialize};

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
    pub camera: CameraConfig,
    #[serde(default)]
    scene_cameras: HashMap<String, CameraConfig>,
    pub light: LightConfig,
    pub shadow: ShadowConfig,
    #[serde(default)]
    pub bake: BakeConfig,
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

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ShadowConfig {
    pub enabled: bool,
    pub resolution: u32,
    pub bias: f32,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LightConfig {
    pub direction: [f32; 3],
    pub color: [f32; 3],
    pub intensity: f32,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct BakeConfig {
    pub resolution: u32,
    pub hemicube_resolution: u32,
    pub bias: f32,
}

impl Default for BakeConfig {
    fn default() -> Self {
        Self {
            resolution: 32,
            hemicube_resolution: 16,
            bias: 0.00001,
        }
    }
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
