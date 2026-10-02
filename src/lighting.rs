use std::io;

use glam::Vec3;

pub struct DirectionalLight {
    direction: Vec3,
}

impl DirectionalLight {
    /// World-space direction from the surface toward the light.
    pub fn new(direction: [f32; 3]) -> io::Result<Self> {
        let direction = Vec3::from_array(direction).try_normalize().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "Light direction must be finite and nonzero",
            )
        })?;
        Ok(Self { direction })
    }

    pub fn shade(&self, base_color: Vec3, normal: Vec3) -> Vec3 {
        base_color * normal.dot(self.direction).max(0.0)
    }
}
