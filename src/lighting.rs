use std::io;

use glam::Vec3;

pub struct DirectionalLight {
    direction: Vec3,
    radiance: Vec3,
}

impl DirectionalLight {
    /// World-space direction from the surface toward the light.
    pub fn new(direction: [f32; 3], color: [f32; 3], intensity: f32) -> io::Result<Self> {
        let direction = Vec3::from_array(direction).try_normalize().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "Light direction must be finite and nonzero",
            )
        })?;
        let color = Vec3::from_array(color);
        let radiance = color * intensity;
        if !color.is_finite()
            || color.min_element() < 0.0
            || !intensity.is_finite()
            || intensity < 0.0
            || !radiance.is_finite()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Light color and intensity must be finite, nonnegative, and not overflow",
            ));
        }
        Ok(Self {
            direction,
            radiance,
        })
    }

    pub fn shade(&self, base_color: Vec3, normal: Vec3) -> Vec3 {
        base_color * self.radiance * normal.dot(self.direction).max(0.0)
    }
}
