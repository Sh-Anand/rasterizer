use std::f32::consts::PI;

use glam::Vec3;

pub fn diffuse_color(base_color: Vec3, metallic: f32) -> Vec3 {
    base_color * (1.0 - metallic)
}

pub struct Brdf {
    pub diffuse: Vec3,
    f0: Vec3,
    alpha_squared: f32,
}

impl Brdf {
    pub fn new(base_color: Vec3, metallic: f32, roughness: f32) -> Self {
        Self {
            diffuse: diffuse_color(base_color, metallic),
            f0: Vec3::splat(0.04).lerp(base_color, metallic),
            alpha_squared: roughness.clamp(0.045, 1.0).powi(4),
        }
    }

    pub fn specular(&self, normal: Vec3, view: Vec3, light: Vec3) -> Vec3 {
        let no_v = normal.dot(view).clamp(0.0, 1.0);
        let no_l = normal.dot(light).clamp(0.0, 1.0);
        if no_v == 0.0 || no_l == 0.0 {
            return Vec3::ZERO;
        }
        let Some(half) = (view + light).try_normalize() else {
            return Vec3::ZERO;
        };
        let no_h = normal.dot(half).clamp(0.0, 1.0);
        let vo_h = view.dot(half).clamp(0.0, 1.0);
        let a2 = self.alpha_squared;
        // Cross product avoids cancellation near the GGX peak.
        let denominator = normal.cross(half).length_squared() + a2 * no_h * no_h;
        let distribution = a2 / (PI * denominator * denominator);
        let visibility = 0.5
            / (no_l * (no_v * no_v * (1.0 - a2) + a2).sqrt()
                + no_v * (no_l * no_l * (1.0 - a2) + a2).sqrt());
        let fresnel = self.f0 + (Vec3::ONE - self.f0) * (1.0 - vo_h).powi(5);
        fresnel * (distribution * visibility)
    }
}
