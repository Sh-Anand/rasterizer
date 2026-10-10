use std::{f32::consts::PI, io};

use glam::Vec3;

use crate::{config::LightConfig, material::Material, texture::UvSample, vertex::ClipScene};

pub enum Light {
    Directional(DirectionalLight),
    Area(AreaLight),
}

impl Light {
    pub fn sample(&self, position: Vec3) -> Option<LightSample> {
        match self {
            Self::Directional(light) => Some(LightSample {
                direction: light.direction,
                irradiance: light.irradiance,
            }),
            Self::Area(light) => light.sample(position),
        }
    }
}

pub struct LightSample {
    pub direction: Vec3,
    pub irradiance: Vec3,
}

impl LightSample {
    pub fn diffuse(&self, normal: Vec3) -> Vec3 {
        self.irradiance * (normal.dot(self.direction).max(0.0) / PI)
    }
}

pub struct DirectionalLight {
    direction: Vec3,
    irradiance: Vec3,
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
        // Preserve intensity as the response of a facing white Lambertian surface.
        let irradiance = color * intensity * PI;
        if !color.is_finite()
            || color.min_element() < 0.0
            || !intensity.is_finite()
            || intensity < 0.0
            || !irradiance.is_finite()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Light color and intensity must be finite, nonnegative, and not overflow",
            ));
        }
        Ok(Self {
            direction,
            irradiance,
        })
    }

    pub fn direction(&self) -> Vec3 {
        self.direction
    }
}

pub struct AreaLight {
    position: Vec3,
    normal: Vec3,
    area: f32,
    radiance: Vec3,
    double_sided: bool,
}

impl AreaLight {
    pub fn position(&self) -> Vec3 {
        self.position
    }

    pub fn normal(&self) -> Vec3 {
        self.normal
    }

    pub fn double_sided(&self) -> bool {
        self.double_sided
    }

    pub fn new(vertices: [Vec3; 3], radiance: Vec3, double_sided: bool) -> io::Result<Self> {
        let [a, b, c] = vertices.map(|v| v.as_dvec3());
        let cross = (b - a).cross(c - a);
        let position = ((a + b + c) / 3.0).as_vec3();
        let area = (cross.length() * 0.5) as f32;
        if !position.is_finite()
            || !area.is_finite()
            || !radiance.is_finite()
            || radiance.min_element() < 0.0
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Area light geometry and emission must be finite, with nonnegative emission",
            ));
        }
        Ok(Self {
            position,
            normal: cross.normalize_or_zero().as_vec3(),
            area,
            radiance,
            double_sided,
        })
    }

    fn sample(&self, position: Vec3) -> Option<LightSample> {
        let offset = self.position - position;
        let distance_squared = offset.length_squared();
        if distance_squared == 0.0 || !distance_squared.is_finite() {
            return None;
        }
        let direction = offset / distance_squared.sqrt();
        let emitter_cosine = self.normal.dot(-direction);
        let emitter_cosine = if self.double_sided {
            emitter_cosine.abs()
        } else {
            emitter_cosine.max(0.0)
        };
        let weight = self.area * emitter_cosine / distance_squared;
        Some(LightSample {
            direction,
            irradiance: self.radiance * weight,
        })
    }
}

pub fn collect_lights(
    scene: &ClipScene,
    materials: &[Material],
    config: &LightConfig,
) -> io::Result<Vec<Light>> {
    let mut lights = vec![Light::Directional(DirectionalLight::new(
        config.direction,
        config.color,
        config.intensity,
    )?)];
    for primitive in &scene.primitives {
        let Some(index) = primitive.source.material_index else {
            continue;
        };
        let material = &materials[index];
        if material.emissive_factor == Vec3::ZERO {
            continue;
        }
        for triangle in &primitive.triangles {
            let vertices = triangle.indices.map(|i| primitive.vertices[i as usize]);
            let uv = (vertices[0].uv + vertices[1].uv + vertices[2].uv) / 3.0;
            if material
                .alpha_cutoff
                .is_some_and(|cutoff| material.base_color(UvSample::point(uv)).w < cutoff)
            {
                continue;
            }
            let emissive_uv =
                (vertices[0].emissive_uv + vertices[1].emissive_uv + vertices[2].emissive_uv) / 3.0;
            let radiance = material.emission(UvSample::point(emissive_uv));
            if radiance == Vec3::ZERO {
                continue;
            }
            let mut positions = vertices.map(|v| v.world_position);
            if primitive.mirrored {
                positions.swap(1, 2);
            }
            let light = AreaLight::new(positions, radiance, material.double_sided)?;
            if light.area > 0.0 {
                lights.push(Light::Area(light));
            }
        }
    }
    Ok(lights)
}
