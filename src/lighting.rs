use std::{f32::consts::PI, io};

use glam::Vec3;

use crate::{
    config::LightConfig, geometry::ClipScene, material::Material, shadow::LightShadow,
    texture::UvSample,
};

#[derive(Clone, Copy)]
pub struct RenderLight<S = LightShadow> {
    pub light: Light,
    pub shadow: Option<S>,
}

pub struct Lighting<'a> {
    pub lights: &'a [RenderLight],
    pub camera_position: Vec3,
}

pub(crate) fn visible_lights<S>(
    lights: &[RenderLight<S>],
    position: Vec3,
    visibility: impl Fn(&S, Vec3) -> f32,
) -> impl Iterator<Item = LightSample> {
    lights.iter().filter_map(move |light| {
        let mut sample = light.light.sample(position)?;
        let visibility = light
            .shadow
            .as_ref()
            .map_or(1.0, |shadow| visibility(shadow, position));
        sample.irradiance *= visibility;
        (visibility > 0.0).then_some(sample)
    })
}

#[derive(Clone, Copy)]
pub enum Light {
    Directional(DirectionalLight),
    Area(AreaLight),
}

impl Light {
    pub fn sample(self, position: Vec3) -> Option<LightSample> {
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

#[derive(Clone, Copy)]
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
        // Intensity 1 gives unit response for a facing white Lambertian surface.
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

#[derive(Clone, Copy)]
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
        let [a, b, c] = vertices;
        let cross = (b - a).cross(c - a);
        let position = (a + b + c) / 3.0;
        let area = cross.length() * 0.5;
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
            normal: cross.normalize_or_zero(),
            area,
            radiance,
            double_sided,
        })
    }

    fn sample(self, position: Vec3) -> Option<LightSample> {
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
            let vertices = triangle.map(|i| primitive.vertices[i as usize]);
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
