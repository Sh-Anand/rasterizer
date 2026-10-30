use std::f32::consts::PI;

use glam::{IVec2, Mat4, Vec2, Vec3, camera::rh};

use crate::config::{LightConfig, ShadowConfig};

/// The configured directional light and, with shadows enabled, its orthographic shadow map.
#[derive(Clone, Copy)]
pub struct Light {
    /// World-space direction from the surface toward the light.
    pub(crate) direction: Vec3,
    pub(crate) irradiance: Vec3,
    pub(crate) shadow: Option<ShadowMap>,
}

impl Light {
    /// The shadow map, if enabled, is fitted around `positions`, every scene vertex in world
    /// space.
    pub fn new(light: &LightConfig, shadow: &ShadowConfig, positions: &[Vec3]) -> Self {
        let direction = Vec3::from_array(light.direction).normalize();
        Self {
            direction,
            // Intensity 1 gives unit response for a facing white Lambertian surface.
            irradiance: Vec3::from_array(light.color) * light.intensity * PI,
            shadow: shadow.enabled.then(|| ShadowMap {
                view_projection: light_matrix(positions, direction),
                resolution: shadow.resolution,
                bias: shadow.bias,
            }),
        }
    }

    // CUDA shading must inline depth reads to address the shadow map as global memory.
    #[inline(always)]
    pub(crate) fn visibility(&self, depths: &[f32], world_position: Vec3) -> f32 {
        self.shadow
            .as_ref()
            .map_or(1.0, |map| map.visibility(depths, world_position))
    }
}

#[derive(Clone, Copy)]
pub struct ShadowMap {
    pub view_projection: Mat4,
    /// Width and height in texels.
    pub resolution: u32,
    bias: f32,
}

impl ShadowMap {
    pub(crate) fn len(&self) -> usize {
        self.resolution as usize * self.resolution as usize
    }

    fn visibility(&self, depths: &[f32], world_position: Vec3) -> f32 {
        self.project(world_position).map_or(1.0, |projected| {
            pcf(projected.truncate(), |texel| {
                self.compare(texel, projected.z, depths)
            })
        })
    }

    fn project(&self, world_position: Vec3) -> Option<Vec3> {
        let clip = self.view_projection * world_position.extend(1.0);
        if !clip.is_finite() || clip.w <= 0.0 {
            return None;
        }
        let ndc = clip.truncate() / clip.w;
        if !ndc.is_finite() || ndc.z.abs() > 1.0 || ndc.truncate().abs().max_element() > 1.0 {
            return None;
        }
        let uv = Vec2::new(ndc.x + 1.0, 1.0 - ndc.y) * 0.5;
        let pixel = uv * self.resolution as f32 - 0.5;
        Some(pixel.extend((clip.z / clip.w + 1.0) * 0.5))
    }

    fn compare(&self, texel: IVec2, depth: f32, depths: &[f32]) -> f32 {
        let size = self.resolution as i32;
        let x = texel.x.clamp(0, size - 1) as usize;
        let y = texel.y.clamp(0, size - 1) as usize;
        f32::from(depth <= depths[y * self.resolution as usize + x] + self.bias)
    }
}

fn pcf(pixel: Vec2, sample: impl Fn(IVec2) -> f32) -> f32 {
    let base = pixel.floor();
    let weight = pixel - base;
    let base = base.as_ivec2();
    let top = sample(base) * (1.0 - weight.x) + sample(base + IVec2::X) * weight.x;
    let bottom = sample(base + IVec2::Y) * (1.0 - weight.x) + sample(base + IVec2::ONE) * weight.x;
    top * (1.0 - weight.y) + bottom * weight.y
}

fn light_matrix(positions: &[Vec3], direction: Vec3) -> Mat4 {
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for &position in positions {
        min = min.min(position);
        max = max.max(position);
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
    for &position in positions {
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
    projection * view
}
