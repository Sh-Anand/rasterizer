use std::io;

use glam::{IVec2, Mat3, Mat4, Vec2, Vec3, Vec4, camera::rh};

use crate::{
    config::ShadowConfig,
    lighting::{AreaLight, Light, RenderLight},
    pipeline::ShadowProjection,
};

/// Shadow maps of one light, as a range of the shared map list.
#[derive(Clone, Copy)]
pub enum LightShadow {
    Directional([usize; 2]),
    Area {
        position: Vec3,
        basis: Mat3,
        faces: [usize; 2],
    },
}

impl LightShadow {
    fn new(
        positions: &[Vec3],
        light: &Light,
        config: &ShadowConfig,
        maps: &mut Vec<ShadowMap>,
    ) -> io::Result<Self> {
        match light {
            Light::Directional(light) => {
                let start = maps.len();
                push_map(
                    maps,
                    light_matrix(positions, light.direction())?,
                    ShadowProjection::Orthographic,
                    config,
                )?;
                Ok(Self::Directional([start, maps.len()]))
            }
            Light::Area(light) => Self::area(positions, light, config, maps),
        }
    }

    fn area(
        positions: &[Vec3],
        light: &AreaLight,
        config: &ShadowConfig,
        maps: &mut Vec<ShadowMap>,
    ) -> io::Result<Self> {
        let position = light.position();
        let normal = light.normal();
        let up = if normal.y.abs() > 0.99 {
            Vec3::X
        } else {
            Vec3::Y
        };
        let right = up.cross(normal).try_normalize().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "Area shadow needs a nondegenerate emitter",
            )
        })?;
        let up = normal.cross(right);
        let basis = Mat3::from_cols(right, up, normal);
        let mut radius = 0.0_f32;
        for &vertex in positions {
            let distance = (vertex - position).length();
            if !distance.is_finite() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Non-finite shadow caster position",
                ));
            }
            radius = radius.max(distance);
        }
        let far = (radius * 1.01).max(1e-3);
        let near = far * 1e-4;
        let projection = rh::proj::opengl::perspective(std::f32::consts::FRAC_PI_2, 1.0, near, far);
        let count = if light.double_sided() { 6 } else { 5 };
        let start = maps.len();
        for face in 0..count {
            let axes = basis * cube_basis(face);
            let view = rh::view::look_to_mat4(position, axes.z_axis, axes.y_axis);
            push_map(
                maps,
                projection * view,
                ShadowProjection::Perspective { far },
                config,
            )?;
        }
        Ok(Self::Area {
            position,
            basis,
            faces: [start, maps.len()],
        })
    }

    pub fn visibility(&self, maps: &[ShadowMap], depths: &[f32], world_position: Vec3) -> f32 {
        let (Self::Directional([start, end])
        | Self::Area {
            faces: [start, end],
            ..
        }) = *self;
        let maps = &maps[start..end];
        match self {
            Self::Directional(_) => maps
                .first()
                .map_or(1.0, |map| map.visibility(depths, world_position)),
            Self::Area {
                position, basis, ..
            } => {
                let direction = basis.transpose() * (world_position - position);
                if !direction.is_finite() || (maps.len() == 5 && direction.z <= 0.0) {
                    return 1.0;
                }
                let face = cube_face(direction);
                let Some(map) = maps.get(face) else {
                    return 1.0;
                };
                let Some(projected) = map.project(world_position) else {
                    return 1.0;
                };
                pcf(projected.truncate(), |texel| {
                    let outside = texel.cmplt(IVec2::ZERO) | texel.cmpge(map.size().as_ivec2());
                    if !outside.any() {
                        return map.compare(texel, projected.z, depths);
                    }
                    let uv = (texel.as_vec2() + 0.5) / map.size();
                    let ray = cube_basis(face) * Vec3::new(2.0 * uv.x - 1.0, 1.0 - 2.0 * uv.y, 1.0);
                    if outside.all() {
                        // A corner tap touches three faces; blend their comparisons.
                        (0..3)
                            .map(|axis| {
                                let neighbor = 2 * axis + usize::from(ray[axis] < 0.0);
                                cube_compare(maps, depths, world_position, ray, neighbor)
                            })
                            .sum::<f32>()
                            / 3.0
                    } else {
                        cube_compare(maps, depths, world_position, ray, cube_face(ray))
                    }
                })
            }
        }
    }
}

#[derive(Clone, Copy)]
pub struct ShadowMap {
    pub view_projection: Mat4,
    pub width: u32,
    pub height: u32,
    /// First depth of this map in the shared depth atlas.
    pub offset: usize,
    pub projection: ShadowProjection,
    bias: f32,
}

fn push_map(
    maps: &mut Vec<ShadowMap>,
    view_projection: Mat4,
    projection: ShadowProjection,
    config: &ShadowConfig,
) -> io::Result<()> {
    if config.resolution == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Depth buffer dimensions must be nonzero",
        ));
    }
    if !(0.0..=1.0).contains(&config.bias) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Shadow bias must be finite and in 0..1",
        ));
    }
    if !view_projection.is_finite() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Non-finite shadow matrix",
        ));
    }
    let offset = maps.last().map_or(0, ShadowMap::end);
    let size = config.resolution as usize * config.resolution as usize;
    if offset.checked_add(size).is_none() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Shadow depth atlas size overflow",
        ));
    }
    maps.push(ShadowMap {
        view_projection,
        width: config.resolution,
        height: config.resolution,
        offset,
        projection,
        bias: config.bias,
    });
    Ok(())
}

impl ShadowMap {
    /// One past this map's last depth in the atlas.
    pub(crate) fn end(&self) -> usize {
        self.offset + self.width as usize * self.height as usize
    }

    fn visibility(&self, depths: &[f32], world_position: Vec3) -> f32 {
        self.project(world_position).map_or(1.0, |projected| {
            pcf(projected.truncate(), |texel| {
                self.compare(texel, projected.z, depths)
            })
        })
    }

    fn size(&self) -> Vec2 {
        Vec2::new(self.width as f32, self.height as f32)
    }

    fn reference_depth(&self, clip: Vec4) -> f32 {
        match self.projection {
            ShadowProjection::Orthographic => (clip.z / clip.w + 1.0) * 0.5,
            ShadowProjection::Perspective { far } => clip.w / far,
        }
    }

    fn project(&self, world_position: Vec3) -> Option<Vec3> {
        let clip = self.view_projection * world_position.extend(1.0);
        if !clip.is_finite() || clip.w <= 0.0 {
            return None;
        }
        let ndc = clip.truncate() / clip.w;
        // Adjacent cube faces can disagree by a few ulps at their shared edge.
        let xy_limit = match self.projection {
            ShadowProjection::Orthographic => 1.0,
            ShadowProjection::Perspective { .. } => 1.0 + 1e-5,
        };
        if !ndc.is_finite() || ndc.z.abs() > 1.0 || ndc.truncate().abs().max_element() > xy_limit {
            return None;
        }
        let pixel = ndc_to_uv(ndc.truncate()) * self.size() - 0.5;
        Some(pixel.extend(self.reference_depth(clip)))
    }

    fn compare(&self, texel: IVec2, depth: f32, depths: &[f32]) -> f32 {
        let x = texel.x.clamp(0, self.width as i32 - 1) as usize;
        let y = texel.y.clamp(0, self.height as i32 - 1) as usize;
        f32::from(depth <= depths[self.offset + y * self.width as usize + x] + self.bias)
    }
}

fn ndc_to_uv(ndc: Vec2) -> Vec2 {
    Vec2::new(ndc.x + 1.0, 1.0 - ndc.y) * 0.5
}

fn pcf(pixel: Vec2, sample: impl Fn(IVec2) -> f32) -> f32 {
    let base = pixel.floor();
    let weight = pixel - base;
    let base = base.as_ivec2();
    let top = sample(base) * (1.0 - weight.x) + sample(base + IVec2::X) * weight.x;
    let bottom = sample(base + IVec2::Y) * (1.0 - weight.x) + sample(base + IVec2::ONE) * weight.x;
    top * (1.0 - weight.y) + bottom * weight.y
}

fn cube_basis(face: usize) -> Mat3 {
    // +X, -X, +Y, -Y, +Z, -Z in the emitter's local frame.
    let forward = [
        Vec3::X,
        Vec3::NEG_X,
        Vec3::Y,
        Vec3::NEG_Y,
        Vec3::Z,
        Vec3::NEG_Z,
    ][face];
    let up = [Vec3::Y, Vec3::Y, Vec3::NEG_Z, Vec3::Z, Vec3::Y, Vec3::Y][face];
    Mat3::from_cols(forward.cross(up), up, forward)
}

fn cube_face(direction: Vec3) -> usize {
    let magnitude = direction.abs();
    let axis = if magnitude.x >= magnitude.y && magnitude.x >= magnitude.z {
        0
    } else if magnitude.y >= magnitude.z {
        1
    } else {
        2
    };
    2 * axis + usize::from(direction[axis] < 0.0)
}

fn cube_compare(
    maps: &[ShadowMap],
    depths: &[f32],
    world_position: Vec3,
    ray: Vec3,
    face: usize,
) -> f32 {
    let Some(map) = maps.get(face) else {
        return 1.0;
    };
    let ray = cube_basis(face).transpose() * ray;
    let texel = (ndc_to_uv(ray.truncate() / ray.z) * map.size())
        .floor()
        .as_ivec2();
    // Each face stores linear depth along its own view axis.
    let depth = map.reference_depth(map.view_projection * world_position.extend(1.0));
    map.compare(texel, depth, depths)
}

fn light_matrix(positions: &[Vec3], direction: Vec3) -> io::Result<Mat4> {
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for &position in positions {
        if !position.is_finite() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Non-finite shadow caster position",
            ));
        }
        min = min.min(position);
        max = max.max(position);
    }
    if min.x == f32::INFINITY {
        return Ok(Mat4::IDENTITY);
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
    Ok(projection * view)
}

/// Lay out each light's shadow maps; their depths follow one another in one atlas.
pub fn prepare_lights(
    lights: Vec<Light>,
    positions: &[Vec3],
    config: &ShadowConfig,
) -> io::Result<(Vec<RenderLight>, Vec<ShadowMap>)> {
    let mut maps = Vec::new();
    let lights = lights
        .into_iter()
        .map(|light| {
            let shadow = config
                .enabled
                .then(|| LightShadow::new(positions, &light, config, &mut maps))
                .transpose()?;
            Ok(RenderLight { light, shadow })
        })
        .collect::<io::Result<_>>()?;
    Ok((lights, maps))
}
