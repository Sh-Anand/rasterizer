use std::io;

use glam::{IVec2, Mat3, Mat4, Vec2, Vec3, Vec4, camera::rh};

use crate::{
    config::ShadowConfig,
    framebuffer::DepthBuffer,
    geometry::ClipScene,
    lighting::{AreaLight, Light, RenderLight},
    pipeline::ShadowProjection,
};

#[derive(Clone, Copy)]
pub enum LightShadow<M = Vec<ShadowMap>> {
    Directional(M),
    Area {
        position: Vec3,
        basis: Mat3,
        faces: M,
    },
}

impl LightShadow {
    pub fn new(scene: &ClipScene, light: &Light, config: &ShadowConfig) -> io::Result<Self> {
        match light {
            Light::Directional(light) => Ok(Self::Directional(vec![ShadowMap::new(
                light_matrix(scene, light.direction())?,
                ShadowProjection::Orthographic,
                config,
            )?])),
            Light::Area(light) => Self::area(scene, light, config),
        }
    }

    fn area(scene: &ClipScene, light: &AreaLight, config: &ShadowConfig) -> io::Result<Self> {
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
        for vertex in scene.primitives.iter().flat_map(|p| &p.vertices) {
            let distance = (vertex.world_position - position).length();
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
        let faces = (0..count)
            .map(|face| {
                let axes = basis * cube_basis(face);
                let view = rh::view::look_to_mat4(position, axes.z_axis, axes.y_axis);
                ShadowMap::new(
                    projection * view,
                    ShadowProjection::Perspective { far },
                    config,
                )
            })
            .collect::<io::Result<Vec<_>>>()?;
        Ok(Self::Area {
            position,
            basis,
            faces,
        })
    }
}

impl<M> LightShadow<M> {
    pub fn maps(&self) -> &M {
        match self {
            Self::Directional(maps) | Self::Area { faces: maps, .. } => maps,
        }
    }

    fn maps_mut(&mut self) -> &mut M {
        match self {
            Self::Directional(maps) | Self::Area { faces: maps, .. } => maps,
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn with_maps<N>(&self, maps: N) -> LightShadow<N> {
        match self {
            Self::Directional(_) => LightShadow::Directional(maps),
            Self::Area {
                position, basis, ..
            } => LightShadow::Area {
                position: *position,
                basis: *basis,
                faces: maps,
            },
        }
    }

    pub fn visibility<P: AsRef<[f32]>>(&self, world_position: Vec3) -> f32
    where
        M: AsRef<[ShadowMap<P>]>,
    {
        self.visibility_with(world_position, AsRef::as_ref)
    }

    pub(crate) fn visibility_with<'a, P: 'a>(
        &'a self,
        world_position: Vec3,
        depth: impl Fn(&'a P) -> &'a [f32],
    ) -> f32
    where
        M: AsRef<[ShadowMap<P>]>,
    {
        let maps = self.maps().as_ref();
        let get_map = |face| {
            maps.get(face)
                .map(|map: &ShadowMap<P>| map.with_depth(depth(&map.depth.values)))
        };
        match self {
            Self::Directional(_) => get_map(0).map_or(1.0, |map| map.visibility(world_position)),
            Self::Area {
                position, basis, ..
            } => {
                let direction = basis.transpose() * (world_position - position);
                if !direction.is_finite() || (maps.len() == 5 && direction.z <= 0.0) {
                    return 1.0;
                }
                let face = cube_face(direction);
                let Some(map) = get_map(face) else {
                    return 1.0;
                };
                let Some(projected) = map.project(world_position) else {
                    return 1.0;
                };
                pcf(projected.truncate(), |texel| {
                    let outside = texel.cmplt(IVec2::ZERO) | texel.cmpge(map.size().as_ivec2());
                    if !outside.any() {
                        return map.compare(texel, projected.z);
                    }
                    let uv = (texel.as_vec2() + 0.5) / map.size();
                    let ray = cube_basis(face) * Vec3::new(2.0 * uv.x - 1.0, 1.0 - 2.0 * uv.y, 1.0);
                    if outside.all() {
                        // A corner tap touches three faces; blend their comparisons.
                        (0..3)
                            .map(|axis| {
                                let neighbor = 2 * axis + usize::from(ray[axis] < 0.0);
                                cube_compare(get_map, world_position, ray, neighbor)
                            })
                            .sum::<f32>()
                            / 3.0
                    } else {
                        cube_compare(get_map, world_position, ray, cube_face(ray))
                    }
                })
            }
        }
    }
}

#[derive(Clone, Copy)]
pub struct ShadowMap<P = Vec<f32>> {
    pub view_projection: Mat4,
    pub depth: DepthBuffer<P>,
    pub projection: ShadowProjection,
    bias: f32,
}

impl ShadowMap {
    fn new(
        view_projection: Mat4,
        projection: ShadowProjection,
        config: &ShadowConfig,
    ) -> io::Result<Self> {
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
        Ok(Self {
            view_projection,
            depth: DepthBuffer::new(config.resolution, config.resolution)?,
            projection,
            bias: config.bias,
        })
    }
}

impl<P> ShadowMap<P> {
    pub(crate) fn with_depth<Q>(&self, values: Q) -> ShadowMap<Q> {
        ShadowMap {
            view_projection: self.view_projection,
            depth: DepthBuffer {
                width: self.depth.width,
                height: self.depth.height,
                values,
            },
            projection: self.projection,
            bias: self.bias,
        }
    }
}

impl<P: AsRef<[f32]>> ShadowMap<P> {
    pub fn visibility(&self, world_position: Vec3) -> f32 {
        self.project(world_position).map_or(1.0, |projected| {
            pcf(projected.truncate(), |texel| {
                self.compare(texel, projected.z)
            })
        })
    }

    fn size(&self) -> Vec2 {
        Vec2::new(self.depth.width as f32, self.depth.height as f32)
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

    fn compare(&self, texel: IVec2, depth: f32) -> f32 {
        let x = texel.x.clamp(0, self.depth.width as i32 - 1) as usize;
        let y = texel.y.clamp(0, self.depth.height as i32 - 1) as usize;
        f32::from(
            depth <= self.depth.values.as_ref()[y * self.depth.width as usize + x] + self.bias,
        )
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

fn cube_compare<P: AsRef<[f32]>>(
    get_map: impl Fn(usize) -> Option<ShadowMap<P>>,
    world_position: Vec3,
    ray: Vec3,
    face: usize,
) -> f32 {
    let Some(map) = get_map(face) else {
        return 1.0;
    };
    let ray = cube_basis(face).transpose() * ray;
    let texel = (ndc_to_uv(ray.truncate() / ray.z) * map.size())
        .floor()
        .as_ivec2();
    // Each face stores linear depth along its own view axis.
    let depth = map.reference_depth(map.view_projection * world_position.extend(1.0));
    map.compare(texel, depth)
}

fn light_matrix(scene: &ClipScene, direction: Vec3) -> io::Result<Mat4> {
    let positions = || {
        scene
            .primitives
            .iter()
            .flat_map(|p| p.vertices.iter().map(|v| v.world_position))
    };
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for position in positions() {
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
    for position in positions() {
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

pub fn prepare_lights(
    lights: Vec<Light>,
    scene: &ClipScene,
    config: &ShadowConfig,
    mut render_depth: impl FnMut(&mut ShadowMap) -> io::Result<()>,
) -> io::Result<Vec<RenderLight>> {
    lights
        .into_iter()
        .map(|light| {
            let shadow = if config.enabled {
                let mut shadow = LightShadow::new(scene, &light, config)?;
                for map in shadow.maps_mut() {
                    render_depth(map)?;
                }
                Some(shadow)
            } else {
                None
            };
            Ok(RenderLight { light, shadow })
        })
        .collect()
}
