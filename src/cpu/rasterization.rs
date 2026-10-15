use std::io;

use glam::{DVec2, DVec3, DVec4, Vec2, Vec3, Vec4};

use crate::{
    framebuffer::DepthBuffer,
    geometry::{ClipPrimitive, ClipVertex, Triangle},
    material::Material,
    pipeline::{Fragment, FragmentAttributes, RasterizationStage, RasterizedBatch},
    shadow::ShadowProjection,
    texture::UvSample,
};

#[derive(Clone, Copy)]
pub(crate) enum RasterPass {
    Camera,
    Capture,
    Shadow(ShadowProjection),
}

pub(crate) fn rasterize(
    primitive: &ScreenPrimitive,
    material: &Material,
    depth_buffer: &mut DepthBuffer,
    pass: RasterPass,
    mut emit: impl FnMut(usize, &Material, &FragmentAttributes, bool),
) -> u64 {
    let linear_far = match pass {
        RasterPass::Shadow(ShadowProjection::Perspective { far }) => Some(far),
        _ => None,
    };
    let (width, height) = (depth_buffer.width, depth_buffer.height);
    let mut covered_fragments = 0;
    for triangle in &primitive.triangles {
        let vertices = triangle
            .indices
            .map(|index| primitive.vertices[index as usize]);
        let [a, b, c] = vertices.map(|v| v.position.as_dvec3());
        let ab = b - a;
        let ac = c - a;
        let area = ab.truncate().perp_dot(ac.truncate());
        // Downward viewport Y reverses winding.
        let front_facing = (area < 0.0) != primitive.mirrored;
        if area == 0.0
            || (!front_facing && !material.double_sided && !matches!(pass, RasterPass::Capture))
        {
            continue;
        }
        let depths = vertices.map(|v| {
            if linear_far.is_some() {
                v.inv_w
            } else {
                v.position.z
            }
        });
        // Projected depth and 1/w are screen-linear.
        let depth_bias = if matches!(pass, RasterPass::Shadow(_)) {
            let db = f64::from(depths[1]) - f64::from(depths[0]);
            let dc = f64::from(depths[2]) - f64::from(depths[0]);
            let dx = (db * ac.y - dc * ab.y) / area;
            let dy = (ab.x * dc - ac.x * db) / area;
            // A bilinear PCF tap can be up to one texel from the receiver.
            let bias = (dx.abs() + dy.abs()) as f32;
            if linear_far.is_some() { -bias } else { bias }
        } else {
            0.0
        };
        let gradients = UvGradients::new(vertices);
        rasterize_triangle(
            vertices.map(|v| v.position.truncate()),
            width,
            height,
            |sample| {
                covered_fragments += 1;
                let depth = depths
                    .iter()
                    .zip(sample.barycentric)
                    .map(|(depth, weight)| depth * weight)
                    .sum::<f32>()
                    + depth_bias;
                let depth = if let Some(far) = linear_far {
                    if depth <= 0.0 {
                        return;
                    }
                    depth.recip() / far
                } else {
                    depth
                };
                let [x, y] = sample.pixel;
                let index = y as usize * width as usize + x as usize;
                if !depth.is_finite() || depth >= depth_buffer.values[index] {
                    return;
                }
                if matches!(pass, RasterPass::Shadow(_)) && material.alpha_cutoff.is_none() {
                    depth_buffer.values[index] = depth;
                    return;
                }
                let attributes = interpolate_attributes(vertices, sample.barycentric, &gradients);
                if material
                    .alpha_cutoff
                    .is_some_and(|cutoff| material.base_color(attributes.uv).w < cutoff)
                {
                    return;
                }
                depth_buffer.values[index] = depth;
                if !matches!(pass, RasterPass::Shadow(_)) {
                    emit(index, material, &attributes, front_facing);
                }
            },
        );
    }
    covered_fragments
}

pub struct CpuRasterization;

impl RasterizationStage for CpuRasterization {
    fn rasterize(
        &mut self,
        primitives: &[&ClipPrimitive],
        material: &Material,
        depth: &mut DepthBuffer,
    ) -> io::Result<RasterizedBatch> {
        let mut batch = RasterizedBatch::default();
        let mut slot_for_sample = Vec::new();
        slot_for_sample
            .try_reserve_exact(depth.values.len())
            .map_err(io::Error::other)?;
        slot_for_sample.resize(depth.values.len(), usize::MAX);
        for primitive in primitives {
            let screen = project(primitive, depth.width, depth.height)?;
            batch.screen_vertices += screen.vertices.len();
            batch.screen_triangles += screen.triangles.len();
            batch.covered_fragments += rasterize(
                &screen,
                material,
                depth,
                RasterPass::Camera,
                |sample_index, _, attributes, front_facing| {
                    let fragment = Fragment {
                        sample_index,
                        attributes: *attributes,
                        front_facing,
                    };
                    // Replace the accepted winner in place.
                    let slot = &mut slot_for_sample[sample_index];
                    if *slot == usize::MAX {
                        *slot = batch.fragments.len();
                        batch.fragments.push(fragment);
                    } else {
                        batch.fragments[*slot] = fragment;
                    }
                },
            );
        }
        Ok(batch)
    }
}

// Left, right, bottom, top, near, far: w + sign * coordinate >= 0.
const PLANES: [(usize, f64); 6] = [
    (0, 1.0),
    (0, -1.0),
    (1, 1.0),
    (1, -1.0),
    (2, 1.0),
    (2, -1.0),
];

fn distance(vertex: Vec4, axis: usize, sign: f64) -> f64 {
    f64::from(vertex.w) + sign * f64::from(vertex[axis])
}

fn outcode(vertex: Vec4) -> u8 {
    let mut code = 0;
    for (plane, (axis, sign)) in PLANES.into_iter().enumerate() {
        if distance(vertex, axis, sign) < 0.0 {
            code |= 1 << plane;
        }
    }
    code
}

fn intersection(outside: ClipVertex, inside: ClipVertex, axis: usize, sign: f64) -> ClipVertex {
    let a = distance(outside.position, axis, sign);
    let b = distance(inside.position, axis, sign);
    let t = a / (a - b);
    let mut position =
        (outside.position.as_dvec4() * (1.0 - t) + inside.position.as_dvec4() * t).as_vec4();
    position[axis] = -sign as f32 * position.w;
    let world_position = (outside.world_position.as_dvec3() * (1.0 - t)
        + inside.world_position.as_dvec3() * t)
        .as_vec3();
    let uv = |get: fn(ClipVertex) -> Vec2| {
        (get(outside).as_dvec2() * (1.0 - t) + get(inside).as_dvec2() * t).as_vec2()
    };
    let tangent =
        (outside.tangent.as_dvec4() * (1.0 - t) + inside.tangent.as_dvec4() * t).as_vec4();
    let normal = (outside.normal.as_dvec3() * (1.0 - t) + inside.normal.as_dvec3() * t).as_vec3();
    ClipVertex {
        position,
        world_position,
        uv: uv(|v| v.uv),
        normal_uv: uv(|v| v.normal_uv),
        metallic_roughness_uv: uv(|v| v.metallic_roughness_uv),
        occlusion_uv: uv(|v| v.occlusion_uv),
        emissive_uv: uv(|v| v.emissive_uv),
        lightmap_uv: uv(|v| v.lightmap_uv),
        normal,
        tangent,
    }
}

fn clip_primitive(primitive: &ClipPrimitive) -> io::Result<ClipPrimitive> {
    let mut output = ClipPrimitive {
        source: primitive.source,
        mirrored: primitive.mirrored,
        vertices: Vec::new(),
        triangles: Vec::new(),
    };
    let mut polygon = Vec::with_capacity(9);
    let mut scratch = Vec::with_capacity(9);

    for triangle in &primitive.triangles {
        let vertices = triangle
            .indices
            .map(|index| primitive.vertices[index as usize]);
        let codes = vertices.map(|vertex| outcode(vertex.position));
        if codes[0] & codes[1] & codes[2] != 0 {
            continue;
        }
        polygon.clear();
        polygon.extend(vertices);
        if codes[0] | codes[1] | codes[2] != 0 {
            for (axis, sign) in PLANES {
                if polygon.len() < 3 {
                    break;
                }
                scratch.clear();
                let mut previous = *polygon.last().unwrap();
                let mut previous_distance = distance(previous.position, axis, sign);
                for &current in &polygon {
                    let current_distance = distance(current.position, axis, sign);
                    if previous_distance < 0.0 && current_distance > 0.0 {
                        scratch.push(intersection(previous, current, axis, sign));
                    } else if current_distance < 0.0 && previous_distance > 0.0 {
                        scratch.push(intersection(current, previous, axis, sign));
                    }
                    if current_distance >= 0.0 {
                        scratch.push(current);
                    }
                    previous = current;
                    previous_distance = current_distance;
                }
                std::mem::swap(&mut polygon, &mut scratch);
            }
        }

        // The homogeneous origin cannot undergo perspective division.
        polygon.retain(|vertex| vertex.position.w > 0.0);
        if polygon.len() < 3 {
            continue;
        }
        let end = u32::try_from(output.vertices.len() + polygon.len())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Too many clipped vertices"))?;
        let base = end - polygon.len() as u32;
        output.vertices.extend_from_slice(&polygon);
        for index in 1..polygon.len() as u32 - 1 {
            output.triangles.push(Triangle {
                indices: [base, base + index, base + index + 1],
                source_index: triangle.source_index,
            });
        }
    }
    Ok(output)
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ScreenVertex {
    pub position: Vec3,
    pub world_position: Vec3,
    pub inv_w: f32,
    pub uv: Vec2,
    pub normal_uv: Vec2,
    pub metallic_roughness_uv: Vec2,
    pub occlusion_uv: Vec2,
    pub emissive_uv: Vec2,
    pub lightmap_uv: Vec2,
    /// World-space normal.
    pub normal: Vec3,
    pub tangent: Vec4,
}

#[derive(Debug)]
pub(crate) struct ScreenPrimitive {
    pub mirrored: bool,
    pub vertices: Vec<ScreenVertex>,
    pub triangles: Vec<Triangle>,
}

struct UvGradients {
    inv_w: DVec2,
    sets: [[DVec2; 2]; 5],
}

impl UvGradients {
    fn new(vertices: [ScreenVertex; 3]) -> Self {
        let [a, b, c] = vertices.map(|v| v.position.truncate().as_dvec2());
        let ab = b - a;
        let ac = c - a;
        let area = ab.perp_dot(ac);
        let b = DVec2::new(ac.y, -ac.x) / area;
        let c = DVec2::new(-ab.y, ab.x) / area;
        let weights: [DVec2; 3] =
            std::array::from_fn(|i| [-b - c, b, c][i] * f64::from(vertices[i].inv_w));
        let uvs = vertices.map(|v| {
            [
                v.uv,
                v.normal_uv,
                v.metallic_roughness_uv,
                v.occlusion_uv,
                v.emissive_uv,
            ]
        });
        Self {
            inv_w: weights.iter().sum(),
            sets: std::array::from_fn(|set| {
                [
                    (0..3).map(|i| uvs[i][set].as_dvec2() * weights[i].x).sum(),
                    (0..3).map(|i| uvs[i][set].as_dvec2() * weights[i].y).sum(),
                ]
            }),
        }
    }

    fn sample(&self, set: usize, uv: DVec2, inv_w: f64) -> UvSample {
        let [dx, dy] = self.sets[set];
        UvSample {
            uv: uv.as_vec2(),
            // Quotient rule for (UV/w) / (1/w).
            dx: ((dx - uv * self.inv_w.x) / inv_w).as_vec2(),
            dy: ((dy - uv * self.inv_w.y) / inv_w).as_vec2(),
        }
    }
}

pub(crate) fn project(
    primitive: &ClipPrimitive,
    width: u32,
    height: u32,
) -> io::Result<ScreenPrimitive> {
    let clipped = clip_primitive(primitive)?;
    Ok(ScreenPrimitive {
        mirrored: clipped.mirrored,
        vertices: clipped
            .vertices
            .into_iter()
            .map(|vertex| project_vertex(vertex, width as f32, height as f32))
            .collect::<io::Result<_>>()?,
        triangles: clipped.triangles,
    })
}

fn project_vertex(vertex: ClipVertex, width: f32, height: f32) -> io::Result<ScreenVertex> {
    let clip = vertex.position;
    if !clip.is_finite() || clip.w <= 0.0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Perspective division requires a finite vertex with positive w",
        ));
    }
    let inv_w = clip.w.recip();
    let ndc = clip.truncate() / clip.w;
    let position = Vec3::new(
        (ndc.x + 1.0) * 0.5 * width,
        (1.0 - ndc.y) * 0.5 * height,
        (ndc.z + 1.0) * 0.5,
    );
    if !position.is_finite() || !inv_w.is_finite() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Perspective division produced a non-finite result",
        ));
    }
    Ok(ScreenVertex {
        position,
        world_position: vertex.world_position,
        inv_w,
        uv: vertex.uv,
        normal_uv: vertex.normal_uv,
        metallic_roughness_uv: vertex.metallic_roughness_uv,
        occlusion_uv: vertex.occlusion_uv,
        emissive_uv: vertex.emissive_uv,
        lightmap_uv: vertex.lightmap_uv,
        normal: vertex.normal,
        tangent: vertex.tangent,
    })
}

fn interpolate_attributes(
    vertices: [ScreenVertex; 3],
    barycentric: [f32; 3],
    gradients: &UvGradients,
) -> FragmentAttributes {
    let weights: [f64; 3] =
        std::array::from_fn(|i| f64::from(barycentric[i]) * f64::from(vertices[i].inv_w));
    let world_position: DVec3 = vertices
        .iter()
        .zip(weights)
        .map(|(vertex, weight)| vertex.world_position.as_dvec3() * weight)
        .sum();
    let sum = weights.iter().sum::<f64>();
    let uv = |get: fn(&ScreenVertex) -> Vec2| {
        vertices
            .iter()
            .zip(weights)
            .map(|(vertex, weight)| get(vertex).as_dvec2() * weight)
            .sum::<DVec2>()
            / sum
    };
    let tangent: DVec4 = vertices
        .iter()
        .zip(weights)
        .map(|(vertex, weight)| vertex.tangent.as_dvec4() * weight)
        .sum();
    let normal: DVec3 = vertices
        .iter()
        .zip(weights)
        .map(|(vertex, weight)| vertex.normal.as_dvec3() * weight)
        .sum();
    FragmentAttributes {
        world_position: (world_position / sum).as_vec3(),
        uv: gradients.sample(0, uv(|v| v.uv), sum),
        normal_uv: gradients.sample(1, uv(|v| v.normal_uv), sum),
        metallic_roughness_uv: gradients.sample(2, uv(|v| v.metallic_roughness_uv), sum),
        occlusion_uv: gradients.sample(3, uv(|v| v.occlusion_uv), sum),
        emissive_uv: gradients.sample(4, uv(|v| v.emissive_uv), sum),
        lightmap_uv: uv(|v| v.lightmap_uv).as_vec2(),
        normal: (normal / sum).normalize_or_zero().as_vec3(),
        tangent: (tangent / sum).as_vec4(),
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct CoverageSample {
    pub pixel: [u32; 2],
    /// Screen-space weights in the input vertex order.
    pub barycentric: [f32; 3],
}

#[derive(Clone, Copy)]
struct Edge {
    a: f64,
    b: f64,
    c: f64,
    inclusive: bool,
}

impl Edge {
    fn new(start: DVec2, end: DVec2) -> Self {
        let delta = end - start;
        Self {
            a: -delta.y,
            b: delta.x,
            c: start.x * end.y - start.y * end.x,
            inclusive: delta.y < 0.0 || (delta.y == 0.0 && delta.x > 0.0),
        }
    }

    fn evaluate(self, point: DVec2) -> f64 {
        self.a * point.x + self.b * point.y + self.c
    }
}

pub(crate) fn rasterize_triangle(
    vertices: [Vec2; 3],
    width: u32,
    height: u32,
    mut emit: impl FnMut(CoverageSample),
) {
    if width == 0 || height == 0 || vertices.iter().any(|v| !v.is_finite()) {
        return;
    }
    let mut vertices = vertices.map(|v| v.as_dvec2());
    let area = Edge::new(vertices[0], vertices[1]).evaluate(vertices[2]);
    if area == 0.0 {
        return;
    }
    let reversed = area < 0.0;
    if reversed {
        vertices.swap(1, 2);
    }
    let inv_area = area.abs().recip();
    let edges = [
        Edge::new(vertices[1], vertices[2]),
        Edge::new(vertices[2], vertices[0]),
        Edge::new(vertices[0], vertices[1]),
    ];

    let min = vertices[0].min(vertices[1]).min(vertices[2]);
    let max = vertices[0].max(vertices[1]).max(vertices[2]);
    let x0 = (min.x - 0.5).ceil().clamp(0.0, f64::from(width)) as u32;
    let y0 = (min.y - 0.5).ceil().clamp(0.0, f64::from(height)) as u32;
    let x1 = ((max.x - 0.5).floor() + 1.0).clamp(0.0, f64::from(width)) as u32;
    let y1 = ((max.y - 0.5).floor() + 1.0).clamp(0.0, f64::from(height)) as u32;

    for y in y0..y1 {
        for x in x0..x1 {
            let point = DVec2::new(f64::from(x) + 0.5, f64::from(y) + 0.5);
            let values = edges.map(|edge| edge.evaluate(point));
            if edges
                .iter()
                .zip(values)
                .any(|(edge, value)| value < 0.0 || (value == 0.0 && !edge.inclusive))
            {
                continue;
            }
            let mut barycentric = values.map(|value| (value * inv_area) as f32);
            if reversed {
                barycentric.swap(1, 2);
            }
            emit(CoverageSample {
                pixel: [x, y],
                barycentric,
            });
        }
    }
}
