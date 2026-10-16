use std::{io, ops::Range};

use glam::{Vec2, Vec3, Vec4};
use rayon::prelude::*;

use crate::{
    framebuffer::DepthBuffer,
    geometry::{ClipPrimitive, ClipVertex},
    material::Material,
    pipeline::{
        Fragment, FragmentAttributes, RasterizationStage, RasterizedScene, ShadowProjection,
    },
    texture::UvSample,
};

pub(super) fn rasterize(
    primitive: &ScreenPrimitive,
    double_sided: bool,
    width: u32,
    y_start: u32,
    depth_buffer: &mut [f32],
    shadow: Option<ShadowProjection>,
    mut emit: impl FnMut(usize, &FragmentAttributes, bool),
) -> u64 {
    let linear_far = match shadow {
        Some(ShadowProjection::Perspective { far }) => Some(far),
        _ => None,
    };
    let y_end = y_start + (depth_buffer.len() / width as usize) as u32;
    let first_sample = y_start as usize * width as usize;
    let mut covered_fragments = 0;
    for triangle in &primitive.triangles {
        let vertices = triangle.map(|index| primitive.vertices[index as usize]);
        let [a, b, c] = vertices.map(|v| v.position);
        if a.y.max(b.y).max(c.y) < y_start as f32 + 0.5
            || a.y.min(b.y).min(c.y) > y_end as f32 - 0.5
        {
            continue;
        }
        let ab = b - a;
        let ac = c - a;
        let area = ab.truncate().perp_dot(ac.truncate());
        // Downward viewport Y reverses winding.
        let front_facing = (area < 0.0) != primitive.mirrored;
        if area == 0.0 || (!front_facing && !double_sided) {
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
        let depth_bias = if shadow.is_some() {
            let db = depths[1] - depths[0];
            let dc = depths[2] - depths[0];
            let dx = (db * ac.y - dc * ab.y) / area;
            let dy = (ab.x * dc - ac.x * db) / area;
            // A bilinear PCF tap can be up to one texel from the receiver.
            let bias = dx.abs() + dy.abs();
            if linear_far.is_some() { -bias } else { bias }
        } else {
            0.0
        };
        let gradients = shadow.is_none().then(|| UvGradients::new(vertices));
        rasterize_triangle(
            vertices.map(|v| v.position.truncate()),
            width,
            y_start..y_end,
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
                let index = (y - y_start) as usize * width as usize + x as usize;
                if !depth.is_finite() || depth >= depth_buffer[index] {
                    return;
                }
                depth_buffer[index] = depth;
                if let Some(gradients) = &gradients {
                    let attributes =
                        interpolate_attributes(vertices, sample.barycentric, gradients);
                    emit(first_sample + index, &attributes, front_facing);
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
        materials: &[Material],
        depth: &mut DepthBuffer,
        shadow: Option<ShadowProjection>,
    ) -> io::Result<RasterizedScene> {
        let screens = primitives
            .par_iter()
            .map(|primitive| project(primitive, depth.width, depth.height))
            .collect::<io::Result<Vec<_>>>()?;
        let mut output = RasterizedScene {
            screen_vertices: screens.iter().map(|screen| screen.vertices.len()).sum(),
            screen_triangles: screens.iter().map(|screen| screen.triangles.len()).sum(),
            ..RasterizedScene::default()
        };
        let width = depth.width;
        let band_height = (depth.height as usize).div_ceil(rayon::current_num_threads());
        let band_samples = band_height * width as usize;
        let default_material = Material::default();
        let bands = depth
            .values
            .par_chunks_mut(band_samples)
            .enumerate()
            .map(|(band, values)| {
                let mut output = RasterizedScene::default();
                let mut slot_for_sample = Vec::new();
                if shadow.is_none() {
                    slot_for_sample
                        .try_reserve_exact(values.len())
                        .map_err(io::Error::other)?;
                    slot_for_sample.resize(values.len(), usize::MAX);
                }
                for (primitive, screen) in primitives.iter().zip(&screens) {
                    let material_index = primitive.source.material_index;
                    let material =
                        material_index.map_or(&default_material, |index| &materials[index]);
                    output.covered_fragments += rasterize(
                        screen,
                        material.double_sided,
                        width,
                        (band * band_height) as u32,
                        values,
                        shadow,
                        |sample_index, attributes, front_facing| {
                            let fragment = Fragment {
                                sample_index,
                                material_index,
                                attributes: *attributes,
                                front_facing,
                            };
                            let slot = &mut slot_for_sample[sample_index - band * band_samples];
                            if *slot == usize::MAX {
                                *slot = output.fragments.len();
                                output.fragments.push(fragment);
                            } else {
                                output.fragments[*slot] = fragment;
                            }
                        },
                    );
                }
                Ok(output)
            })
            .collect::<io::Result<Vec<_>>>()?;
        output
            .fragments
            .try_reserve_exact(bands.iter().map(|band| band.fragments.len()).sum())
            .map_err(io::Error::other)?;
        for band in bands {
            output.covered_fragments += band.covered_fragments;
            output.fragments.extend(band.fragments);
        }
        Ok(output)
    }
}

// Left, right, bottom, top, near, far: w + sign * coordinate >= 0.
const PLANES: [(usize, f32); 6] = [
    (0, 1.0),
    (0, -1.0),
    (1, 1.0),
    (1, -1.0),
    (2, 1.0),
    (2, -1.0),
];

fn distance(vertex: Vec4, axis: usize, sign: f32) -> f32 {
    vertex.w + sign * vertex[axis]
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

fn intersection(outside: ClipVertex, inside: ClipVertex, axis: usize, sign: f32) -> ClipVertex {
    let a = distance(outside.position, axis, sign);
    let b = distance(inside.position, axis, sign);
    let t = a / (a - b);
    let mut position = outside.position * (1.0 - t) + inside.position * t;
    position[axis] = -sign * position.w;
    let world_position = outside.world_position * (1.0 - t) + inside.world_position * t;
    let uv = |get: fn(ClipVertex) -> Vec2| get(outside) * (1.0 - t) + get(inside) * t;
    let tangent = outside.tangent * (1.0 - t) + inside.tangent * t;
    let normal = outside.normal * (1.0 - t) + inside.normal * t;
    ClipVertex {
        position,
        world_position,
        uv: uv(|v| v.uv),
        normal_uv: uv(|v| v.normal_uv),
        metallic_roughness_uv: uv(|v| v.metallic_roughness_uv),
        emissive_uv: uv(|v| v.emissive_uv),
        normal,
        tangent,
    }
}

pub(super) fn project(
    primitive: &ClipPrimitive,
    width: u32,
    height: u32,
) -> io::Result<ScreenPrimitive> {
    let mut output = ScreenPrimitive {
        mirrored: primitive.mirrored,
        vertices: Vec::new(),
        triangles: Vec::new(),
    };
    let mut polygon = Vec::with_capacity(9);
    let mut scratch = Vec::with_capacity(9);

    for triangle in &primitive.triangles {
        let vertices = triangle.map(|index| primitive.vertices[index as usize]);
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
        for &vertex in &polygon {
            let vertex = project_vertex(vertex, width as f32, height as f32)?;
            output.vertices.push(vertex);
        }
        for index in 1..polygon.len() as u32 - 1 {
            output
                .triangles
                .push([base, base + index, base + index + 1]);
        }
    }
    Ok(output)
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ScreenVertex {
    pub position: Vec3,
    pub world_position: Vec3,
    pub inv_w: f32,
    pub uv: Vec2,
    pub normal_uv: Vec2,
    pub metallic_roughness_uv: Vec2,
    pub emissive_uv: Vec2,
    /// World-space normal.
    pub normal: Vec3,
    pub tangent: Vec4,
}

#[derive(Debug)]
pub(super) struct ScreenPrimitive {
    pub mirrored: bool,
    pub vertices: Vec<ScreenVertex>,
    pub triangles: Vec<[u32; 3]>,
}

struct UvGradients {
    inv_w: Vec2,
    sets: [[Vec2; 2]; 4],
}

impl UvGradients {
    fn new(vertices: [ScreenVertex; 3]) -> Self {
        let [a, b, c] = vertices.map(|v| v.position.truncate());
        let ab = b - a;
        let ac = c - a;
        let area = ab.perp_dot(ac);
        let b = Vec2::new(ac.y, -ac.x) / area;
        let c = Vec2::new(-ab.y, ab.x) / area;
        let weights: [Vec2; 3] = std::array::from_fn(|i| [-b - c, b, c][i] * vertices[i].inv_w);
        let uvs = vertices.map(|v| [v.uv, v.normal_uv, v.metallic_roughness_uv, v.emissive_uv]);
        Self {
            inv_w: weights.iter().sum(),
            sets: std::array::from_fn(|set| {
                [
                    (0..3).map(|i| uvs[i][set] * weights[i].x).sum(),
                    (0..3).map(|i| uvs[i][set] * weights[i].y).sum(),
                ]
            }),
        }
    }

    fn sample(&self, set: usize, uv: Vec2, inv_w: f32) -> UvSample {
        let [dx, dy] = self.sets[set];
        UvSample {
            uv,
            // Quotient rule for (UV/w) / (1/w).
            dx: (dx - uv * self.inv_w.x) / inv_w,
            dy: (dy - uv * self.inv_w.y) / inv_w,
        }
    }
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
        emissive_uv: vertex.emissive_uv,
        normal: vertex.normal,
        tangent: vertex.tangent,
    })
}

fn interpolate_attributes(
    vertices: [ScreenVertex; 3],
    barycentric: [f32; 3],
    gradients: &UvGradients,
) -> FragmentAttributes {
    let weights: [f32; 3] = std::array::from_fn(|i| barycentric[i] * vertices[i].inv_w);
    let world_position: Vec3 = vertices
        .iter()
        .zip(weights)
        .map(|(vertex, weight)| vertex.world_position * weight)
        .sum();
    let sum = weights.iter().sum::<f32>();
    let uv = |get: fn(&ScreenVertex) -> Vec2| {
        vertices
            .iter()
            .zip(weights)
            .map(|(vertex, weight)| get(vertex) * weight)
            .sum::<Vec2>()
            / sum
    };
    let tangent: Vec4 = vertices
        .iter()
        .zip(weights)
        .map(|(vertex, weight)| vertex.tangent * weight)
        .sum();
    let normal: Vec3 = vertices
        .iter()
        .zip(weights)
        .map(|(vertex, weight)| vertex.normal * weight)
        .sum();
    FragmentAttributes {
        world_position: world_position / sum,
        uv: gradients.sample(0, uv(|v| v.uv), sum),
        normal_uv: gradients.sample(1, uv(|v| v.normal_uv), sum),
        metallic_roughness_uv: gradients.sample(2, uv(|v| v.metallic_roughness_uv), sum),
        emissive_uv: gradients.sample(3, uv(|v| v.emissive_uv), sum),
        normal: (normal / sum).normalize_or_zero(),
        tangent: tangent / sum,
    }
}

#[derive(Debug, Clone, Copy)]
struct CoverageSample {
    pixel: [u32; 2],
    /// Screen-space weights in the input vertex order.
    barycentric: [f32; 3],
}

#[derive(Clone, Copy)]
struct Edge {
    start: Vec2,
    delta: Vec2,
    inclusive: bool,
}

impl Edge {
    fn new(start: Vec2, end: Vec2) -> Self {
        let delta = end - start;
        Self {
            start,
            delta,
            inclusive: delta.y < 0.0 || (delta.y == 0.0 && delta.x > 0.0),
        }
    }

    fn evaluate(self, point: Vec2) -> f32 {
        // Local coordinates avoid cancellation in FP32 edge tests.
        self.delta.perp_dot(point - self.start)
    }
}

fn rasterize_triangle(
    mut vertices: [Vec2; 3],
    width: u32,
    rows: Range<u32>,
    mut emit: impl FnMut(CoverageSample),
) {
    if width == 0 || rows.is_empty() || vertices.iter().any(|v| !v.is_finite()) {
        return;
    }
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
    let x0 = (min.x - 0.5).ceil().clamp(0.0, width as f32) as u32;
    let y0 = (min.y - 0.5)
        .ceil()
        .clamp(rows.start as f32, rows.end as f32) as u32;
    let x1 = ((max.x - 0.5).floor() + 1.0).clamp(0.0, width as f32) as u32;
    let y1 = ((max.y - 0.5).floor() + 1.0).clamp(rows.start as f32, rows.end as f32) as u32;

    for y in y0..y1 {
        for x in x0..x1 {
            let point = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
            let values = edges.map(|edge| edge.evaluate(point));
            if edges
                .iter()
                .zip(values)
                .any(|(edge, value)| value < 0.0 || (value == 0.0 && !edge.inclusive))
            {
                continue;
            }
            let mut barycentric = values.map(|value| value * inv_area);
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
