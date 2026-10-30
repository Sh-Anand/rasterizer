use crate::{geometry::ClipVertex, texture::UvSample};
use glam::{Vec2, Vec3, Vec4};
use std::ops::Range;

#[derive(Default, Clone, Copy)]
pub struct FragmentAttributes {
    pub world_position: Vec3,
    pub uv: UvSample,
    pub normal: Vec3,
    pub tangent: Vec4,
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

pub(crate) fn outcode(vertex: Vec4) -> u8 {
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
    let tangent = outside.tangent * (1.0 - t) + inside.tangent * t;
    let normal = outside.normal * (1.0 - t) + inside.normal * t;
    ClipVertex {
        position,
        world_position,
        uv: outside.uv * (1.0 - t) + inside.uv * t,
        normal,
        tangent,
    }
}

#[derive(Clone, Copy)]
pub(crate) struct ScreenVertex {
    pub position: Vec3,
    pub world_position: Vec3,
    pub inv_w: f32,
    pub uv: Vec2,
    /// World-space normal.
    pub normal: Vec3,
    pub tangent: Vec4,
}

pub(crate) struct UvGradients {
    inv_w: Vec2,
    uv: [Vec2; 2],
}

impl UvGradients {
    pub(crate) fn new(vertices: &[ScreenVertex; 3]) -> Self {
        let [a, b, c] = std::array::from_fn(|i| vertices[i].position.truncate());
        let ab = b - a;
        let ac = c - a;
        let area = ab.perp_dot(ac);
        let b = Vec2::new(ac.y, -ac.x) / area;
        let c = Vec2::new(-ab.y, ab.x) / area;
        let weights: [Vec2; 3] = std::array::from_fn(|i| [-b - c, b, c][i] * vertices[i].inv_w);
        Self {
            inv_w: weights.iter().sum(),
            uv: [
                (0..3).map(|i| vertices[i].uv * weights[i].x).sum(),
                (0..3).map(|i| vertices[i].uv * weights[i].y).sum(),
            ],
        }
    }

    fn sample(&self, uv: Vec2, inv_w: f32) -> UvSample {
        let [dx, dy] = self.uv;
        UvSample {
            uv,
            // Quotient rule for (UV/w) / (1/w).
            dx: (dx - uv * self.inv_w.x) / inv_w,
            dy: (dy - uv * self.inv_w.y) / inv_w,
        }
    }
}

/// Non-finite results fail `Triangle::new`.
pub(crate) fn project_vertex(vertex: ClipVertex, width: f32, height: f32) -> ScreenVertex {
    let clip = vertex.position;
    let ndc = clip.truncate() / clip.w;
    ScreenVertex {
        position: Vec3::new(
            (ndc.x + 1.0) * 0.5 * width,
            (1.0 - ndc.y) * 0.5 * height,
            (ndc.z + 1.0) * 0.5,
        ),
        world_position: vertex.world_position,
        inv_w: clip.w.recip(),
        uv: vertex.uv,
        normal: vertex.normal,
        tangent: vertex.tangent,
    }
}

pub(crate) fn interpolate_attributes(
    vertices: &[ScreenVertex; 3],
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
    let uv = vertices
        .iter()
        .zip(weights)
        .map(|(vertex, weight)| vertex.uv * weight)
        .sum::<Vec2>()
        / sum;
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
        uv: gradients.sample(uv, sum),
        normal: (normal / sum).normalize_or_zero(),
        tangent: tangent / sum,
    }
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

pub(crate) fn sample_range(min: f32, max: f32, bounds: Range<u32>) -> Range<u32> {
    let lo = bounds.start as f32;
    let hi = bounds.end as f32;
    (min - 0.5).ceil().clamp(lo, hi) as u32..((max - 0.5).floor() + 1.0).clamp(lo, hi) as u32
}

#[derive(Default)]
pub(crate) struct Polygon {
    pub vertices: [ClipVertex; 9],
    pub len: usize,
}

impl Polygon {
    /// Six planes clip a triangle to at most nine vertices; any more are dropped.
    fn push(&mut self, vertex: ClipVertex) {
        if let Some(slot) = self.vertices.get_mut(self.len) {
            *slot = vertex;
            self.len += 1;
        }
    }
}

/// Clips a triangle that crosses a clipping plane.
pub(crate) fn clip_triangle(
    vertex: impl Fn(u32) -> ClipVertex,
    triangle: [u32; 3],
    polygon: &mut Polygon,
) {
    polygon.len = 0;
    for i in triangle {
        polygon.push(vertex(i));
    }
    let mut scratch = Polygon::default();
    let (mut current, mut next) = (&mut *polygon, &mut scratch);
    let mut flipped = false;
    for (axis, sign) in PLANES {
        if current.len == 0 {
            break;
        }
        next.len = 0;
        let mut previous = current.vertices[current.len - 1];
        let mut previous_distance = distance(previous.position, axis, sign);
        for &vertex in &current.vertices[..current.len] {
            let current_distance = distance(vertex.position, axis, sign);
            if (previous_distance < 0.0) != (current_distance < 0.0) {
                next.push(if previous_distance < 0.0 {
                    intersection(previous, vertex, axis, sign)
                } else {
                    intersection(vertex, previous, axis, sign)
                });
            }
            if current_distance >= 0.0 {
                next.push(vertex);
            }
            previous = vertex;
            previous_distance = current_distance;
        }
        std::mem::swap(&mut current, &mut next);
        flipped = !flipped;
    }
    if flipped {
        for i in 0..current.len {
            next.vertices[i] = current.vertices[i];
        }
        next.len = current.len;
    }
    let mut len = 0;
    for i in 0..polygon.len {
        if polygon.vertices[i].position.w > 0.0 {
            polygon.vertices[len] = polygon.vertices[i];
            len += 1;
        }
    }
    polygon.len = len;
}

#[derive(Clone, Copy)]
pub(crate) struct Triangle {
    edges: [Edge; 3],
    inv_area: f32,
    reversed: bool,
    min: Vec2,
    max: Vec2,
    depths: [f32; 3],
    bias: f32,
    pub front_facing: bool,
}

impl Triangle {
    pub(crate) fn new(
        vertices: &[ScreenVertex; 3],
        mirrored: bool,
        double_sided: bool,
        shadow: bool,
    ) -> Option<Self> {
        let [a, b, c] = std::array::from_fn(|i| vertices[i].position);
        let ab = b - a;
        let ac = c - a;
        let area = ab.truncate().perp_dot(ac.truncate());
        let front_facing = (area < 0.0) != mirrored;
        if area == 0.0 || (!front_facing && !double_sided) {
            return None;
        }
        let depths = std::array::from_fn(|i| vertices[i].position.z);
        // Shadow depths add a one-texel slope bias covering the PCF footprint.
        let bias = if shadow {
            let db = depths[1] - depths[0];
            let dc = depths[2] - depths[0];
            let dx = (db * ac.y - dc * ab.y) / area;
            let dy = (ab.x * dc - ac.x * db) / area;
            dx.abs() + dy.abs()
        } else {
            0.0
        };
        let mut points: [Vec2; 3] = std::array::from_fn(|i| vertices[i].position.truncate());
        if points.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let area = Edge::new(points[0], points[1]).evaluate(points[2]);
        if area == 0.0 {
            return None;
        }
        let reversed = area < 0.0;
        if reversed {
            points.swap(1, 2);
        }
        Some(Self {
            edges: [
                Edge::new(points[1], points[2]),
                Edge::new(points[2], points[0]),
                Edge::new(points[0], points[1]),
            ],
            inv_area: area.abs().recip(),
            reversed,
            min: points[0].min(points[1]).min(points[2]),
            max: points[0].max(points[1]).max(points[2]),
            depths,
            bias,
            front_facing,
        })
    }

    pub(crate) fn bounds(&self, width: u32, rows: Range<u32>) -> (Range<u32>, Range<u32>) {
        (
            sample_range(self.min.x, self.max.x, 0..width),
            sample_range(self.min.y, self.max.y, rows),
        )
    }

    pub(crate) fn coverage(&self, x: u32, y: u32) -> Option<[f32; 3]> {
        let point = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
        let values = self.edges.map(|edge| edge.evaluate(point));
        if self
            .edges
            .iter()
            .zip(values)
            .any(|(edge, value)| value < 0.0 || (value == 0.0 && !edge.inclusive))
        {
            return None;
        }
        let mut barycentric = values.map(|value| value * self.inv_area);
        if self.reversed {
            barycentric.swap(1, 2);
        }
        Some(barycentric)
    }

    pub(crate) fn depth(&self, barycentric: [f32; 3]) -> Option<f32> {
        let depth = self
            .depths
            .iter()
            .zip(barycentric)
            .map(|(depth, weight)| depth * weight)
            .sum::<f32>()
            + self.bias;
        depth.is_finite().then_some(depth)
    }
}
