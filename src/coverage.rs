use glam::{DVec2, Vec2};

#[derive(Debug, Clone, Copy)]
pub struct CoverageSample {
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

pub fn rasterize_triangle(
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
