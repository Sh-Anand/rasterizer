use std::io;

use glam::{DVec2, DVec3, DVec4, Vec2, Vec3, Vec4};

use crate::{
    geometry::{PrimitiveSource, Triangle},
    texture::UvSample,
    vertex::{ClipPrimitive, ClipScene, ClipVertex},
};

#[derive(Debug, Clone, Copy)]
pub struct ScreenVertex {
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
pub struct ScreenPrimitive {
    pub source: PrimitiveSource,
    pub mirrored: bool,
    pub vertices: Vec<ScreenVertex>,
    pub triangles: Vec<Triangle>,
}

#[derive(Debug)]
pub struct ScreenScene {
    pub scene_index: usize,
    pub primitives: Vec<ScreenPrimitive>,
}

pub struct FragmentAttributes {
    pub world_position: Vec3,
    pub uv: UvSample,
    pub normal_uv: UvSample,
    pub metallic_roughness_uv: UvSample,
    pub occlusion_uv: UvSample,
    pub emissive_uv: UvSample,
    pub lightmap_uv: Vec2,
    pub normal: Vec3,
    pub tangent: Vec4,
}

pub struct UvGradients {
    inv_w: DVec2,
    sets: [[DVec2; 2]; 5],
}

impl UvGradients {
    pub fn new(vertices: [ScreenVertex; 3]) -> Self {
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

pub fn project_scene(scene: &ClipScene, width: u32, height: u32) -> io::Result<ScreenScene> {
    if width == 0 || height == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Image width and height must be nonzero",
        ));
    }
    let primitives = scene
        .primitives
        .iter()
        .map(|primitive| project_primitive(primitive, width, height))
        .collect::<io::Result<_>>()?;
    Ok(ScreenScene {
        scene_index: scene.scene_index,
        primitives,
    })
}

pub(crate) fn project_primitive(
    primitive: &ClipPrimitive,
    width: u32,
    height: u32,
) -> io::Result<ScreenPrimitive> {
    Ok(ScreenPrimitive {
        source: primitive.source,
        mirrored: primitive.mirrored,
        vertices: primitive
            .vertices
            .iter()
            .map(|&vertex| project_vertex(vertex, width as f32, height as f32))
            .collect::<io::Result<_>>()?,
        triangles: primitive.triangles.clone(),
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

pub fn interpolate_attributes(
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
    let uv: DVec2 = vertices
        .iter()
        .zip(weights)
        .map(|(vertex, weight)| vertex.uv.as_dvec2() * weight)
        .sum();
    let normal_uv: DVec2 = vertices
        .iter()
        .zip(weights)
        .map(|(vertex, weight)| vertex.normal_uv.as_dvec2() * weight)
        .sum();
    let tangent: DVec4 = vertices
        .iter()
        .zip(weights)
        .map(|(vertex, weight)| vertex.tangent.as_dvec4() * weight)
        .sum();
    let emissive_uv: DVec2 = vertices
        .iter()
        .zip(weights)
        .map(|(vertex, weight)| vertex.emissive_uv.as_dvec2() * weight)
        .sum();
    let metallic_roughness_uv: DVec2 = vertices
        .iter()
        .zip(weights)
        .map(|(vertex, weight)| vertex.metallic_roughness_uv.as_dvec2() * weight)
        .sum();
    let normal: DVec3 = vertices
        .iter()
        .zip(weights)
        .map(|(vertex, weight)| vertex.normal.as_dvec3() * weight)
        .sum();
    let occlusion_uv: DVec2 = vertices
        .iter()
        .zip(weights)
        .map(|(vertex, weight)| vertex.occlusion_uv.as_dvec2() * weight)
        .sum();
    let lightmap_uv: DVec2 = vertices
        .iter()
        .zip(weights)
        .map(|(vertex, weight)| vertex.lightmap_uv.as_dvec2() * weight)
        .sum();
    let sum = weights.iter().sum::<f64>();
    FragmentAttributes {
        world_position: (world_position / sum).as_vec3(),
        uv: gradients.sample(0, uv / sum, sum),
        normal_uv: gradients.sample(1, normal_uv / sum, sum),
        metallic_roughness_uv: gradients.sample(2, metallic_roughness_uv / sum, sum),
        occlusion_uv: gradients.sample(3, occlusion_uv / sum, sum),
        emissive_uv: gradients.sample(4, emissive_uv / sum, sum),
        lightmap_uv: (lightmap_uv / sum).as_vec2(),
        normal: (normal / sum).normalize_or_zero().as_vec3(),
        tangent: (tangent / sum).as_vec4(),
    }
}
