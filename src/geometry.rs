use glam::{Mat3, Mat4, Vec2, Vec3, Vec4};

mod prepare;
mod tangent;

pub use prepare::prepare_scene;

/// Camera-independent geometry, with flat-normal and tangent seams already split.
#[derive(Debug, Clone)]
pub struct PreparedScene {
    pub scene_index: usize,
    pub primitives: Vec<PreparedPrimitive>,
}

#[derive(Debug, Clone)]
pub struct PreparedPrimitive {
    pub source: PrimitiveSource,
    pub model: Mat4,
    pub normal_matrix: Mat3,
    pub mirrored: bool,
    /// Generated flat normals may be zero for degenerate triangles.
    pub flat_shaded: bool,
    pub vertices: Vec<PreparedVertex>,
    pub triangles: Vec<Triangle>,
}

#[derive(Debug, Clone, Copy)]
pub struct PreparedVertex {
    /// Object-space position.
    pub position: Vec3,
    pub uv: Vec2,
    pub normal_uv: Vec2,
    pub metallic_roughness_uv: Vec2,
    pub occlusion_uv: Vec2,
    pub emissive_uv: Vec2,
    pub lightmap_uv: Vec2,
    /// Object-space normal, normalized after transformation to world space.
    pub normal: Vec3,
    /// Object-space tangent; W is the bitangent sign (zero for an unusable frame).
    pub tangent: Vec4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrimitiveSource {
    pub node_index: usize,
    pub mesh_index: usize,
    pub primitive_index: usize,
    pub material_index: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Triangle {
    pub indices: [u32; 3],
    /// Triangle index within the source mesh primitive.
    pub source_index: usize,
}
