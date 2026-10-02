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
