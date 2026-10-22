use std::{collections::HashMap, io, path::Path};

use glam::{Mat3, Mat4, Vec2, Vec3, Vec4};

#[derive(Debug)]
pub struct GltfAsset {
    pub document: gltf::Document,
    pub buffers: Vec<gltf::buffer::Data>,
    pub images: Vec<gltf::image::Data>,
}

impl GltfAsset {
    pub fn load(path: impl AsRef<Path>) -> gltf::Result<Self> {
        let (document, buffers, images) = gltf::import(path)?;
        Ok(Self {
            document,
            buffers,
            images,
        })
    }
}

/// Object-space geometry with normal/tangent seams split.
#[derive(Debug, Clone)]
pub struct PreparedScene {
    pub primitives: Vec<PreparedPrimitive>,
}

#[derive(Debug, Clone)]
pub struct PreparedPrimitive {
    pub source: PrimitiveSource,
    pub model: Mat4,
    pub normal_matrix: Mat3,
    pub mirrored: bool,
    /// Allow zero normals on degenerate triangles.
    pub flat_shaded: bool,
    pub vertices: Vec<PreparedVertex>,
    pub triangles: Vec<[u32; 3]>,
}

#[derive(Debug, Clone, Copy)]
pub struct PreparedVertex {
    /// Object-space position.
    pub position: Vec3,
    pub uv: Vec2,
    /// Object-space normal; normalize after transformation.
    pub normal: Vec3,
    /// Object-space tangent; W is handedness, or zero for an unusable frame.
    pub tangent: Vec4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrimitiveSource {
    pub node_index: usize,
    pub mesh_index: usize,
    pub primitive_index: usize,
    pub material_index: Option<usize>,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct ClipVertex {
    pub position: Vec4,
    pub world_position: Vec3,
    pub uv: Vec2,
    /// World-space normal.
    pub normal: Vec3,
    /// World-space tangent; W is handedness, or zero for an unusable frame.
    pub tangent: Vec4,
}

#[derive(Clone, Copy)]
pub(crate) struct VertexTransform {
    model: Mat4,
    mvp: Mat4,
    normal: Mat3,
    linear: Mat3,
    sign: f32,
    pub(crate) flat_shaded: bool,
}

impl VertexTransform {
    pub(crate) fn new(primitive: &PreparedPrimitive, view_projection: Mat4) -> Self {
        let linear = Mat3::from_mat4(primitive.model);
        Self {
            model: primitive.model,
            mvp: view_projection * primitive.model,
            normal: primitive.normal_matrix,
            linear,
            sign: linear.determinant().signum(),
            flat_shaded: primitive.flat_shaded,
        }
    }

    pub(crate) fn apply(&self, vertex: &PreparedVertex) -> ClipVertex {
        let tangent = (self.linear * vertex.tangent.truncate())
            .try_normalize()
            .filter(|_| vertex.tangent.w.abs() == 1.0)
            .map_or(Vec4::ZERO, |t| t.extend(vertex.tangent.w * self.sign));
        ClipVertex {
            position: self.mvp * vertex.position.extend(1.0),
            world_position: self.model.transform_point3(vertex.position),
            uv: vertex.uv,
            normal: (self.normal * vertex.normal).normalize_or_zero(),
            tangent,
        }
    }
}

impl ClipVertex {
    pub(crate) fn invalid(&self, flat_shaded: bool) -> u32 {
        if !self.position.is_finite() || !self.world_position.is_finite() {
            1
        } else if !flat_shaded && self.normal == Vec3::ZERO {
            2
        } else {
            0
        }
    }
}

pub(crate) fn check_view_projection(view_projection: Mat4) -> io::Result<()> {
    if !view_projection.is_finite() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Non-finite projection matrix",
        ));
    }
    Ok(())
}

pub fn prepare_scene(asset: &GltfAsset) -> io::Result<PreparedScene> {
    let scene = asset
        .document
        .default_scene()
        .or_else(|| asset.document.scenes().next())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Asset has no scene"))?;
    let mut primitives = Vec::new();
    for node in scene.nodes() {
        prepare_node(asset, node, Mat4::IDENTITY, &mut primitives)?;
    }
    Ok(PreparedScene { primitives })
}

fn prepare_node(
    asset: &GltfAsset,
    node: gltf::Node<'_>,
    parent_model: Mat4,
    output: &mut Vec<PreparedPrimitive>,
) -> io::Result<()> {
    let model = parent_model * Mat4::from_cols_array_2d(&node.transform().matrix());
    if let Some(mesh) = node.mesh() {
        if node.skin().is_some() {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "Skinned meshes are not supported yet",
            ));
        }
        let linear = Mat3::from_mat4(model);
        let determinant = linear.determinant();
        if !linear.is_finite() || determinant == 0.0 || !determinant.is_finite() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Node {}: Normal transform requires an invertible model matrix",
                    node.index()
                ),
            ));
        }
        let normal_matrix = linear.inverse().transpose();
        if !normal_matrix.is_finite() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Node {}: Non-finite normal matrix", node.index()),
            ));
        }
        for primitive in mesh.primitives() {
            let invalid = |message: &str| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "Node {}, mesh {}, primitive {}: {message}",
                        node.index(),
                        mesh.index(),
                        primitive.index()
                    ),
                )
            };
            if primitive.mode() != gltf::mesh::Mode::Triangles {
                return Err(invalid("Only triangle-list primitives are supported"));
            }
            if primitive.morph_targets().len() != 0 {
                return Err(invalid("Morph targets are not supported yet"));
            }
            let reader = primitive
                .reader(|buffer| asset.buffers.get(buffer.index()).map(|data| &data.0[..]));
            let positions: Vec<Vec3> = reader
                .read_positions()
                .ok_or_else(|| invalid("Cannot read vertex positions"))?
                .map(Vec3::from_array)
                .collect();
            if positions.iter().any(|position| !position.is_finite()) {
                return Err(invalid("Non-finite vertex position"));
            }
            let material = primitive.material();
            let pbr = material.pbr_metallic_roughness();
            let mut tex_coords = [
                pbr.base_color_texture().map(|info| info.tex_coord()),
                material.normal_texture().map(|info| info.tex_coord()),
                pbr.metallic_roughness_texture()
                    .map(|info| info.tex_coord()),
                material.emissive_texture().map(|info| info.tex_coord()),
            ]
            .into_iter()
            .flatten();
            let tex_coord = tex_coords.next();
            if tex_coords.any(|set| Some(set) != tex_coord) {
                return Err(invalid(
                    "Material textures must share one texture coordinate set",
                ));
            }
            let uvs: Vec<Vec2> = match (reader.read_tex_coords(tex_coord.unwrap_or(0)), tex_coord) {
                (Some(uvs), _) => uvs.into_f32().map(Vec2::from).collect(),
                (None, Some(set)) => {
                    return Err(invalid(&format!(
                        "Missing TEXCOORD_{set} for material texture"
                    )));
                }
                (None, None) => vec![Vec2::ZERO; positions.len()],
            };
            if uvs.len() != positions.len() || uvs.iter().any(|uv| !uv.is_finite()) {
                return Err(invalid("Invalid texture coordinates"));
            }
            let indices: Vec<u32> = if primitive.indices().is_some() {
                reader
                    .read_indices()
                    .ok_or_else(|| invalid("Cannot read triangle indices"))?
                    .into_u32()
                    .collect()
            } else {
                let count =
                    u32::try_from(positions.len()).map_err(|_| invalid("Too many vertices"))?;
                (0..count).collect()
            };
            if !indices.len().is_multiple_of(3) {
                return Err(invalid("Triangle index count must be a multiple of three"));
            }
            if indices
                .iter()
                .any(|&index| index as usize >= positions.len())
            {
                return Err(invalid("Triangle index is outside the vertex buffer"));
            }
            let mut triangles: Vec<[u32; 3]> = indices
                .chunks_exact(3)
                .map(|triangle| [triangle[0], triangle[1], triangle[2]])
                .collect();
            let normals: Option<Vec<Vec3>> = reader
                .read_normals()
                .map(|normals| normals.map(Vec3::from_array).collect());
            if normals
                .as_ref()
                .is_some_and(|normals| normals.len() != positions.len())
            {
                return Err(invalid("Normal count does not match vertex count"));
            }
            if normals.is_none() && primitive.get(&gltf::Semantic::Normals).is_some() {
                return Err(invalid("Cannot read vertex normals"));
            }
            // Unusable supplied normals fall back to flat normals.
            let normals = normals.filter(|normals| {
                normals
                    .iter()
                    .all(|normal| normal.try_normalize().is_some())
            });
            let tangents = if material.normal_texture().is_some() {
                let supplied = normals.as_ref().and_then(|_| reader.read_tangents());
                Some(if let Some(supplied) = supplied {
                    let tangents: Vec<Vec4> = supplied.map(Vec4::from_array).collect();
                    supplied_tangents(&tangents, positions.len(), &triangles).unwrap_or_else(|| {
                        generate_tangents(&positions, normals.as_deref(), &uvs, &triangles)
                    })
                } else {
                    if normals.is_some() && primitive.get(&gltf::Semantic::Tangents).is_some() {
                        return Err(invalid("Cannot read vertex tangents"));
                    }
                    generate_tangents(&positions, normals.as_deref(), &uvs, &triangles)
                })
            } else {
                None
            };
            let mut vertices: Vec<PreparedVertex> = positions
                .iter()
                .enumerate()
                .map(|(i, position)| PreparedVertex {
                    position: *position,
                    uv: uvs[i],
                    normal: Vec3::ZERO,
                    tangent: Vec4::ZERO,
                })
                .collect();
            let flat_shaded = normals.is_none();
            if let Some(normals) = normals {
                for (vertex, normal) in vertices.iter_mut().zip(normals) {
                    vertex.normal = normal;
                }
            } else {
                u32::try_from(indices.len())
                    .map_err(|_| invalid("Too many flat-shaded vertices"))?;
                let mut flat_vertices = Vec::with_capacity(indices.len());
                for triangle in &mut triangles {
                    let [a, b, c] = triangle.map(|index| positions[index as usize]);
                    let normal = (b - a).cross(c - a);
                    for index in triangle {
                        let mut vertex = vertices[*index as usize];
                        vertex.normal = normal;
                        *index = flat_vertices.len() as u32;
                        flat_vertices.push(vertex);
                    }
                }
                vertices = flat_vertices;
            }
            if let Some(tangents) = tangents {
                split_tangents(&mut vertices, &mut triangles, &tangents)
                    .map_err(|error| invalid(&error.to_string()))?;
            }
            output.push(PreparedPrimitive {
                source: PrimitiveSource {
                    node_index: node.index(),
                    mesh_index: mesh.index(),
                    primitive_index: primitive.index(),
                    material_index: primitive.material().index(),
                },
                model,
                normal_matrix,
                mirrored: determinant < 0.0,
                flat_shaded,
                vertices,
                triangles,
            });
        }
    }
    for child in node.children() {
        prepare_node(asset, child, model, output)?;
    }
    Ok(())
}

/// None when tangents are invalid or a triangle mixes handedness.
fn supplied_tangents(
    tangents: &[Vec4],
    vertex_count: usize,
    triangles: &[[u32; 3]],
) -> Option<Vec<[Vec4; 3]>> {
    if tangents.len() != vertex_count
        || tangents
            .iter()
            .any(|t| !t.is_finite() || t.w.abs() != 1.0 || t.truncate().try_normalize().is_none())
    {
        return None;
    }
    triangles
        .iter()
        .map(|triangle| {
            let [a, b, c] = triangle.map(|i| tangents[i as usize]);
            (a.w == b.w && a.w == c.w).then_some([a, b, c])
        })
        .collect()
}

struct TangentGeometry<'a> {
    positions: &'a [Vec3],
    normals: Option<&'a [Vec3]>,
    uvs: &'a [Vec2],
    triangles: &'a [[u32; 3]],
    tangents: Vec<[Vec4; 3]>,
}

impl mikktspace::Geometry for TangentGeometry<'_> {
    fn num_faces(&self) -> usize {
        self.triangles.len()
    }

    fn num_vertices_of_face(&self, _: usize) -> usize {
        3
    }

    fn position(&self, face: usize, vert: usize) -> [f32; 3] {
        self.positions[self.triangles[face][vert] as usize].to_array()
    }

    fn normal(&self, face: usize, vert: usize) -> [f32; 3] {
        let triangle = &self.triangles[face];
        let normal = if let Some(normals) = self.normals {
            normals[triangle[vert] as usize]
        } else {
            let [a, b, c] = triangle.map(|i| self.positions[i as usize]);
            (b - a).cross(c - a)
        };
        normal.normalize_or_zero().to_array()
    }

    fn tex_coord(&self, face: usize, vert: usize) -> [f32; 2] {
        self.uvs[self.triangles[face][vert] as usize].to_array()
    }

    fn set_tangent_encoded(&mut self, tangent: [f32; 4], face: usize, vert: usize) {
        self.tangents[face][vert] = Vec4::from_array(tangent);
    }
}

fn generate_tangents(
    positions: &[Vec3],
    normals: Option<&[Vec3]>,
    uvs: &[Vec2],
    triangles: &[[u32; 3]],
) -> Vec<[Vec4; 3]> {
    let mut geometry = TangentGeometry {
        positions,
        normals,
        uvs,
        triangles,
        tangents: vec![[Vec4::ZERO; 3]; triangles.len()],
    };
    let has_surface = triangles.iter().any(|triangle| {
        let [a, b, c] = triangle.map(|i| positions[i as usize]);
        (b - a).cross(c - a).try_normalize().is_some()
    });
    if has_surface && !mikktspace::generate_tangents(&mut geometry) {
        geometry.tangents.fill([Vec4::ZERO; 3]);
    }
    geometry.tangents
}

fn split_tangents(
    vertices: &mut Vec<PreparedVertex>,
    triangles: &mut [[u32; 3]],
    tangents: &[[Vec4; 3]],
) -> io::Result<()> {
    let mut assigned = vec![false; vertices.len()];
    let mut variants = HashMap::new();
    for (triangle, tangents) in triangles.iter_mut().zip(tangents) {
        for (index, tangent) in triangle.iter_mut().zip(tangents) {
            let tangent = *tangent;
            let original = *index;
            if !assigned[original as usize] {
                vertices[original as usize].tangent = tangent;
                assigned[original as usize] = true;
            } else if vertices[original as usize].tangent != tangent {
                // Split tangent discontinuities at UV seams.
                let key = (original, tangent.to_array().map(f32::to_bits));
                *index = if let Some(&index) = variants.get(&key) {
                    index
                } else {
                    let index = u32::try_from(vertices.len()).map_err(|_| {
                        io::Error::new(io::ErrorKind::InvalidData, "Too many tangent vertices")
                    })?;
                    let mut vertex = vertices[original as usize];
                    vertex.tangent = tangent;
                    vertices.push(vertex);
                    variants.insert(key, index);
                    index
                };
            }
        }
    }
    Ok(())
}

impl PreparedScene {
    pub fn world_positions(&self) -> io::Result<Vec<Vec3>> {
        self.primitives
            .iter()
            .flat_map(|p| {
                p.vertices
                    .iter()
                    .map(|v| p.model.transform_point3(v.position))
            })
            .map(|p| {
                if p.is_finite() {
                    Ok(p)
                } else {
                    Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Non-finite world-space vertex position",
                    ))
                }
            })
            .collect()
    }
}
