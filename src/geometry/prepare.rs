use std::io;

use glam::{Mat3, Mat4, Vec2, Vec3, Vec4};

use super::{PreparedPrimitive, PreparedScene, PreparedVertex, PrimitiveSource, Triangle, tangent};
use crate::asset::GltfAsset;

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
    Ok(PreparedScene {
        scene_index: scene.index(),
        primitives,
    })
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
            let read_uvs = |tex_coord: Option<u32>| -> io::Result<Vec<Vec2>> {
                let required = tex_coord.is_some();
                let tex_coord = tex_coord.unwrap_or(0);
                let uvs: Vec<Vec2> = match reader.read_tex_coords(tex_coord) {
                    Some(uvs) => uvs.into_f32().map(Vec2::from).collect(),
                    None if required => {
                        return Err(invalid(&format!(
                            "Missing TEXCOORD_{tex_coord} for material texture"
                        )));
                    }
                    None => vec![Vec2::ZERO; positions.len()],
                };
                if uvs.len() != positions.len() || uvs.iter().any(|uv| !uv.is_finite()) {
                    return Err(invalid("Invalid texture coordinates"));
                }
                Ok(uvs)
            };
            let material = primitive.material();
            let pbr = material.pbr_metallic_roughness();
            let uvs = read_uvs(pbr.base_color_texture().map(|info| info.tex_coord()))?;
            let normal_uvs = read_uvs(material.normal_texture().map(|info| info.tex_coord()))?;
            let metallic_roughness_uvs = read_uvs(
                pbr.metallic_roughness_texture()
                    .map(|info| info.tex_coord()),
            )?;
            let occlusion_uvs =
                read_uvs(material.occlusion_texture().map(|info| info.tex_coord()))?;
            let emissive_uvs = read_uvs(material.emissive_texture().map(|info| info.tex_coord()))?;
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
            let mut triangles: Vec<Triangle> = indices
                .chunks_exact(3)
                .enumerate()
                .map(|(source_index, triangle)| Triangle {
                    indices: [triangle[0], triangle[1], triangle[2]],
                    source_index,
                })
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
            if normals.as_ref().is_some_and(|normals| {
                normals
                    .iter()
                    .any(|normal| normal.try_normalize().is_none())
            }) {
                return Err(invalid("Invalid vertex normal"));
            }
            let tangents = if material.normal_texture().is_some() {
                let supplied = normals.as_ref().and_then(|_| reader.read_tangents());
                Some(if let Some(supplied) = supplied {
                    let tangents: Vec<Vec4> = supplied.map(Vec4::from_array).collect();
                    tangent::from_vertices(&tangents, positions.len(), &triangles)
                        .map_err(|error| invalid(&error.to_string()))?
                } else {
                    if normals.is_some() && primitive.get(&gltf::Semantic::Tangents).is_some() {
                        return Err(invalid("Cannot read vertex tangents"));
                    }
                    tangent::generate(&positions, normals.as_deref(), &normal_uvs, &triangles)
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
                    normal_uv: normal_uvs[i],
                    metallic_roughness_uv: metallic_roughness_uvs[i],
                    occlusion_uv: occlusion_uvs[i],
                    emissive_uv: emissive_uvs[i],
                    lightmap_uv: Vec2::ZERO,
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
                    let [a, b, c] = triangle.indices.map(|index| positions[index as usize]);
                    let normal = (b - a).cross(c - a);
                    for index in &mut triangle.indices {
                        let mut vertex = vertices[*index as usize];
                        vertex.normal = normal;
                        *index = flat_vertices.len() as u32;
                        flat_vertices.push(vertex);
                    }
                }
                vertices = flat_vertices;
            }
            if let Some(tangents) = tangents {
                tangent::split(&mut vertices, &mut triangles, &tangents)
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
