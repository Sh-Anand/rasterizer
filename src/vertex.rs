use std::io;

use glam::{Mat3, Mat4, Vec2, Vec3, Vec4, camera::rh};

mod tangent;

use crate::{
    asset::GltfAsset,
    config::Config,
    geometry::{PrimitiveSource, Triangle},
};

#[derive(Debug, Clone)]
pub struct ClipPrimitive {
    pub source: PrimitiveSource,
    pub mirrored: bool,
    pub vertices: Vec<ClipVertex>,
    pub triangles: Vec<Triangle>,
}

#[derive(Debug, Clone, Copy)]
pub struct ClipVertex {
    pub position: Vec4,
    pub world_position: Vec3,
    pub uv: Vec2,
    pub normal_uv: Vec2,
    pub metallic_roughness_uv: Vec2,
    pub occlusion_uv: Vec2,
    pub emissive_uv: Vec2,
    pub lightmap_uv: Vec2,
    /// World-space normal.
    pub normal: Vec3,
    /// World-space tangent; W is the bitangent sign (zero for an unusable frame).
    pub tangent: Vec4,
}

#[derive(Debug, Clone)]
pub struct ClipScene {
    pub scene_index: usize,
    pub primitives: Vec<ClipPrimitive>,
}

pub fn transform_scene(asset: &GltfAsset, config: &Config) -> io::Result<ClipScene> {
    let view_projection = camera_matrix(config)?;
    let scene = asset
        .document
        .default_scene()
        .or_else(|| asset.document.scenes().next())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Asset has no scene"))?;
    let mut primitives = Vec::new();
    for node in scene.nodes() {
        transform_node(
            asset,
            node,
            Mat4::IDENTITY,
            view_projection,
            &mut primitives,
        )?;
    }
    Ok(ClipScene {
        scene_index: scene.index(),
        primitives,
    })
}

pub fn reproject_scene(scene: &ClipScene, view_projection: Mat4) -> io::Result<ClipScene> {
    if !view_projection.is_finite() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Non-finite projection matrix",
        ));
    }
    let mut scene = scene.clone();
    for primitive in &mut scene.primitives {
        for vertex in &mut primitive.vertices {
            vertex.position = view_projection * vertex.world_position.extend(1.0);
            if !vertex.position.is_finite() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Non-finite reprojected position",
                ));
            }
        }
    }
    Ok(scene)
}

fn camera_matrix(config: &Config) -> io::Result<Mat4> {
    let camera = &config.camera;
    if config.width == 0 || config.height == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Image width and height must be nonzero",
        ));
    }
    if !camera.fov_y_degrees.is_finite()
        || camera.fov_y_degrees <= 0.0
        || camera.fov_y_degrees >= 180.0
        || !camera.near.is_finite()
        || !camera.far.is_finite()
        || camera.near <= 0.0
        || camera.far <= camera.near
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Camera needs 0 < fov_y_degrees < 180 and finite 0 < near < far",
        ));
    }

    let position = Vec3::from_array(camera.position);
    let target = Vec3::from_array(camera.target);
    let forward = (target - position).try_normalize();
    let up = Vec3::from_array(camera.up).try_normalize();
    let (Some(forward), Some(up)) = (forward, up) else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Camera needs a finite, nonzero viewing direction and up vector",
        ));
    };
    if !position.is_finite() || forward.cross(up).length_squared() < 1e-12 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Camera position must be finite and up must not parallel the viewing direction",
        ));
    }

    let view = rh::view::look_to_mat4(position, forward, up);
    // Camera looks down -Z; clipping bounds are -w..w on all three axes.
    let projection = rh::proj::opengl::perspective(
        camera.fov_y_degrees.to_radians(),
        config.width as f32 / config.height as f32,
        camera.near,
        camera.far,
    );
    let view_projection = projection * view;
    if !view_projection.is_finite() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Camera settings produce a non-finite matrix",
        ));
    }
    Ok(view_projection)
}

fn transform_node(
    asset: &GltfAsset,
    node: gltf::Node<'_>,
    parent_model: Mat4,
    view_projection: Mat4,
    output: &mut Vec<ClipPrimitive>,
) -> io::Result<()> {
    let model = parent_model * Mat4::from_cols_array_2d(&node.transform().matrix());
    if let Some(mesh) = node.mesh() {
        if node.skin().is_some() {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "Skinned meshes are not supported yet",
            ));
        }
        let mvp = view_projection * model;
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
            let mut vertices: Vec<ClipVertex> = positions
                .iter()
                .enumerate()
                .map(|(i, position)| ClipVertex {
                    position: mvp * position.extend(1.0),
                    world_position: model.transform_point3(*position),
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
            if vertices
                .iter()
                .any(|vertex| !vertex.position.is_finite() || !vertex.world_position.is_finite())
            {
                return Err(invalid("Non-finite vertex position"));
            }
            if let Some(normals) = normals {
                for (vertex, normal) in vertices.iter_mut().zip(normals) {
                    vertex.normal = (normal_matrix * normal)
                        .try_normalize()
                        .ok_or_else(|| invalid("Invalid vertex normal"))?;
                }
            } else {
                u32::try_from(indices.len())
                    .map_err(|_| invalid("Too many flat-shaded vertices"))?;
                let mut flat_vertices = Vec::with_capacity(indices.len());
                for triangle in &mut triangles {
                    let [a, b, c] = triangle.indices.map(|index| positions[index as usize]);
                    let normal = (normal_matrix * (b - a).cross(c - a)).normalize_or_zero();
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
                tangent::apply(&mut vertices, &mut triangles, &tangents, linear)
                    .map_err(|error| invalid(&error.to_string()))?;
            }
            output.push(ClipPrimitive {
                source: PrimitiveSource {
                    node_index: node.index(),
                    mesh_index: mesh.index(),
                    primitive_index: primitive.index(),
                    material_index: primitive.material().index(),
                },
                mirrored: determinant < 0.0,
                vertices,
                triangles,
            });
        }
    }
    for child in node.children() {
        transform_node(asset, child, model, view_projection, output)?;
    }
    Ok(())
}
