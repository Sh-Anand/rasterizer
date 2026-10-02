use std::io;

use glam::{Mat3, Mat4, Vec2, Vec3, Vec4, camera::rh};

use crate::{
    asset::GltfAsset,
    config::Config,
    geometry::{PrimitiveSource, Triangle},
};

#[derive(Debug)]
pub struct ClipPrimitive {
    pub source: PrimitiveSource,
    pub vertices: Vec<ClipVertex>,
    pub triangles: Vec<Triangle>,
}

#[derive(Debug, Clone, Copy)]
pub struct ClipVertex {
    pub position: Vec4,
    pub uv: Vec2,
    /// World-space normal.
    pub normal: Vec3,
}

#[derive(Debug)]
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
        if !linear.is_finite() || linear.determinant() == 0.0 {
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
            let base_color_texture = primitive
                .material()
                .pbr_metallic_roughness()
                .base_color_texture();
            let tex_coord = base_color_texture
                .as_ref()
                .map_or(0, |texture| texture.tex_coord());
            let uvs: Vec<Vec2> = match reader.read_tex_coords(tex_coord) {
                Some(uvs) => uvs.into_f32().map(Vec2::from).collect(),
                None if base_color_texture.is_some() => {
                    return Err(invalid(
                        "Missing texture coordinates for base-color texture",
                    ));
                }
                None => vec![Vec2::ZERO; positions.len()],
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
            let mut triangles: Vec<Triangle> = indices
                .chunks_exact(3)
                .enumerate()
                .map(|(source_index, triangle)| Triangle {
                    indices: [triangle[0], triangle[1], triangle[2]],
                    source_index,
                })
                .collect();
            let mut vertices: Vec<ClipVertex> = positions
                .iter()
                .zip(uvs)
                .map(|(position, uv)| ClipVertex {
                    position: mvp * position.extend(1.0),
                    uv,
                    normal: Vec3::ZERO,
                })
                .collect();
            if vertices.iter().any(|vertex| !vertex.position.is_finite()) {
                return Err(invalid("Non-finite clip-space position"));
            }
            if let Some(normals) = reader.read_normals() {
                if normals.len() != vertices.len() {
                    return Err(invalid("Normal count does not match vertex count"));
                }
                for (vertex, normal) in vertices.iter_mut().zip(normals) {
                    vertex.normal = (normal_matrix * Vec3::from_array(normal))
                        .try_normalize()
                        .ok_or_else(|| invalid("Invalid vertex normal"))?;
                }
            } else {
                if primitive.get(&gltf::Semantic::Normals).is_some() {
                    return Err(invalid("Cannot read vertex normals"));
                }
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
            output.push(ClipPrimitive {
                source: PrimitiveSource {
                    node_index: node.index(),
                    mesh_index: mesh.index(),
                    primitive_index: primitive.index(),
                    material_index: primitive.material().index(),
                },
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
