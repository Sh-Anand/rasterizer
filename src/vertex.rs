use std::io;

use glam::{Mat4, Vec3, Vec4, camera::rh};

use crate::{
    asset::GltfAsset,
    config::Config,
    geometry::{PrimitiveSource, Triangle},
};

#[derive(Debug)]
pub struct ClipPrimitive {
    pub source: PrimitiveSource,
    pub positions: Vec<Vec4>,
    pub triangles: Vec<Triangle>,
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
            let positions: Vec<_> = reader
                .read_positions()
                .ok_or_else(|| invalid("Cannot read vertex positions"))?
                .map(|position| mvp * Vec3::from_array(position).extend(1.0))
                .collect();
            if positions.iter().any(|position| !position.is_finite()) {
                return Err(invalid("Non-finite clip-space position"));
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
            let triangles = indices
                .chunks_exact(3)
                .enumerate()
                .map(|(source_index, triangle)| Triangle {
                    indices: [triangle[0], triangle[1], triangle[2]],
                    source_index,
                })
                .collect();
            output.push(ClipPrimitive {
                source: PrimitiveSource {
                    node_index: node.index(),
                    mesh_index: mesh.index(),
                    primitive_index: primitive.index(),
                    material_index: primitive.material().index(),
                },
                positions,
                triangles,
            });
        }
    }
    for child in node.children() {
        transform_node(asset, child, model, view_projection, output)?;
    }
    Ok(())
}
