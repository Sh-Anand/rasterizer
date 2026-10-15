use glam::{Mat4, Vec2, Vec3, Vec4};

use rasterizer::{
    cpu::CpuVertex,
    geometry::{GltfAsset, prepare_scene},
    pipeline::VertexStage,
};

const QUAD: [[f32; 3]; 4] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
];

// Mirrored UVs and nonuniform scale under a translated parent.
fn asset(positions: [[f32; 3]; 4], indices: [u32; 6], normals: bool, tangents: bool) -> GltfAsset {
    let mut buffer = Vec::new();
    for value in positions
        .into_iter()
        .flatten()
        .chain([[0.0, 0.0, 1.0]; 4].into_iter().flatten())
        .chain(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
                .into_iter()
                .flatten(),
        )
        .chain([[2.0, 0.0, 0.0, 1.0]; 4].into_iter().flatten())
    {
        buffer.extend_from_slice(&value.to_le_bytes());
    }
    for index in indices {
        buffer.extend_from_slice(&index.to_le_bytes());
    }
    let normal_attribute = if normals { ", \"NORMAL\": 1" } else { "" };
    let tangent_attribute = if tangents { ", \"TANGENT\": 3" } else { "" };
    let json = format!(
        r#"{{
            "asset": {{"version": "2.0"}},
            "buffers": [{{"byteLength": 216}}],
            "bufferViews": [
                {{"buffer": 0, "byteOffset": 0, "byteLength": 48}},
                {{"buffer": 0, "byteOffset": 48, "byteLength": 48}},
                {{"buffer": 0, "byteOffset": 96, "byteLength": 32}},
                {{"buffer": 0, "byteOffset": 128, "byteLength": 64}},
                {{"buffer": 0, "byteOffset": 192, "byteLength": 24}}
            ],
            "accessors": [
                {{"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3", "min": [0, 0, 0], "max": [1, 1, 1]}},
                {{"bufferView": 1, "componentType": 5126, "count": 4, "type": "VEC3"}},
                {{"bufferView": 2, "componentType": 5126, "count": 4, "type": "VEC2"}},
                {{"bufferView": 3, "componentType": 5126, "count": 4, "type": "VEC4"}},
                {{"bufferView": 4, "componentType": 5125, "count": 6, "type": "SCALAR"}}
            ],
            "images": [{{"uri": "unused.png"}}],
            "textures": [{{"source": 0}}],
            "materials": [{{"normalTexture": {{"index": 0}}}}],
            "meshes": [{{"primitives": [{{
                "attributes": {{"POSITION": 0, "TEXCOORD_0": 2{normal_attribute}{tangent_attribute}}},
                "indices": 4, "material": 0
            }}]}}],
            "nodes": [
                {{"translation": [5, 1, 2], "children": [1]}},
                {{"mesh": 0, "scale": [-2, 3, 0.5]}}
            ],
            "scenes": [{{"nodes": [0]}}],
            "scene": 0
        }}"#
    );
    let document = gltf::Gltf::from_slice(json.as_bytes()).unwrap().document;
    GltfAsset {
        document,
        buffers: vec![gltf::buffer::Data(buffer)],
        images: Vec::new(),
    }
}

#[test]
fn flat_normals_are_split_before_transforming() {
    let asset = asset(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        [0, 1, 2, 0, 3, 1],
        false,
        false,
    );
    let prepared = prepare_scene(&asset).unwrap();
    let primitive = &prepared.primitives[0];
    assert!(primitive.flat_shaded);
    assert!(primitive.mirrored);
    assert_eq!(primitive.source.node_index, 1);
    assert_eq!(primitive.source.material_index, Some(0));
    assert_eq!(primitive.vertices.len(), 6);
    assert_eq!(primitive.triangles[0].indices, [0, 1, 2]);
    assert_eq!(primitive.triangles[1].indices, [3, 4, 5]);
    assert_eq!(primitive.vertices[1].position, Vec3::X);
    assert_eq!(primitive.vertices[0].normal, Vec3::Z);
    assert_eq!(primitive.vertices[3].normal, Vec3::Y);

    let transformed = CpuVertex.transform(&prepared, Mat4::IDENTITY).unwrap();
    let vertices = &transformed.primitives[0].vertices;
    assert_eq!(vertices[0].world_position, Vec3::new(5.0, 1.0, 2.0));
    assert_eq!(vertices[1].world_position, Vec3::new(3.0, 1.0, 2.0));
    assert_eq!(vertices[0].normal, Vec3::Z);
    assert_eq!(vertices[3].normal, Vec3::Y);
}

#[test]
fn generated_tangent_seams_are_fixed_and_reusable_across_views() {
    let asset = asset(QUAD, [0, 1, 2, 0, 2, 3], true, false);
    let prepared = prepare_scene(&asset).unwrap();
    drop(asset);
    let primitive = &prepared.primitives[0];
    assert_eq!(primitive.vertices.len(), 6);
    let first = primitive.triangles[0].indices[0] as usize;
    let second = primitive.triangles[1].indices[0] as usize;
    assert_ne!(first, second);
    assert_eq!(
        primitive.vertices[first].position,
        primitive.vertices[second].position
    );
    assert_eq!(
        primitive.vertices[first].tangent.w,
        -primitive.vertices[second].tangent.w
    );

    let before = format!("{prepared:?}");
    let first_view = CpuVertex.transform(&prepared, Mat4::IDENTITY).unwrap();
    let second_view = CpuVertex
        .transform(&prepared, Mat4::from_translation(Vec3::X))
        .unwrap();
    assert_eq!(before, format!("{prepared:?}"));
    for output in [&first_view, &second_view] {
        assert_eq!(output.primitives[0].triangles, primitive.triangles);
        assert_eq!(
            output.primitives[0].vertices.len(),
            primitive.vertices.len()
        );
        for (input, vertex) in primitive
            .vertices
            .iter()
            .zip(&output.primitives[0].vertices)
        {
            assert_eq!(vertex.tangent.w, -input.tangent.w);
        }
    }
    for (a, b) in first_view.primitives[0]
        .vertices
        .iter()
        .zip(&second_view.primitives[0].vertices)
    {
        assert_eq!(a.world_position, b.world_position);
        assert_eq!(a.normal, b.normal);
        assert_eq!(a.tangent, b.tangent);
        assert_eq!(a.position + Vec4::X, b.position);
    }
}

#[test]
fn supplied_frames_transform_in_world_space_and_preserve_uvs() {
    let asset = asset(QUAD, [0, 1, 2, 0, 2, 3], true, true);
    let mut prepared = prepare_scene(&asset).unwrap();
    let vertex = &mut prepared.primitives[0].vertices[0];
    // An oblique normal distinguishes the inverse-transpose from the model matrix.
    vertex.normal = Vec3::new(1.0, 1.0, 0.0);
    vertex.uv = Vec2::new(0.1, 0.2);
    vertex.normal_uv = Vec2::new(0.3, 0.4);
    vertex.metallic_roughness_uv = Vec2::new(0.5, 0.6);
    vertex.occlusion_uv = Vec2::new(0.7, 0.8);
    vertex.emissive_uv = Vec2::new(0.9, 1.0);
    vertex.lightmap_uv = Vec2::new(0.2, 0.8);
    let input = *vertex;
    let transformed = CpuVertex.transform(&prepared, Mat4::IDENTITY).unwrap();
    let output = &transformed.primitives[0].vertices[0];
    assert_eq!(output.normal, Vec3::new(-0.5, 1.0 / 3.0, 0.0).normalize());
    assert_eq!(output.tangent, Vec4::new(-1.0, 0.0, 0.0, -1.0));
    assert_eq!(output.uv, input.uv);
    assert_eq!(output.normal_uv, input.normal_uv);
    assert_eq!(output.metallic_roughness_uv, input.metallic_roughness_uv);
    assert_eq!(output.occlusion_uv, input.occlusion_uv);
    assert_eq!(output.emissive_uv, input.emissive_uv);
    assert_eq!(output.lightmap_uv, input.lightmap_uv);
}

#[test]
fn degenerate_flat_geometry_keeps_zero_frames_and_nonfinite_views_are_rejected() {
    let asset = asset(QUAD, [0, 0, 0, 1, 1, 1], false, false);
    let prepared = prepare_scene(&asset).unwrap();
    let transformed = CpuVertex.transform(&prepared, Mat4::IDENTITY).unwrap();
    for vertex in &transformed.primitives[0].vertices {
        assert_eq!(vertex.normal, Vec3::ZERO);
        assert_eq!(vertex.tangent, Vec4::ZERO);
    }
    let error = CpuVertex
        .transform(&prepared, Mat4::from_cols_array(&[f32::NAN; 16]))
        .unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
}
