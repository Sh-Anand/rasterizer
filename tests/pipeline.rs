use std::{collections::HashSet, io};

use glam::{Mat4, Vec2, Vec3, Vec4};

use rasterizer::{
    cpu::{self, CpuFragment, CpuRasterization, CpuVertex},
    framebuffer::{DepthBuffer, RenderOutput},
    geometry::{
        ClipPrimitive, ClipScene, ClipVertex, GltfAsset, PreparedScene, PrimitiveSource, Triangle,
    },
    lighting::Lighting,
    material::{Material, load_materials},
    pipeline::{
        Fragment, FragmentAttributes, FragmentStage, Pipeline, RasterizationStage, RasterizedBatch,
        VertexStage,
    },
    texture::{Sampling, UvSample},
};

fn triangle(material_index: Option<usize>, z: f32) -> ClipPrimitive {
    ClipPrimitive {
        source: PrimitiveSource {
            node_index: 0,
            mesh_index: 0,
            primitive_index: 0,
            material_index,
        },
        mirrored: false,
        vertices: [
            Vec3::new(-0.9, -0.9, z),
            Vec3::new(0.9, -0.9, z),
            Vec3::new(0.0, 0.9, z),
        ]
        .map(|position| ClipVertex {
            position: position.extend(1.0),
            world_position: position,
            uv: Vec2::ZERO,
            normal_uv: Vec2::ZERO,
            metallic_roughness_uv: Vec2::ZERO,
            occlusion_uv: Vec2::ZERO,
            emissive_uv: Vec2::ZERO,
            lightmap_uv: Vec2::ZERO,
            normal: Vec3::Z,
            tangent: Vec4::ZERO,
        })
        .to_vec(),
        triangles: vec![Triangle {
            indices: [0, 1, 2],
            source_index: 0,
        }],
    }
}

fn scene(primitives: Vec<ClipPrimitive>) -> ClipScene {
    ClipScene {
        scene_index: 0,
        primitives,
    }
}

fn material(color: Vec3) -> Material {
    Material {
        emissive_factor: color,
        ..Material::default()
    }
}

fn lighting() -> Lighting<'static> {
    Lighting {
        indirect: None,
        lights: &[],
        camera_position: Vec3::Z,
    }
}

fn assert_same(actual: &RenderOutput, expected: &RenderOutput) {
    assert_eq!(actual.covered_fragments, expected.covered_fragments);
    assert_eq!(actual.screen_vertices, expected.screen_vertices);
    assert_eq!(actual.screen_triangles, expected.screen_triangles);
    let color_bits = |output: &RenderOutput| {
        output
            .framebuffer
            .color
            .iter()
            .map(|color| color.to_array().map(f32::to_bits))
            .collect::<Vec<_>>()
    };
    let depth_bits = |output: &RenderOutput| {
        output
            .framebuffer
            .depth
            .values
            .iter()
            .map(|depth| depth.to_bits())
            .collect::<Vec<_>>()
    };
    assert_eq!(color_bits(actual), color_bits(expected));
    assert_eq!(depth_bits(actual), depth_bits(expected));
}

fn reference(scene: &ClipScene, materials: &[Material], size: u32) -> RenderOutput {
    let mut scene = scene.clone();
    scene.primitives.sort_by_key(|p| p.source.material_index);
    cpu::render_reference(&scene, materials, size, size, lighting()).unwrap()
}

#[derive(Default)]
struct RecordingFragment {
    materials: Vec<Vec3>,
    sample_counts: Vec<usize>,
}

impl FragmentStage for &mut RecordingFragment {
    fn shade(
        &mut self,
        fragments: &[Fragment],
        material: &Material,
        lighting: &Lighting<'_>,
    ) -> io::Result<Vec<Vec3>> {
        assert!(!fragments.is_empty());
        assert_unique_samples(fragments);
        self.materials.push(material.emissive_factor);
        self.sample_counts.push(fragments.len());
        CpuFragment.shade(fragments, material, lighting)
    }
}

#[test]
fn batches_share_depth_use_material_order_and_skip_occluded_or_masked_work() {
    let scene = scene(vec![
        triangle(Some(1), 0.0),
        triangle(Some(0), 0.5),
        triangle(Some(0), 0.0),
        triangle(None, 0.75),
        triangle(Some(2), -0.5),
    ]);
    let materials = [
        material(Vec3::X),
        material(Vec3::Y),
        Material {
            alpha_cutoff: Some(0.5),
            base_color_factor: Vec4::ZERO,
            ..material(Vec3::Z)
        },
    ];
    let mut fragment = RecordingFragment::default();
    let mut pipeline = Pipeline::new(CpuVertex, CpuRasterization, &mut fragment);
    let result = pipeline
        .render_transformed(&scene, &materials, 8, 8, lighting())
        .unwrap();
    assert_eq!(fragment.materials, [Vec3::ZERO, Vec3::X]);
    assert_same(&result, &reference(&scene, &materials, 8));
    let visible_samples = result
        .framebuffer
        .depth
        .values
        .iter()
        .filter(|z| z.is_finite())
        .count();
    assert_eq!(fragment.sample_counts, [visible_samples, visible_samples]);
    assert!(visible_samples > 0);
    for (color, depth) in result
        .framebuffer
        .color
        .iter()
        .zip(&result.framebuffer.depth.values)
    {
        if depth.is_finite() {
            assert_eq!(*color, Vec3::X);
            assert!((*depth - 0.5).abs() < 1e-6);
        }
    }
}

fn assert_unique_samples(fragments: &[Fragment]) {
    let samples: HashSet<_> = fragments
        .iter()
        .map(|fragment| fragment.sample_index)
        .collect();
    assert_eq!(samples.len(), fragments.len());
}

fn collect_batch(
    primitives: &[ClipPrimitive],
    material: &Material,
) -> (RasterizedBatch, DepthBuffer) {
    let mut depth = DepthBuffer::new(16, 16).unwrap();
    let batch = CpuRasterization
        .rasterize(&primitives.iter().collect::<Vec<_>>(), material, &mut depth)
        .unwrap();
    assert_unique_samples(&batch.fragments);
    assert_eq!(
        batch.fragments.len(),
        depth.values.iter().filter(|z| z.is_finite()).count()
    );
    (batch, depth)
}

#[test]
fn nearer_fragments_replace_all_attributes_and_facing_within_and_across_primitives() {
    let far = triangle(Some(0), 0.5);
    let mut near = triangle(Some(0), -0.5);
    near.triangles[0].indices.swap(1, 2);
    for vertex in &mut near.vertices {
        vertex.uv = Vec2::new(0.25, 0.75);
        vertex.normal_uv = Vec2::ONE;
        vertex.normal = Vec3::X;
        vertex.tangent = Vec4::new(0.0, 1.0, 0.0, -1.0);
    }
    let material = Material {
        double_sided: true,
        ..Material::default()
    };
    for combine in [false, true] {
        let primitives = if combine {
            let mut combined = far.clone();
            combined.vertices.extend_from_slice(&near.vertices);
            combined.triangles.push(Triangle {
                indices: [3, 5, 4],
                source_index: 1,
            });
            vec![combined]
        } else {
            vec![far.clone(), near.clone()]
        };
        let (batch, _) = collect_batch(&primitives, &material);
        assert!(!batch.fragments.is_empty());
        assert_eq!(batch.covered_fragments, 2 * batch.fragments.len() as u64);
        for fragment in batch.fragments {
            assert!(!fragment.front_facing);
            assert_eq!(fragment.attributes.uv.uv, Vec2::new(0.25, 0.75));
            assert_eq!(fragment.attributes.normal_uv.uv, Vec2::ONE);
            assert_eq!(fragment.attributes.normal, Vec3::X);
            assert_eq!(fragment.attributes.tangent, Vec4::new(0.0, 1.0, 0.0, -1.0));
            assert!((fragment.attributes.world_position.z + 0.5).abs() < 1e-6);
        }
    }
}

#[test]
fn partial_occlusion_retains_the_uncovered_samples_of_an_earlier_triangle() {
    let far = triangle(Some(0), 0.5);
    let mut near = triangle(Some(0), -0.5);
    for vertex in &mut near.vertices {
        vertex.position.x *= 0.5;
        vertex.position.y *= 0.5;
        vertex.normal_uv = Vec2::ONE;
    }
    let material = Material::default();
    let (far_only, _) = collect_batch(std::slice::from_ref(&far), &material);
    let (near_only, _) = collect_batch(std::slice::from_ref(&near), &material);
    let (batch, _) = collect_batch(&[far, near], &material);
    let near_samples: HashSet<_> = near_only
        .fragments
        .iter()
        .map(|fragment| fragment.sample_index)
        .collect();
    assert!(!near_samples.is_empty());
    assert!(near_samples.len() < far_only.fragments.len());
    assert_eq!(batch.fragments.len(), far_only.fragments.len());
    for fragment in batch.fragments {
        let expected_tag = if near_samples.contains(&fragment.sample_index) {
            Vec2::ONE
        } else {
            Vec2::ZERO
        };
        assert_eq!(fragment.attributes.normal_uv.uv, expected_tag);
    }
}

#[test]
fn equal_depth_keeps_the_first_winner_and_collector_state_does_not_cross_batches() {
    let mut rasterizer = CpuRasterization;
    let material = Material::default();
    for (width, height) in [(8, 8), (16, 16)] {
        let mut depth = DepthBuffer::new(width, height).unwrap();
        let first = triangle(Some(0), 0.5);
        let mut equal = first.clone();
        for vertex in &mut equal.vertices {
            vertex.normal_uv = Vec2::ONE;
        }
        let batch = rasterizer
            .rasterize(&[&first, &equal], &material, &mut depth)
            .unwrap();
        assert_unique_samples(&batch.fragments);
        let samples = batch.fragments.len();
        assert!(samples > 0);
        assert!(
            batch
                .fragments
                .iter()
                .all(|fragment| fragment.attributes.normal_uv.uv == Vec2::ZERO)
        );
        let previous_depth = depth.values.clone();
        let rejected = rasterizer
            .rasterize(&[&equal], &material, &mut depth)
            .unwrap();
        assert!(rejected.fragments.is_empty());
        assert_eq!(depth.values, previous_depth);

        let closer = triangle(Some(0), -0.5);
        let next = rasterizer
            .rasterize(&[&closer], &material, &mut depth)
            .unwrap();
        assert_unique_samples(&next.fragments);
        assert_eq!(next.fragments.len(), samples);
        let rejected = rasterizer
            .rasterize(&[&first], &material, &mut depth)
            .unwrap();
        assert!(rejected.fragments.is_empty());
        assert!(
            rasterizer
                .rasterize(&[], &material, &mut depth)
                .unwrap()
                .fragments
                .is_empty()
        );
    }
}

#[test]
fn alpha_rejection_preserves_earlier_winners_inside_the_same_material_batch() {
    let document = gltf::Gltf::from_slice(
        br#"{
        "asset": {"version": "2.0"},
        "images": [{"uri": "unused.png"}],
        "textures": [{"source": 0}],
        "materials": [{"alphaMode": "MASK", "alphaCutoff": 0.5,
            "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}]
    }"#,
    )
    .unwrap()
    .document;
    let asset = GltfAsset {
        document,
        buffers: Vec::new(),
        images: vec![gltf::image::Data {
            pixels: vec![255, 255, 255, 255, 255, 255, 255, 0],
            format: gltf::image::Format::R8G8B8A8,
            width: 2,
            height: 1,
        }],
    };
    let materials = load_materials(&asset, Sampling::None).unwrap();
    let mut far = triangle(Some(0), 0.5);
    for vertex in &mut far.vertices {
        vertex.uv = Vec2::new(0.25, 0.5);
    }
    let mut cutout = triangle(Some(0), 0.0);
    for vertex in &mut cutout.vertices {
        vertex.uv = Vec2::new((vertex.position.x + 1.0) * 0.5, 0.5);
        vertex.normal_uv = Vec2::ONE;
    }
    let mut rejected = triangle(Some(0), -0.5);
    for vertex in &mut rejected.vertices {
        vertex.uv = Vec2::new(0.75, 0.5);
        vertex.normal_uv = Vec2::splat(2.0);
    }
    let (far_only, _) = collect_batch(std::slice::from_ref(&far), &materials[0]);
    let (batch, _) = collect_batch(&[far, cutout, rejected], &materials[0]);
    assert_eq!(batch.fragments.len(), far_only.fragments.len());
    assert!(
        batch
            .fragments
            .iter()
            .any(|fragment| fragment.attributes.normal_uv.uv == Vec2::ZERO)
    );
    assert!(
        batch
            .fragments
            .iter()
            .any(|fragment| fragment.attributes.normal_uv.uv == Vec2::ONE)
    );
    assert!(
        batch
            .fragments
            .iter()
            .all(|fragment| fragment.attributes.normal_uv.uv != Vec2::splat(2.0))
    );
}

#[test]
fn cpu_stages_preserve_clipping_mirrored_winding_and_double_sided_faces() {
    let mut clipped = triangle(Some(0), 0.0);
    clipped.vertices[0].position.x = -2.0;
    let mut mirrored = triangle(Some(1), -0.5);
    mirrored.mirrored = true;
    mirrored.triangles[0].indices.swap(1, 2);
    let mut back = triangle(Some(2), -0.75);
    back.triangles[0].indices.swap(1, 2);
    let scene = scene(vec![back, mirrored, clipped]);
    let materials = [
        material(Vec3::X),
        material(Vec3::Y),
        Material {
            double_sided: true,
            ..material(Vec3::Z)
        },
    ];
    let mut pipeline = Pipeline::new(CpuVertex, CpuRasterization, CpuFragment);
    let result = pipeline
        .render_transformed(&scene, &materials, 16, 16, lighting())
        .unwrap();
    assert_same(&result, &reference(&scene, &materials, 16));
    assert!(result.framebuffer.color.contains(&Vec3::Z));
}

struct FixedVertex {
    scene: ClipScene,
    called: bool,
}

impl VertexStage for &mut FixedVertex {
    fn transform(&mut self, scene: &PreparedScene, view_projection: Mat4) -> io::Result<ClipScene> {
        assert!(scene.primitives.is_empty());
        assert_eq!(view_projection, Mat4::IDENTITY);
        self.called = true;
        Ok(self.scene.clone())
    }
}

#[test]
fn vertex_backend_can_be_replaced_independently() {
    let mut vertex = FixedVertex {
        scene: scene(vec![triangle(Some(0), 0.0)]),
        called: false,
    };
    let mut pipeline = Pipeline::new(&mut vertex, CpuRasterization, CpuFragment);
    let prepared = PreparedScene {
        scene_index: 0,
        primitives: Vec::new(),
    };
    let result = pipeline
        .render(
            &prepared,
            Mat4::IDENTITY,
            &[material(Vec3::X)],
            8,
            8,
            lighting(),
        )
        .unwrap();
    assert!(vertex.called);
    assert!(result.framebuffer.color.contains(&Vec3::X));
}

struct FixedRasterization {
    sample_index: usize,
}

impl RasterizationStage for FixedRasterization {
    fn rasterize(
        &mut self,
        primitives: &[&ClipPrimitive],
        _: &Material,
        depth: &mut DepthBuffer,
    ) -> io::Result<RasterizedBatch> {
        assert_eq!(primitives.len(), 1);
        if let Some(sample) = depth.values.get_mut(self.sample_index) {
            *sample = 0.25;
        }
        let uv = UvSample::point(Vec2::ZERO);
        Ok(RasterizedBatch {
            fragments: vec![Fragment {
                sample_index: self.sample_index,
                front_facing: true,
                attributes: FragmentAttributes {
                    world_position: Vec3::ZERO,
                    normal: Vec3::Z,
                    tangent: Vec4::ZERO,
                    uv,
                    normal_uv: uv,
                    metallic_roughness_uv: uv,
                    occlusion_uv: uv,
                    emissive_uv: uv,
                    lightmap_uv: Vec2::ZERO,
                },
            }],
            covered_fragments: 1,
            ..RasterizedBatch::default()
        })
    }
}

#[test]
fn rasterization_backend_can_be_replaced_independently() {
    let mut pipeline = Pipeline::new(
        CpuVertex,
        FixedRasterization { sample_index: 3 },
        CpuFragment,
    );
    let result = pipeline
        .render_transformed(
            &scene(vec![triangle(Some(0), 0.0)]),
            &[material(Vec3::Y)],
            2,
            2,
            lighting(),
        )
        .unwrap();
    assert_eq!(
        result.framebuffer.color,
        [Vec3::ZERO, Vec3::ZERO, Vec3::ZERO, Vec3::Y]
    );
    assert_eq!(result.framebuffer.depth.values[3], 0.25);
}

struct SolidFragment;

impl FragmentStage for SolidFragment {
    fn shade(
        &mut self,
        fragments: &[Fragment],
        _: &Material,
        _: &Lighting<'_>,
    ) -> io::Result<Vec<Vec3>> {
        assert!(fragments.len() > 1);
        Ok(vec![Vec3::Z; fragments.len()])
    }
}

#[test]
fn fragment_backend_can_be_replaced_independently() {
    let mut pipeline = Pipeline::new(CpuVertex, CpuRasterization, SolidFragment);
    let result = pipeline
        .render_transformed(
            &scene(vec![triangle(Some(0), 0.0)]),
            &[material(Vec3::X)],
            8,
            8,
            lighting(),
        )
        .unwrap();
    assert!(result.framebuffer.color.contains(&Vec3::Z));
    assert!(!result.framebuffer.color.contains(&Vec3::X));
}

struct BrokenFragment(bool);

impl FragmentStage for BrokenFragment {
    fn shade(&mut self, _: &[Fragment], _: &Material, _: &Lighting<'_>) -> io::Result<Vec<Vec3>> {
        if self.0 {
            Err(io::Error::other("backend failure"))
        } else {
            Ok(Vec::new())
        }
    }
}

#[test]
fn backend_errors_and_invalid_outputs_are_reported() {
    let scene = scene(vec![triangle(Some(0), 0.0)]);
    for (fail, kind) in [
        (false, io::ErrorKind::InvalidData),
        (true, io::ErrorKind::Other),
    ] {
        let mut pipeline = Pipeline::new(CpuVertex, CpuRasterization, BrokenFragment(fail));
        let error = pipeline
            .render_transformed(&scene, &[material(Vec3::X)], 8, 8, lighting())
            .err()
            .unwrap();
        assert_eq!(error.kind(), kind);
    }
    let mut pipeline = Pipeline::new(
        CpuVertex,
        FixedRasterization { sample_index: 64 },
        CpuFragment,
    );
    let error = pipeline
        .render_transformed(&scene, &[material(Vec3::X)], 8, 8, lighting())
        .err()
        .unwrap();
    assert_eq!(error.kind(), io::ErrorKind::InvalidData);
}

#[test]
fn empty_scenes_and_invalid_render_inputs_are_handled_before_shading() {
    let mut pipeline = Pipeline::new(CpuVertex, CpuRasterization, BrokenFragment(true));
    let empty = scene(Vec::new());
    let result = pipeline
        .render_transformed(&empty, &[], 2, 2, lighting())
        .unwrap();
    assert_eq!(result.framebuffer.color, [Vec3::ZERO; 4]);
    assert_eq!(result.covered_fragments, 0);
    assert!(
        pipeline
            .render_transformed(&empty, &[], 0, 2, lighting())
            .is_err()
    );
    assert!(
        pipeline
            .render_transformed(&scene(vec![triangle(Some(0), 0.0)]), &[], 2, 2, lighting())
            .is_err()
    );
    assert!(
        pipeline
            .render_transformed(
                &empty,
                &[],
                2,
                2,
                Lighting {
                    camera_position: Vec3::NAN,
                    ..lighting()
                }
            )
            .is_err()
    );
}
