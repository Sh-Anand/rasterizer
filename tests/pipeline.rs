use std::io;

use glam::{Mat4, Vec2, Vec3, Vec4};

use rasterizer::{
    backend::cpu::{self, CpuFragment, CpuRasterization, CpuVertex},
    clip::clip_scene,
    framebuffer::{DepthBuffer, RenderOutput},
    geometry::{PreparedScene, PrimitiveSource, Triangle},
    lighting::Lighting,
    material::Material,
    pipeline::{
        Fragment, FragmentStage, Pipeline, RasterizationStage, RasterizedBatch, VertexStage,
    },
    texture::UvSample,
    vertex::{ClipPrimitive, ClipScene, ClipVertex},
    viewport::{FragmentAttributes, project_scene},
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
    for (a, b) in actual
        .framebuffer
        .color
        .iter()
        .zip(&expected.framebuffer.color)
    {
        assert_eq!(
            a.to_array().map(f32::to_bits),
            b.to_array().map(f32::to_bits)
        );
    }
    for (a, b) in actual
        .framebuffer
        .depth
        .values
        .iter()
        .zip(&expected.framebuffer.depth.values)
    {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

#[derive(Default)]
struct RecordingFragment {
    materials: Vec<Vec3>,
}

impl FragmentStage for RecordingFragment {
    fn shade(
        &mut self,
        fragments: &[Fragment],
        material: &Material,
        lighting: &Lighting<'_>,
    ) -> io::Result<Vec<Vec3>> {
        assert!(!fragments.is_empty());
        self.materials.push(material.emissive_factor);
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
    let mut pipeline = Pipeline {
        vertex: CpuVertex,
        rasterization: CpuRasterization,
        fragment: RecordingFragment::default(),
    };
    let result = pipeline
        .render_transformed(&scene, &materials, 8, 8, lighting())
        .unwrap();
    assert_eq!(pipeline.fragment.materials, [Vec3::ZERO, Vec3::X]);
    let mut reference_scene = scene.clone();
    reference_scene
        .primitives
        .sort_by_key(|p| p.source.material_index);
    let screen = project_scene(&clip_scene(&reference_scene).unwrap(), 8, 8).unwrap();
    let reference = cpu::render(&screen, &materials, 8, 8, lighting()).unwrap();
    assert_same(&result, &reference);
    assert!(
        result
            .framebuffer
            .depth
            .values
            .iter()
            .any(|z| z.is_finite())
    );
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
    let mut pipeline = Pipeline {
        vertex: CpuVertex,
        rasterization: CpuRasterization,
        fragment: CpuFragment,
    };
    let result = pipeline
        .render_transformed(&scene, &materials, 16, 16, lighting())
        .unwrap();
    let mut reference_scene = scene.clone();
    reference_scene
        .primitives
        .sort_by_key(|p| p.source.material_index);
    let screen = project_scene(&clip_scene(&reference_scene).unwrap(), 16, 16).unwrap();
    let reference = cpu::render(&screen, &materials, 16, 16, lighting()).unwrap();
    assert_same(&result, &reference);
    assert!(result.framebuffer.color.contains(&Vec3::Z));
}

struct FixedVertex {
    scene: ClipScene,
    called: bool,
}

impl VertexStage for FixedVertex {
    fn transform(&mut self, scene: &PreparedScene, view_projection: Mat4) -> io::Result<ClipScene> {
        assert!(scene.primitives.is_empty());
        assert_eq!(view_projection, Mat4::IDENTITY);
        self.called = true;
        Ok(self.scene.clone())
    }
}

#[test]
fn vertex_backend_can_be_replaced_independently() {
    let mut pipeline = Pipeline {
        vertex: FixedVertex {
            scene: scene(vec![triangle(Some(0), 0.0)]),
            called: false,
        },
        rasterization: CpuRasterization,
        fragment: CpuFragment,
    };
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
    assert!(pipeline.vertex.called);
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
    let mut pipeline = Pipeline {
        vertex: CpuVertex,
        rasterization: FixedRasterization { sample_index: 3 },
        fragment: CpuFragment,
    };
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
    let mut pipeline = Pipeline {
        vertex: CpuVertex,
        rasterization: CpuRasterization,
        fragment: SolidFragment,
    };
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
        let mut pipeline = Pipeline {
            vertex: CpuVertex,
            rasterization: CpuRasterization,
            fragment: BrokenFragment(fail),
        };
        let error = pipeline
            .render_transformed(&scene, &[material(Vec3::X)], 8, 8, lighting())
            .err()
            .unwrap();
        assert_eq!(error.kind(), kind);
    }
    let mut pipeline = Pipeline {
        vertex: CpuVertex,
        rasterization: FixedRasterization { sample_index: 64 },
        fragment: CpuFragment,
    };
    let error = pipeline
        .render_transformed(&scene, &[material(Vec3::X)], 8, 8, lighting())
        .err()
        .unwrap();
    assert_eq!(error.kind(), io::ErrorKind::InvalidData);
}

#[test]
fn empty_scenes_and_invalid_render_inputs_are_handled_before_shading() {
    let mut pipeline = Pipeline {
        vertex: CpuVertex,
        rasterization: CpuRasterization,
        fragment: BrokenFragment(true),
    };
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
