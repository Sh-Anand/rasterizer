use std::{collections::BTreeMap, io};

use glam::{Mat4, Vec3};

use crate::{
    framebuffer::{DepthBuffer, Framebuffer, RenderOutput},
    geometry::PreparedScene,
    lighting::Lighting,
    material::Material,
    vertex::{ClipPrimitive, ClipScene},
    viewport::FragmentAttributes,
};

pub trait VertexStage {
    /// Transform prepared geometry without changing its topology or source identities.
    fn transform(&mut self, scene: &PreparedScene, view_projection: Mat4) -> io::Result<ClipScene>;
}

pub struct Fragment {
    /// Linear index into the render-resolution framebuffer (including supersampling).
    pub sample_index: usize,
    pub attributes: FragmentAttributes,
    pub front_facing: bool,
}

#[derive(Default)]
pub struct RasterizedBatch {
    /// Accepted fragments in draw order. A sample may currently appear more than once;
    /// the final-winner collector will remove this within-batch overdraw.
    pub fragments: Vec<Fragment>,
    pub covered_fragments: u64,
    pub screen_vertices: usize,
    pub screen_triangles: usize,
}

pub trait RasterizationStage {
    /// Clip, project, interpolate, alpha-test, and depth-test one material batch.
    /// Depth is shared across batches; implementations must not clear it.
    fn rasterize(
        &mut self,
        primitives: &[&ClipPrimitive],
        material: &Material,
        depth: &mut DepthBuffer,
    ) -> io::Result<RasterizedBatch>;
}

pub trait FragmentStage {
    /// Return one linear HDR color per input fragment, in the same order.
    /// Completion includes any device readback; the coordinator owns color writes.
    fn shade(
        &mut self,
        fragments: &[Fragment],
        material: &Material,
        lighting: &Lighting<'_>,
    ) -> io::Result<Vec<Vec3>>;
}

pub struct Pipeline<V, R, F> {
    pub vertex: V,
    pub rasterization: R,
    pub fragment: F,
}

impl<V: VertexStage, R: RasterizationStage, F: FragmentStage> Pipeline<V, R, F> {
    pub fn render(
        &mut self,
        scene: &PreparedScene,
        view_projection: Mat4,
        materials: &[Material],
        width: u32,
        height: u32,
        lighting: Lighting<'_>,
    ) -> io::Result<RenderOutput> {
        let transformed = self.vertex.transform(scene, view_projection)?;
        self.render_transformed(&transformed, materials, width, height, lighting)
    }

    /// Render an existing vertex result, including any applied or newly baked lightmap UVs.
    pub fn render_transformed(
        &mut self,
        scene: &ClipScene,
        materials: &[Material],
        width: u32,
        height: u32,
        lighting: Lighting<'_>,
    ) -> io::Result<RenderOutput> {
        if !lighting.camera_position.is_finite() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Non-finite camera position",
            ));
        }
        // Default material first, then ascending glTF material index. Preserve draw
        // order inside each material; strict depth comparison makes the first tie win.
        let mut batches = BTreeMap::<Option<usize>, Vec<&ClipPrimitive>>::new();
        for primitive in &scene.primitives {
            let index = primitive.source.material_index;
            if index.is_some_and(|index| index >= materials.len()) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "Primitive material index is outside the material list",
                ));
            }
            batches.entry(index).or_default().push(primitive);
        }
        let default_material = Material::default();
        let mut output = RenderOutput {
            framebuffer: Framebuffer::new(width, height)?,
            covered_fragments: 0,
            screen_vertices: 0,
            screen_triangles: 0,
        };
        for (index, primitives) in batches {
            let material = index.map_or(&default_material, |index| &materials[index]);
            let batch = self.rasterization.rasterize(
                &primitives,
                material,
                &mut output.framebuffer.depth,
            )?;
            output.covered_fragments += batch.covered_fragments;
            output.screen_vertices += batch.screen_vertices;
            output.screen_triangles += batch.screen_triangles;
            if batch.fragments.is_empty() {
                continue;
            }
            let colors = self.fragment.shade(&batch.fragments, material, &lighting)?;
            if colors.len() != batch.fragments.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Fragment stage returned the wrong number of colors",
                ));
            }
            for (fragment, color) in batch.fragments.iter().zip(colors) {
                let sample = output
                    .framebuffer
                    .color
                    .get_mut(fragment.sample_index)
                    .ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            "Rasterization stage returned an invalid sample index",
                        )
                    })?;
                *sample = color;
            }
        }
        Ok(output)
    }
}
