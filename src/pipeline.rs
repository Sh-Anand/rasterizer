use std::{collections::BTreeMap, io};

use glam::{Mat4, Vec2, Vec3, Vec4};

use crate::{
    framebuffer::{DepthBuffer, Framebuffer, RenderOutput},
    geometry::{ClipPrimitive, ClipScene, PreparedScene},
    lighting::Lighting,
    material::Material,
    texture::UvSample,
};

pub trait VertexStage {
    /// Preserve topology and source identities.
    fn transform(&mut self, scene: &PreparedScene, view_projection: Mat4) -> io::Result<ClipScene>;
}

#[derive(Debug, Clone, Copy)]
pub struct FragmentAttributes {
    pub world_position: Vec3,
    pub uv: UvSample,
    pub normal_uv: UvSample,
    pub metallic_roughness_uv: UvSample,
    pub occlusion_uv: UvSample,
    pub emissive_uv: UvSample,
    pub lightmap_uv: Vec2,
    pub normal: Vec3,
    pub tangent: Vec4,
}

pub struct Fragment {
    /// Index in the render-resolution framebuffer.
    pub sample_index: usize,
    pub attributes: FragmentAttributes,
    pub front_facing: bool,
}

#[derive(Default)]
pub struct RasterizedBatch {
    /// Final winners with unique sample indices; order is unspecified.
    pub fragments: Vec<Fragment>,
    pub covered_fragments: u64,
    pub screen_vertices: usize,
    pub screen_triangles: usize,
}

pub trait RasterizationStage {
    /// Clip, project, interpolate, and depth-test one material batch.
    /// Preserve shared depth; emit final winners only for updated samples. Ties keep the first.
    fn rasterize(
        &mut self,
        primitives: &[&ClipPrimitive],
        material: &Material,
        depth: &mut DepthBuffer,
    ) -> io::Result<RasterizedBatch>;
}

pub trait FragmentStage {
    /// Return one linear HDR color per fragment, in input order, after any device readback.
    fn shade(
        &mut self,
        fragments: &[Fragment],
        material: &Material,
        lighting: &Lighting<'_>,
    ) -> io::Result<Vec<Vec3>>;
}

pub struct Pipeline<V, R, F> {
    vertex: V,
    rasterization: R,
    fragment: F,
}

impl<V: VertexStage, R: RasterizationStage, F: FragmentStage> Pipeline<V, R, F> {
    pub fn new(vertex: V, rasterization: R, fragment: F) -> Self {
        Self {
            vertex,
            rasterization,
            fragment,
        }
    }

    /// Transform geometry for shadow and lightmap preparation before rendering.
    pub fn transform(
        &mut self,
        scene: &PreparedScene,
        view_projection: Mat4,
    ) -> io::Result<ClipScene> {
        self.vertex.transform(scene, view_projection)
    }

    pub fn render(
        &mut self,
        scene: &PreparedScene,
        view_projection: Mat4,
        materials: &[Material],
        width: u32,
        height: u32,
        lighting: Lighting<'_>,
    ) -> io::Result<RenderOutput> {
        let transformed = self.transform(scene, view_projection)?;
        self.render_transformed(&transformed, materials, width, height, lighting)
    }

    /// Render transformed geometry with its current lightmap UVs.
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
        // Default material first, then material index; preserve order within each batch.
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
