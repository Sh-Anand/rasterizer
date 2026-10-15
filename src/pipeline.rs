use std::io;

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
    /// None selects the default glTF material.
    pub material_index: Option<usize>,
    pub attributes: FragmentAttributes,
    pub front_facing: bool,
}

#[derive(Default)]
pub struct RasterizedScene {
    /// Final winners with unique sample indices; order is unspecified.
    pub fragments: Vec<Fragment>,
    pub covered_fragments: u64,
    pub screen_vertices: usize,
    pub screen_triangles: usize,
}

pub trait RasterizationStage {
    /// Clip, project, interpolate, and depth-test the scene in input order.
    /// Emit final winners only for updated samples. Equal depths keep the first.
    fn rasterize(
        &mut self,
        primitives: &[&ClipPrimitive],
        materials: &[Material],
        depth: &mut DepthBuffer,
    ) -> io::Result<RasterizedScene>;
}

pub trait FragmentStage {
    /// Return one linear HDR color per fragment, in input order, after any device readback.
    fn shade(
        &mut self,
        fragments: &[Fragment],
        materials: &[Material],
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
        let mut primitives: Vec<_> = scene.primitives.iter().collect();
        if primitives.iter().any(|primitive| {
            primitive
                .source
                .material_index
                .is_some_and(|index| index >= materials.len())
        }) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Primitive material index is outside the material list",
            ));
        }
        // Preserve equal-depth winners: default material first, then stable material order.
        primitives.sort_by_key(|primitive| primitive.source.material_index);
        let mut framebuffer = Framebuffer::new(width, height)?;
        let rasterized =
            self.rasterization
                .rasterize(&primitives, materials, &mut framebuffer.depth)?;
        if !rasterized.fragments.is_empty() {
            let colors = self
                .fragment
                .shade(&rasterized.fragments, materials, &lighting)?;
            if colors.len() != rasterized.fragments.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Fragment stage returned the wrong number of colors",
                ));
            }
            for (fragment, color) in rasterized.fragments.iter().zip(colors) {
                let sample = framebuffer
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
        Ok(RenderOutput {
            framebuffer,
            covered_fragments: rasterized.covered_fragments,
            screen_vertices: rasterized.screen_vertices,
            screen_triangles: rasterized.screen_triangles,
        })
    }
}
