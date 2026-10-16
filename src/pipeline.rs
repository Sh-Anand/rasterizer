use std::{
    io,
    time::{Duration, Instant},
};

use glam::{Mat4, Vec3, Vec4};

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
    pub emissive_uv: UvSample,
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

#[derive(Default)]
pub struct PipelineStats {
    pub vertex_time: Duration,
    pub rasterization_time: Duration,
    pub fragment_time: Duration,
    pub vertices_shaded: usize,
    pub fragments_shaded: usize,
}

pub struct Pipeline<V, R, F> {
    vertex: V,
    rasterization: R,
    fragment: F,
    stats: PipelineStats,
}

impl<V: VertexStage, R: RasterizationStage, F: FragmentStage> Pipeline<V, R, F> {
    pub fn new(vertex: V, rasterization: R, fragment: F) -> Self {
        Self {
            vertex,
            rasterization,
            fragment,
            stats: PipelineStats::default(),
        }
    }

    /// Latest stage measurements; transforming a new scene resets all statistics.
    pub fn stats(&self) -> &PipelineStats {
        &self.stats
    }

    /// Transform geometry for shadow preparation and rendering.
    pub fn transform(
        &mut self,
        scene: &PreparedScene,
        view_projection: Mat4,
    ) -> io::Result<ClipScene> {
        self.stats = PipelineStats::default();
        let started = Instant::now();
        let transformed = self.vertex.transform(scene, view_projection)?;
        self.stats.vertex_time = started.elapsed();
        self.stats.vertices_shaded = scene.primitives.iter().map(|p| p.vertices.len()).sum();
        Ok(transformed)
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

    /// Render geometry returned by the vertex stage.
    pub fn render_transformed(
        &mut self,
        scene: &ClipScene,
        materials: &[Material],
        width: u32,
        height: u32,
        lighting: Lighting<'_>,
    ) -> io::Result<RenderOutput> {
        self.stats.rasterization_time = Duration::ZERO;
        self.stats.fragment_time = Duration::ZERO;
        self.stats.fragments_shaded = 0;
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
        let started = Instant::now();
        let rasterized =
            self.rasterization
                .rasterize(&primitives, materials, &mut framebuffer.depth)?;
        self.stats.rasterization_time = started.elapsed();
        if !rasterized.fragments.is_empty() {
            let started = Instant::now();
            let colors = self
                .fragment
                .shade(&rasterized.fragments, materials, &lighting)?;
            self.stats.fragment_time = started.elapsed();
            if colors.len() != rasterized.fragments.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Fragment stage returned the wrong number of colors",
                ));
            }
            self.stats.fragments_shaded = rasterized.fragments.len();
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
