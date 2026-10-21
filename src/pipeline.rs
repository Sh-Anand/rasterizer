use std::{
    io,
    time::{Duration, Instant},
};

use glam::{Mat4, Vec3, Vec4};

use crate::{
    framebuffer::{DepthBuffer, Framebuffer, RenderOutput},
    geometry::{ClipPrimitive, ClipScene, PreparedScene},
    lighting::RenderLight,
    material::Material,
    shadow::ShadowMap,
    texture::{Texel, UvSample},
};

pub trait VertexStage {
    /// Call after editing prepared geometry or replacing the scene.
    fn invalidate_resources(&mut self) {}

    /// Preserve topology and source identities.
    fn transform(&mut self, scene: &PreparedScene, view_projection: Mat4) -> io::Result<ClipScene>;
}

impl<V: VertexStage + ?Sized> VertexStage for &mut V {
    fn invalidate_resources(&mut self) {
        (**self).invalidate_resources();
    }

    fn transform(&mut self, scene: &PreparedScene, view_projection: Mat4) -> io::Result<ClipScene> {
        (**self).transform(scene, view_projection)
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct FragmentAttributes {
    pub world_position: Vec3,
    pub uv: UvSample,
    pub normal: Vec3,
    pub tangent: Vec4,
}

#[derive(Default, Clone, Copy)]
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

#[derive(Clone, Copy)]
pub enum ShadowProjection {
    Orthographic,
    Perspective { far: f32 },
}

pub trait RasterizationStage {
    /// Clip, project, and depth-test in input order. Equal depths keep the first.
    /// None collects final fragments; Some writes biased shadow depth only.
    fn rasterize(
        &mut self,
        primitives: &[&ClipPrimitive],
        materials: &[Material],
        depth: &mut DepthBuffer,
        shadow: Option<ShadowProjection>,
    ) -> io::Result<RasterizedScene>;
}

/// Fragment-shading resources other than the material table.
#[derive(Clone, Copy)]
pub struct Shading<'a> {
    /// Every material texture's mip levels.
    pub texels: &'a [Texel],
    pub lights: &'a [RenderLight],
    pub maps: &'a [ShadowMap],
    /// Every shadow map's depths.
    pub depths: &'a [f32],
    pub camera_position: Vec3,
}

pub trait FragmentStage {
    /// Call after editing scene resources or replacing the scene, before shading again.
    /// Camera and fragment changes alone need no invalidation.
    fn invalidate_resources(&mut self) {}

    /// Return one linear HDR color per fragment, in input order, after any device readback.
    fn shade(
        &mut self,
        fragments: &[Fragment],
        materials: &[Material],
        shading: &Shading<'_>,
    ) -> io::Result<Vec<Vec3>>;
}

impl<F: FragmentStage + ?Sized> FragmentStage for &mut F {
    fn invalidate_resources(&mut self) {
        (**self).invalidate_resources();
    }

    fn shade(
        &mut self,
        fragments: &[Fragment],
        materials: &[Material],
        shading: &Shading<'_>,
    ) -> io::Result<Vec<Vec3>> {
        (**self).shade(fragments, materials, shading)
    }
}

#[derive(Default)]
pub struct PipelineStats {
    pub vertex_time: Duration,
    pub rasterization_time: Duration,
    pub fragment_time: Duration,
    /// CUDA diffuse bounce rays and hit shading.
    pub bounce_time: Duration,
    pub vertices_shaded: usize,
    pub fragments_shaded: usize,
}

pub struct Pipeline<V, R, F> {
    pub(crate) vertex: V,
    pub(crate) rasterization: R,
    pub(crate) fragment: F,
    pub(crate) stats: PipelineStats,
}

impl<V, R, F> Pipeline<V, R, F> {
    pub fn new(vertex: V, rasterization: R, fragment: F) -> Self {
        Self {
            vertex,
            rasterization,
            fragment,
            stats: PipelineStats::default(),
        }
    }

    /// Camera-pass measurements; depth passes leave these unchanged.
    /// Host stage calls use wall time; resident CUDA stages use stream events.
    pub fn stats(&self) -> &PipelineStats {
        &self.stats
    }
}

impl<V: VertexStage, R, F: FragmentStage> Pipeline<V, R, F> {
    /// Invalidate cached geometry and shading resources.
    pub fn invalidate_resources(&mut self) {
        self.vertex.invalidate_resources();
        self.fragment.invalidate_resources();
    }
}

impl<V: VertexStage, R: RasterizationStage, F: FragmentStage> Pipeline<V, R, F> {
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

    /// Render every shadow map into one depth atlas, in map order.
    pub fn render_shadows(
        &mut self,
        scene: &PreparedScene,
        materials: &[Material],
        maps: &[ShadowMap],
    ) -> io::Result<Vec<f32>> {
        let mut depths = Vec::new();
        for map in maps {
            let transformed = self.vertex.transform(scene, map.view_projection)?;
            let primitives = checked_primitives(&transformed, materials)?;
            let mut depth = DepthBuffer::new(map.width, map.height)?;
            self.rasterization.rasterize(
                &primitives,
                materials,
                &mut depth,
                Some(map.projection),
            )?;
            depths.extend(depth.values);
        }
        Ok(depths)
    }

    pub fn render(
        &mut self,
        scene: &PreparedScene,
        view_projection: Mat4,
        materials: &[Material],
        width: u32,
        height: u32,
        shading: Shading<'_>,
    ) -> io::Result<RenderOutput> {
        let transformed = self.transform(scene, view_projection)?;
        self.render_transformed(&transformed, materials, width, height, shading)
    }

    /// Render geometry returned by the vertex stage.
    pub fn render_transformed(
        &mut self,
        scene: &ClipScene,
        materials: &[Material],
        width: u32,
        height: u32,
        shading: Shading<'_>,
    ) -> io::Result<RenderOutput> {
        self.stats.rasterization_time = Duration::ZERO;
        self.stats.fragment_time = Duration::ZERO;
        self.stats.fragments_shaded = 0;
        if !shading.camera_position.is_finite() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Non-finite camera position",
            ));
        }
        let mut primitives = checked_primitives(scene, materials)?;
        // Preserve equal-depth winners: default material first, then stable material order.
        primitives.sort_by_key(|primitive| primitive.source.material_index);
        let mut framebuffer = Framebuffer::new(width, height)?;
        let started = Instant::now();
        let rasterized =
            self.rasterization
                .rasterize(&primitives, materials, &mut framebuffer.depth, None)?;
        self.stats.rasterization_time = started.elapsed();
        if !rasterized.fragments.is_empty() {
            let started = Instant::now();
            let colors = self
                .fragment
                .shade(&rasterized.fragments, materials, &shading)?;
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
            screen_vertices: Some(rasterized.screen_vertices),
            screen_triangles: rasterized.screen_triangles,
        })
    }
}

fn checked_primitives<'a>(
    scene: &'a ClipScene,
    materials: &[Material],
) -> io::Result<Vec<&'a ClipPrimitive>> {
    if scene.primitives.iter().any(|primitive| {
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
    Ok(scene.primitives.iter().collect())
}
