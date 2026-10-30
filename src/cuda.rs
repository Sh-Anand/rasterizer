use std::{io, sync::Arc, time::Duration};

use cuda_core::{
    CudaContext, CudaEvent, CudaStream, DeviceBuffer,
    simt::{LaunchConfig, memory},
};
use cuda_device::{atomic::DeviceAtomicU64, cuda_module};
use glam::{Mat4, Vec2, Vec3, Vec4};

use crate::{
    bvh::{self, Node},
    config::{LightConfig, ShadowConfig},
    fragment::{Shading, uniform},
    geometry::{ClipVertex, PreparedScene, PreparedVertex, VertexTransform},
    lighting::Light,
    material::Material,
    rasterization::{
        FragmentAttributes, Polygon, ScreenVertex, Triangle, UvGradients, clip_triangle,
        interpolate_attributes, outcode, project_vertex,
    },
    texture::{Texel, UvSample},
    trace::{Site, TRACE, mark},
};

/// Diffuse bounce rays per pixel, fixed at build time by the `BOUNCE_RAYS` environment variable.
pub const BOUNCE_RAYS: u32 = match option_env!("BOUNCE_RAYS") {
    None => 4,
    Some(value) => match u32::from_str_radix(value, 10) {
        Ok(rays) if rays > 0 => rays,
        _ => panic!("BOUNCE_RAYS must be a positive integer"),
    },
};

/// Diffuse bounces per path, fixed at build time by the `BOUNCES` environment variable.
pub const BOUNCES: u32 = match option_env!("BOUNCES") {
    None => 1,
    Some(value) => match u32::from_str_radix(value, 10) {
        Ok(bounces) if bounces > 0 => bounces,
        _ => panic!("BOUNCES must be a positive integer"),
    },
};

/// Per-pixel slots in the trace buffer: the primary surface, then each ray's hits by depth.
pub const SLOTS: usize = 1 + (BOUNCE_RAYS * BOUNCES) as usize;

// Component planes are grouped in warps to keep indexed vertex reads nearby.
fn padded_count(count: usize) -> usize {
    count.next_multiple_of(32)
}

fn soa_offset<const COMPONENTS: usize>(index: usize, component: usize) -> usize {
    (index / 32) * (COMPONENTS * 32) + component * 32 + index % 32
}

macro_rules! soa {
    ($ty:ty, $components:literal, $( $field:ident: $vector:ident = $offset:literal ),+ $(,)?) => {
        impl $ty {
            const COMPONENTS: usize = $components;

            fn load_soa(values: &[f32], index: usize) -> Self {
                assert!(index / 32 < values.len() / (Self::COMPONENTS * 32));
                Self {
                    $($field: $vector::from_array(std::array::from_fn(|c| {
                        // The checked index and fixed component offsets stay within the slice.
                        unsafe { *values.get_unchecked(soa_offset::<$components>(index, $offset + c)) }
                    }))),+
                }
            }

            fn store_soa(&self, index: usize, mut write: impl FnMut(usize, f32)) {
                $({
                    let values = self.$field.to_array();
                    for c in 0..values.len() {
                        write(soa_offset::<$components>(index, $offset + c), values[c]);
                    }
                })+
            }
        }
    };
}

impl UvSample {
    fn from_array([u, v, dx_u, dx_v, dy_u, dy_v]: [f32; 6]) -> Self {
        Self {
            uv: Vec2::new(u, v),
            dx: Vec2::new(dx_u, dx_v),
            dy: Vec2::new(dy_u, dy_v),
        }
    }

    fn to_array(self) -> [f32; 6] {
        [
            self.uv.x, self.uv.y, self.dx.x, self.dx.y, self.dy.x, self.dy.y,
        ]
    }
}

soa!(PreparedVertex, 12,
    position: Vec3 = 0, uv: Vec2 = 3, normal: Vec3 = 5, tangent: Vec4 = 8,
);
soa!(ClipVertex, 16,
    position: Vec4 = 0, world_position: Vec3 = 4, uv: Vec2 = 7,
    normal: Vec3 = 9, tangent: Vec4 = 12,
);
soa!(FragmentAttributes, 16,
    world_position: Vec3 = 0, uv: UvSample = 3, normal: Vec3 = 9, tangent: Vec4 = 12,
);

struct Buffer<T> {
    device: DeviceBuffer<T>,
    capacity: usize,
}

impl<T> Buffer<T> {
    fn new(context: &Arc<CudaContext>) -> Self {
        Self {
            // Empty buffers own no allocation.
            device: unsafe { DeviceBuffer::from_raw_parts(0, 0, context.clone()) },
            capacity: 0,
        }
    }

    fn resize(&mut self, len: usize) -> io::Result<&mut DeviceBuffer<T>> {
        let bytes = len.checked_mul(size_of::<T>()).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "CUDA buffer size overflow")
        })?;
        if len > self.capacity {
            let ptr = unsafe { memory::malloc_sync(bytes) }.map_err(io::Error::other)?;
            // Allocation is owned here; callers initialize it before device reads.
            self.device =
                unsafe { DeviceBuffer::from_raw_parts(ptr, len, self.device.context().clone()) };
            self.capacity = len;
        } else if self.device.len() != len {
            let empty = Self::new(self.device.context());
            let (ptr, _, context) =
                std::mem::replace(&mut self.device, empty.device).into_raw_parts();
            // Change the visible length while retaining the larger allocation.
            self.device = unsafe { DeviceBuffer::from_raw_parts(ptr, len, context) };
        }
        Ok(&mut self.device)
    }

    // Used only for integer buffers.
    fn fill(&mut self, byte: u8, stream: &CudaStream) -> io::Result<()> {
        if !self.device.is_empty() {
            unsafe {
                memory::memset_d8_async(
                    self.device.cu_deviceptr(),
                    byte,
                    self.device.num_bytes(),
                    stream.cu_stream(),
                )
            }
            .map_err(io::Error::other)?;
        }
        Ok(())
    }

    /// Values must contain no host pointers; the buffer's context must be current.
    unsafe fn upload(&mut self, values: &[T]) -> io::Result<()>
    where
        T: Copy,
    {
        let dst = self.resize(values.len())?.cu_deviceptr();
        if !values.is_empty() {
            // The copy fills the visible length and finishes before return.
            unsafe { memory::memcpy_htod_sync(dst, values.as_ptr(), size_of_val(values)) }
                .map_err(io::Error::other)?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct CudaTriangle {
    vertices: [u32; 3],
    primitive: u32,
}
#[derive(Clone, Copy)]
struct CudaPrimitive {
    /// The material count selects the default material appended after the scene's.
    material_index: u32,
    mirrored: bool,
    double_sided: bool,
}

/// Event timings of the last camera render; shadow passes leave them unchanged. Event
/// intervals can include host launch delays.
#[derive(Default)]
pub struct Stats {
    pub vertex_time: Duration,
    pub rasterization_time: Duration,
    pub fragment_time: Duration,
    /// Diffuse bounce rays and hit shading.
    pub bounce_time: Duration,
}

/// Renders on one CUDA stream, keeping geometry, shading resources and intermediates resident.
pub struct Renderer {
    stream: Arc<CudaStream>,
    module: kernels::LoadedModule,
    /// Set by `prepare`.
    light: Option<Light>,
    stats: Stats,
    vertices: Buffer<f32>,
    primitive_indices: Buffer<u32>,
    transforms: Buffer<VertexTransform>,
    output: Buffer<f32>,
    triangles: Buffer<CudaTriangle>,
    primitives: Buffer<CudaPrimitive>,
    /// Bounce BVH over `triangles`; empty without the bounce.
    nodes: Buffer<Node>,
    order: Buffer<u32>,
    winners: Buffer<DeviceAtomicU64>,
    /// Large-triangle queue length and cursor.
    queue: Buffer<DeviceAtomicU64>,
    large: Buffer<[u32; 2]>,
    fragments: Buffer<f32>,
    fragment_meta: Buffer<u32>,
    materials: Buffer<Material>,
    texels: Buffer<Texel>,
    depths: Buffer<f32>,
    colors: Buffer<[f32; 3]>,
    /// Primary diffuse albedo per pixel, written by shading for the bounce.
    albedo: Buffer<[f32; 3]>,
    /// With `TRACE`, the region bits and material of each pixel's primary and bounce lanes.
    trace: Buffer<u32>,
}

impl Renderer {
    pub fn new(device: usize) -> io::Result<Self> {
        let context = CudaContext::new(device).map_err(io::Error::other)?;
        let c = &context;
        Ok(Self {
            stream: context.default_stream(),
            module: kernels::load(&context).map_err(io::Error::other)?,
            light: None,
            stats: Stats::default(),
            vertices: Buffer::new(c),
            primitive_indices: Buffer::new(c),
            transforms: Buffer::new(c),
            output: Buffer::new(c),
            triangles: Buffer::new(c),
            primitives: Buffer::new(c),
            nodes: Buffer::new(c),
            order: Buffer::new(c),
            winners: Buffer::new(c),
            queue: Buffer::new(c),
            large: Buffer::new(c),
            fragments: Buffer::new(c),
            fragment_meta: Buffer::new(c),
            materials: Buffer::new(c),
            texels: Buffer::new(c),
            depths: Buffer::new(c),
            colors: Buffer::new(c),
            albedo: Buffer::new(c),
            trace: Buffer::new(c),
        })
    }

    pub fn stats(&self) -> &Stats {
        &self.stats
    }

    fn event(&self) -> io::Result<CudaEvent> {
        self.stream.record_event(Some(0)).map_err(io::Error::other)
    }

    /// `world` holds every vertex's world position when the bounce needs a BVH.
    fn upload_geometry(
        &mut self,
        scene: &PreparedScene,
        materials: &[Material],
        world: Option<&[Vec3]>,
    ) -> io::Result<()> {
        let count = scene.primitives.iter().map(|p| p.vertices.len()).sum();
        let mut triangles = Vec::new();
        let mut vertices = vec![0.0; padded_count(count) * PreparedVertex::COMPONENTS];
        let mut indices = Vec::with_capacity(count);
        let mut primitives = Vec::with_capacity(scene.primitives.len());
        for (i, primitive) in scene.primitives.iter().enumerate() {
            let base = indices.len() as u32;
            for (index, vertex) in primitive.vertices.iter().enumerate() {
                vertex.store_soa(base as usize + index, |offset, v| vertices[offset] = v);
            }
            indices.extend(std::iter::repeat_n(i as u32, primitive.vertices.len()));
            primitives.push(CudaPrimitive {
                material_index: primitive.material_index.unwrap_or(materials.len()) as u32,
                mirrored: primitive.mirrored,
                double_sided: primitive
                    .material_index
                    .is_some_and(|i| materials[i].double_sided),
            });
            triangles.extend(primitive.triangles.iter().map(|triangle| CudaTriangle {
                vertices: triangle.map(|v| base + v),
                primitive: i as u32,
            }));
        }
        // Ranks index this stable material order, with the default material first.
        triangles.sort_by_key(|t| scene.primitives[t.primitive as usize].material_index);
        let corners: Vec<_> = world.map_or(Vec::new(), |world| {
            triangles
                .iter()
                .map(|t| t.vertices.map(|v| world[v as usize]))
                .collect()
        });
        let (nodes, order) = bvh::build(&corners);
        // Geometry and descriptors contain values and indices, never host pointers.
        unsafe {
            self.vertices.upload(&vertices)?;
            self.primitive_indices.upload(&indices)?;
            self.triangles.upload(&triangles)?;
            self.primitives.upload(&primitives)?;
            self.nodes.upload(&nodes)?;
            self.order.upload(&order)?;
        }
        Ok(())
    }

    fn transform(
        &mut self,
        scene: &PreparedScene,
        view_projection: Mat4,
    ) -> io::Result<(CudaEvent, CudaEvent)> {
        let count = self.primitive_indices.device.len();
        let transforms: Vec<_> = scene
            .primitives
            .iter()
            .map(|primitive| VertexTransform::new(primitive, view_projection))
            .collect();
        // Transform descriptors contain only matrices and a scalar sign.
        unsafe { self.transforms.upload(&transforms)? };
        self.output
            .resize(padded_count(count) * ClipVertex::COMPONENTS)?;
        let start = self.event()?;
        if count != 0 {
            unsafe {
                self.module
                    .transform(
                        &self.stream,
                        LaunchConfig::for_num_elems(count as u32),
                        &self.vertices.device,
                        &self.primitive_indices.device,
                        &self.transforms.device,
                        &mut self.output.device,
                    )
                    .map_err(io::Error::other)?;
            }
        }
        Ok((start, self.event()?))
    }

    fn rasterize(
        &mut self,
        width: u32,
        height: u32,
        shadow: bool,
    ) -> io::Result<(CudaEvent, CudaEvent)> {
        let count = width * height;
        self.winners.resize(count as usize)?;
        self.queue.resize(2)?;
        self.large.resize(self.triangles.device.len())?;
        if !shadow {
            self.fragments
                .resize(padded_count(count as usize) * FragmentAttributes::COMPONENTS)?;
            self.fragment_meta
                .resize(padded_count(count as usize) * 2)?;
        }
        let start = self.event()?;
        self.winners.fill(255, &self.stream)?;
        self.queue.fill(0, &self.stream)?;
        if !self.triangles.device.is_empty() {
            unsafe {
                self.module
                    .rasterize_small(
                        &self.stream,
                        LaunchConfig::for_num_elems(self.triangles.device.len() as u32),
                        &self.output.device,
                        &self.triangles.device,
                        &self.primitives.device,
                        width,
                        height,
                        shadow,
                        &self.winners.device,
                        &self.queue.device,
                        &mut self.large.device,
                    )
                    .map_err(io::Error::other)?;
                self.module
                    .rasterize_large(
                        &self.stream,
                        LaunchConfig {
                            grid_dim: (256, 1, 1),
                            block_dim: (128, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        &self.output.device,
                        &self.triangles.device,
                        &self.primitives.device,
                        width,
                        height,
                        shadow,
                        &self.winners.device,
                        &self.queue.device,
                        &self.large.device,
                    )
                    .map_err(io::Error::other)?;
            }
        }
        if !shadow {
            unsafe {
                self.module
                    .interpolate(
                        &self.stream,
                        LaunchConfig::for_num_elems(count),
                        &self.output.device,
                        &self.triangles.device,
                        &self.primitives.device,
                        width,
                        height,
                        &self.winners.device,
                        &mut self.fragments.device,
                        &mut self.fragment_meta.device,
                    )
                    .map_err(io::Error::other)?;
            }
        }
        Ok((start, self.event()?))
    }

    fn shade(
        &mut self,
        width: u32,
        height: u32,
        light: Light,
        camera_position: Vec3,
    ) -> io::Result<(CudaEvent, CudaEvent)> {
        let bounce = !self.nodes.device.is_empty();
        if bounce {
            self.albedo.resize(width as usize * height as usize)?;
        }
        if TRACE {
            self.trace
                .resize(2 * SLOTS * width as usize * height as usize)?;
            self.trace.fill(0, &self.stream)?;
        }
        let start = self.event()?;
        // Buffers live through the synchronous readback.
        unsafe {
            self.module.shade(
                &self.stream,
                tile_launch(width, height),
                width,
                height,
                &self.fragments.device,
                &self.fragment_meta.device,
                &self.winners.device,
                &self.materials.device,
                &self.texels.device,
                light,
                &self.depths.device,
                camera_position.to_array(),
                bounce,
                &mut self.colors.device,
                &mut self.albedo.device,
                &mut self.trace.device,
            )
        }
        .map_err(io::Error::other)?;
        Ok((start, self.event()?))
    }

    fn bounce(
        &mut self,
        width: u32,
        height: u32,
        light: Light,
    ) -> io::Result<(CudaEvent, CudaEvent)> {
        let start = self.event()?;
        // Buffers live through the synchronous readback.
        unsafe {
            self.module.bounce(
                &self.stream,
                tile_launch(width, height),
                width,
                height,
                &self.fragments.device,
                &self.fragment_meta.device,
                &self.winners.device,
                &self.output.device,
                &self.triangles.device,
                &self.primitives.device,
                &self.nodes.device,
                &self.order.device,
                &self.materials.device,
                &self.texels.device,
                light,
                &self.depths.device,
                &self.albedo.device,
                &mut self.colors.device,
                &mut self.trace.device,
            )
        }
        .map_err(io::Error::other)?;
        Ok((start, self.event()?))
    }

    /// Upload geometry and textures and render the light's shadow map; repeat after scene
    /// changes. With `bounce`, also build the BVH that adds a diffuse bounce to every render.
    pub fn prepare(
        &mut self,
        scene: &PreparedScene,
        materials: &[Material],
        texels: &[Texel],
        light: &LightConfig,
        shadow: &ShadowConfig,
        bounce: bool,
    ) -> io::Result<()> {
        self.stream
            .context()
            .bind_to_thread()
            .map_err(io::Error::other)?;
        let positions = if shadow.enabled || bounce {
            scene.world_positions()
        } else {
            Vec::new()
        };
        self.upload_geometry(scene, materials, bounce.then_some(&positions[..]))?;
        let light = Light::new(light, shadow, &positions);
        // The default material follows the scene materials, so every index reads device memory.
        let table: Vec<_> = materials
            .iter()
            .copied()
            .chain([Material::default()])
            .collect();
        // Materials hold values and array offsets, never host pointers.
        unsafe {
            self.materials.upload(&table)?;
            self.texels.upload(texels)?;
        }
        self.depths
            .resize(light.shadow.map_or(0, |map| map.len()))?;
        if let Some(map) = light.shadow {
            self.transform(scene, map.view_projection)?;
            self.rasterize(map.resolution, map.resolution, true)?;
            unsafe {
                self.module.resolve_depth(
                    &self.stream,
                    LaunchConfig::for_num_elems(map.resolution * map.resolution),
                    &self.winners.device,
                    &mut self.depths.device,
                )
            }
            .map_err(io::Error::other)?;
        }
        self.light = Some(light);
        Ok(())
    }

    /// Returns linear RGB per sample; all intermediate camera and shadow data stays on the device.
    pub fn render(
        &mut self,
        scene: &PreparedScene,
        view_projection: Mat4,
        width: u32,
        height: u32,
        camera_position: Vec3,
    ) -> io::Result<Vec<Vec3>> {
        self.stream
            .context()
            .bind_to_thread()
            .map_err(io::Error::other)?;
        let light = self.light.expect("prepare before render");
        let vertex_events = self.transform(scene, view_projection)?;
        let raster_events = self.rasterize(width, height, false)?;
        self.colors.resize(width as usize * height as usize)?;
        let fragment_events = self.shade(width, height, light, camera_position)?;
        let bounce_events = if self.nodes.device.is_empty() {
            None
        } else {
            Some(self.bounce(width, height, light)?)
        };
        let color = self
            .colors
            .device
            .to_host_vec(&self.stream)
            .map_err(io::Error::other)?;
        self.stats = Stats {
            vertex_time: elapsed(&vertex_events)?,
            rasterization_time: elapsed(&raster_events)?,
            fragment_time: elapsed(&fragment_events)?,
            bounce_time: bounce_events.as_ref().map_or(Ok(Duration::ZERO), elapsed)?,
        };
        Ok(color.into_iter().map(Vec3::from_array).collect())
    }

    /// With `TRACE`, the last render's region bits and material per pixel and pass, as
    /// `trace::write` reads them; otherwise empty.
    pub fn trace_lanes(&self) -> io::Result<Vec<u32>> {
        if !TRACE {
            return Ok(Vec::new());
        }
        self.trace
            .device
            .to_host_vec(&self.stream)
            .map_err(io::Error::other)
    }
}

#[cuda_module]
mod kernels {
    use super::*;
    use cuda_device::{DisjointSlice, SharedArray, atomic::AtomicOrdering, kernel, thread};

    #[kernel]
    pub fn resolve_depth(winners: &[DeviceAtomicU64], mut depth: DisjointSlice<f32>) {
        let index = thread::index_1d().get();
        if let Some(winner) = winners.get(index) {
            let key = winner.load(AtomicOrdering::Relaxed);
            let ordered = (key >> 32) as u32;
            let bits = if ordered >> 31 != 0 {
                ordered ^ 0x80000000
            } else {
                !ordered
            };
            // Each lane writes one distinct texel in the checked destination range.
            unsafe {
                *depth.get_unchecked_mut(index) = if key == u64::MAX {
                    f32::INFINITY
                } else {
                    f32::from_bits(bits)
                };
            }
        }
    }

    const SMALL_SAMPLES: usize = 256;
    const MAX_PARTS: usize = 16;

    fn project_faces(
        vertices: &[f32],
        triangle: &CudaTriangle,
        width: u32,
        height: u32,
        mut emit: impl FnMut(&[ScreenVertex; 3], u32),
    ) {
        let codes = triangle.vertices.map(|i| {
            outcode(Vec4::from_array(std::array::from_fn(|c| {
                vertices[soa_offset::<{ ClipVertex::COMPONENTS }>(i as usize, c)]
            })))
        });
        if codes[0] & codes[1] & codes[2] != 0 {
            return;
        }
        let project = |vertex| project_vertex(vertex, width as f32, height as f32);
        // Most triangles avoid the clipping scratch entirely.
        if codes[0] | codes[1] | codes[2] == 0 {
            emit(
                &triangle
                    .vertices
                    .map(|i| project(ClipVertex::load_soa(vertices, i as usize))),
                0,
            );
            return;
        }
        let mut polygon = Polygon::default();
        clip_triangle(
            |i| ClipVertex::load_soa(vertices, i as usize),
            triangle.vertices,
            &mut polygon,
        );
        if polygon.len < 3 {
            return;
        }
        let first = project(polygon.vertices[0]);
        let mut previous = project(polygon.vertices[1]);
        for fan in 0..polygon.len - 2 {
            let current = project(polygon.vertices[fan + 2]);
            emit(&[first, previous, current], fan as u32);
            previous = current;
        }
    }

    fn raster_samples(
        setup: &Triangle,
        width: u32,
        height: u32,
        rank: u32,
        first: usize,
        stride: usize,
        winners: &[DeviceAtomicU64],
    ) {
        let (xs, ys) = setup.bounds(width, 0..height);
        let columns = (xs.end - xs.start) as usize;
        let count = columns * (ys.end - ys.start) as usize;
        let mut sample = first;
        while sample < count {
            let x = xs.start + (sample % columns) as u32;
            let y = ys.start + (sample / columns) as u32;
            sample += stride;
            if let Some(barycentric) = setup.coverage(x, y)
                && let Some(z) = setup.depth(barycentric)
            {
                let bits = if z == 0.0 { 0 } else { z.to_bits() };
                let ordered = if bits >> 31 == 0 {
                    bits ^ 0x80000000
                } else {
                    !bits
                };
                winners[y as usize * width as usize + x as usize].fetch_min(
                    ((ordered as u64) << 32) | rank as u64,
                    AtomicOrdering::Relaxed,
                );
            }
        }
    }

    #[kernel]
    pub fn rasterize_small(
        vertices: &[f32],
        triangles: &[CudaTriangle],
        primitives: &[CudaPrimitive],
        width: u32,
        height: u32,
        shadow: bool,
        winners: &[DeviceAtomicU64],
        queue: &[DeviceAtomicU64],
        mut large: DisjointSlice<[u32; 2]>,
    ) {
        let index = thread::index_1d().get();
        if let Some(triangle) = triangles.get(index) {
            let primitive = primitives[triangle.primitive as usize];
            let rank = index as u32 * 7;
            let mut parts = 0;
            project_faces(vertices, triangle, width, height, |points, fan| {
                if let Some(setup) =
                    Triangle::new(points, primitive.mirrored, primitive.double_sided, shadow)
                {
                    let (xs, ys) = setup.bounds(width, 0..height);
                    let count = (xs.end - xs.start) as usize * (ys.end - ys.start) as usize;
                    if count > SMALL_SAMPLES {
                        parts = parts.max(count.div_ceil(2048).min(MAX_PARTS));
                    } else {
                        raster_samples(&setup, width, height, rank + fan, 0, 1, winners);
                    }
                }
            });
            if parts != 0 {
                let slot = queue[0].fetch_add(1, AtomicOrdering::Relaxed) as usize;
                // Each input triangle reserves at most one unique slot.
                unsafe {
                    *large.get_unchecked_mut(slot) = [index as u32, parts as u32];
                }
            }
        }
    }

    #[kernel]
    pub fn rasterize_large(
        vertices: &[f32],
        triangles: &[CudaTriangle],
        primitives: &[CudaPrimitive],
        width: u32,
        height: u32,
        shadow: bool,
        winners: &[DeviceAtomicU64],
        queue: &[DeviceAtomicU64],
        large: &[[u32; 2]],
    ) {
        static mut JOB: SharedArray<u64, 1> = SharedArray::UNINIT;
        static mut LENGTH: SharedArray<u32, 1> = SharedArray::UNINIT;
        static mut SETUPS: SharedArray<Option<Triangle>, 7> = SharedArray::UNINIT;
        let lane = thread::threadIdx_x() as usize;
        loop {
            if lane == 0 {
                unsafe {
                    JOB[0] = queue[1].fetch_add(1, AtomicOrdering::Relaxed);
                    LENGTH[0] = 0;
                }
            }
            thread::sync_threads();
            let job = unsafe { JOB[0] as usize };
            if job >= queue[0].load(AtomicOrdering::Relaxed) as usize * MAX_PARTS {
                return;
            }
            let [index, parts] = large[job / MAX_PARTS];
            if job % MAX_PARTS >= parts as usize {
                thread::sync_threads();
                continue;
            }
            let index = index as usize;
            let triangle = triangles[index];
            if lane == 0 {
                let primitive = primitives[triangle.primitive as usize];
                project_faces(vertices, &triangle, width, height, |points, fan| unsafe {
                    SETUPS[fan as usize] =
                        Triangle::new(points, primitive.mirrored, primitive.double_sided, shadow);
                    LENGTH[0] = fan + 1;
                });
            }
            thread::sync_threads();
            for fan in 0..unsafe { LENGTH[0] } {
                if let Some(setup) = unsafe { SETUPS[fan as usize] } {
                    let (xs, ys) = setup.bounds(width, 0..height);
                    let count = (xs.end - xs.start) as usize * (ys.end - ys.start) as usize;
                    if count > SMALL_SAMPLES {
                        raster_samples(
                            &setup,
                            width,
                            height,
                            index as u32 * 7 + fan,
                            (job % MAX_PARTS) * thread::blockDim_x() as usize + lane,
                            parts as usize * thread::blockDim_x() as usize,
                            winners,
                        );
                    }
                }
            }
            thread::sync_threads();
        }
    }

    #[kernel]
    pub fn interpolate(
        vertices: &[f32],
        triangles: &[CudaTriangle],
        primitives: &[CudaPrimitive],
        width: u32,
        height: u32,
        winners: &[DeviceAtomicU64],
        mut fragments: DisjointSlice<f32>,
        mut fragment_meta: DisjointSlice<u32>,
    ) {
        let index = thread::index_1d().get();
        if let Some(winner) = winners.get(index) {
            let key = winner.load(AtomicOrdering::Relaxed);
            if key != u64::MAX {
                let rank = key as u32;
                let triangle = triangles[(rank / 7) as usize];
                let primitive = primitives[triangle.primitive as usize];
                let mut shaded = false;
                project_faces(vertices, &triangle, width, height, |points, fan| {
                    if fan != rank % 7 {
                        return;
                    }
                    let Some(setup) = Triangle::new(points, primitive.mirrored, true, false) else {
                        return;
                    };
                    let Some(barycentric) = setup.coverage(
                        (index % width as usize) as u32,
                        (index / width as usize) as u32,
                    ) else {
                        return;
                    };
                    let attributes =
                        interpolate_attributes(points, barycentric, &UvGradients::new(points));
                    // Each lane owns its index in every component plane.
                    attributes.store_soa(index, |offset, v| unsafe {
                        *fragments.get_unchecked_mut(offset) = v;
                    });
                    unsafe {
                        *fragment_meta.get_unchecked_mut(soa_offset::<2>(index, 0)) =
                            primitive.material_index;
                        *fragment_meta.get_unchecked_mut(soa_offset::<2>(index, 1)) =
                            u32::from(setup.front_facing);
                    }
                    shaded = true;
                });
                if !shaded {
                    winner.store(u64::MAX, AtomicOrdering::Relaxed);
                }
            }
        }
    }

    #[kernel]
    pub fn transform(
        vertices: &[f32],
        primitive_indices: &[u32],
        transforms: &[VertexTransform],
        mut output: DisjointSlice<f32>,
    ) {
        let offset = thread::index_1d().get();
        if let Some(&primitive) = primitive_indices.get(offset) {
            let vertex =
                transforms[primitive as usize].apply(&PreparedVertex::load_soa(vertices, offset));
            // Each lane owns its index in every component plane.
            vertex.store_soa(offset, |index, v| unsafe {
                *output.get_unchecked_mut(index) = v;
            });
        }
    }

    /// A thread's lane in its warp's 8x4 pixel block (one 8-pixel row per group of 8 lanes), its
    /// pixel's column, row and sample index, and whether a fragment covers it.
    #[inline(always)]
    fn block_pixel(
        index: usize,
        width: u32,
        height: u32,
        winners: &[DeviceAtomicU64],
    ) -> (usize, usize, usize, usize, bool) {
        let (block, lane) = (index / 32, index % 32);
        let blocks = (width as usize).div_ceil(8);
        let x = block % blocks * 8 + lane % 8;
        let y = block / blocks * 4 + lane / 8;
        let sample = y * width as usize + x;
        let covered = x < width as usize
            && y < height as usize
            && winners[sample].load(AtomicOrdering::Relaxed) != u64::MAX;
        (lane, x, y, sample, covered)
    }

    #[kernel]
    #[cuda_device::launch_bounds(32, SHADE_BLOCKS)]
    pub fn shade(
        width: u32,
        height: u32,
        fragments: &[f32],
        fragment_meta: &[u32],
        winners: &[DeviceAtomicU64],
        materials: &[Material],
        texels: &[Texel],
        light: Light,
        depths: &[f32],
        camera: [f32; 3],
        bounce: bool,
        mut colors: DisjointSlice<[f32; 3]>,
        mut albedo: DisjointSlice<[f32; 3]>,
        mut trace: DisjointSlice<u32>,
    ) {
        // Empty samples take the default material: every lane samples one, since the
        // tensor-core decoder runs warp-wide.
        let (lane, x, y, sample, covered) =
            block_pixel(thread::index_1d().get(), width, height, winners);
        let material_index = if covered {
            fragment_meta[soa_offset::<2>(sample, 0)]
        } else {
            materials.len() as u32 - 1
        };
        let material = &materials[material_index as usize];
        let attributes = if covered {
            FragmentAttributes::load_soa(fragments, sample)
        } else {
            FragmentAttributes::default()
        };
        let mut sites = 0;
        mark(&mut sites, Site::Input);
        let surface = material.sample(texels, attributes.uv, &mut sites);
        let mut color = Vec3::ZERO;
        if covered {
            let shading = Shading {
                light,
                depths,
                camera_position: Vec3::from_array(camera),
                texels,
            };
            let scattered;
            (color, scattered) = crate::fragment::shade(
                &shading,
                material,
                &surface,
                &attributes,
                fragment_meta[soa_offset::<2>(sample, 1)] != 0,
                &mut sites,
            );
            mark(&mut sites, Site::Output);
            store_trace(&mut trace, sample * SLOTS, sites, material_index);
            if bounce {
                // Each lane owns its pixel's albedo.
                unsafe {
                    *albedo.get_unchecked_mut(sample) = scattered.to_array();
                }
            }
        }
        // Transpose each row's colors so its lanes store contiguous floats in host RGB order.
        let valid = (width as usize - (x - lane % 8)).min(8) * 3;
        for plane in 0..3 {
            let slot = plane * 8 + lane % 8;
            let source = (lane / 8 * 8 + slot / 3) as u32;
            let r = cuda_device::warp::shuffle_f32(color.x, source);
            let g = cuda_device::warp::shuffle_f32(color.y, source);
            let b = cuda_device::warp::shuffle_f32(color.z, source);
            let value = match slot % 3 {
                0 => r,
                1 => g,
                _ => b,
            };
            if y < height as usize && slot < valid {
                // Each lane writes a distinct scalar of an in-bounds row.
                unsafe {
                    colors
                        .as_mut_ptr()
                        .cast::<f32>()
                        .add((sample - lane % 8) * 3 + slot)
                        .write(value);
                }
            }
        }
    }

    /// Adds cosine-weighted diffuse light from `BOUNCE_RAYS` paths of `BOUNCES` bounces. Lanes keep
    /// shading's 8x4 pixel blocks, so a warp shades each ray's hits in launch order, depth by
    /// depth, across whatever materials they reach.
    #[kernel]
    // 24 resident 32-lane blocks, the most an SM holds, cap registers at 80: traversal gains
    // more from the occupancy than the decoders lose to spills.
    #[cuda_device::launch_bounds(32, 24)]
    pub fn bounce(
        width: u32,
        height: u32,
        fragments: &[f32],
        fragment_meta: &[u32],
        winners: &[DeviceAtomicU64],
        vertices: &[f32],
        triangles: &[CudaTriangle],
        primitives: &[CudaPrimitive],
        nodes: &[Node],
        order: &[u32],
        materials: &[Material],
        texels: &[Texel],
        light: Light,
        depths: &[f32],
        albedo: &[[f32; 3]],
        mut colors: DisjointSlice<[f32; 3]>,
        mut trace: DisjointSlice<u32>,
    ) {
        // Lanes of empty samples and misses stay to sample the default material, since the
        // tensor-core decoder runs warp-wide.
        let (_, _, _, sample, covered) =
            block_pixel(thread::index_1d().get(), width, height, winners);
        // Each lane owns its pixel's color and trace slots.
        let slots = sample * SLOTS;
        let (origin, normal) = if covered {
            let attributes = FragmentAttributes::load_soa(fragments, sample);
            let normal = if fragment_meta[soa_offset::<2>(sample, 1)] != 0 {
                attributes.normal
            } else {
                -attributes.normal
            };
            let position = attributes.world_position;
            (
                position + normal * (1e-4 * (1.0 + position.abs().max_element())),
                normal,
            )
        } else {
            (Vec3::ZERO, Vec3::Z)
        };
        let corners = |triangle: u32| {
            triangles[triangle as usize].vertices.map(|v| {
                Vec3::from_array(std::array::from_fn(|c| {
                    vertices[soa_offset::<{ ClipVertex::COMPONENTS }>(v as usize, 4 + c)]
                }))
            })
        };
        let mut radiance = Vec3::ZERO;
        for ray in 0..BOUNCE_RAYS {
            // A path continues from each hit, weighted by its diffuse albedo; a miss ends it.
            let (mut origin, mut normal, mut weight, mut alive) =
                (origin, normal, Vec3::ONE, covered);
            for depth in 0..BOUNCES {
                let path = ray * BOUNCES + depth;
                // Each depth draws from its own range, so first bounces match any `BOUNCES`.
                let seed = (sample as u32)
                    .wrapping_mul(BOUNCE_RAYS)
                    .wrapping_add(ray)
                    .wrapping_add((depth * BOUNCE_RAYS).wrapping_mul(width * height));
                let direction = cosine_direction(normal, seed);
                let mut sites = 0;
                mark(&mut sites, Site::Traverse);
                // Slots are recomputed at each store: keeping one live across traversal cost 15%.
                let hit = if alive {
                    bvh::closest_hit(nodes, order, origin, direction, corners)
                } else {
                    None
                };
                let (material_index, hit_attributes, front_facing) = match &hit {
                    Some(hit) => {
                        mark(&mut sites, Site::Hit);
                        let triangle = triangles[hit.triangle as usize];
                        let primitive = primitives[triangle.primitive as usize];
                        let [a, b, c] = triangle
                            .vertices
                            .map(|v| ClipVertex::load_soa(vertices, v as usize));
                        let [wa, wb, wc] = hit.barycentric;
                        let hit_attributes = FragmentAttributes {
                            world_position: origin + direction * hit.t,
                            // Hits read level zero; ray cones could select a mip level.
                            uv: UvSample::point(a.uv * wa + b.uv * wb + c.uv * wc),
                            normal: (a.normal * wa + b.normal * wb + c.normal * wc)
                                .normalize_or_zero(),
                            tangent: a.tangent * wa + b.tangent * wb + c.tangent * wc,
                        };
                        let geometric = (b.world_position - a.world_position)
                            .cross(c.world_position - a.world_position);
                        let front_facing = (geometric.dot(direction) < 0.0) != primitive.mirrored;
                        (primitive.material_index, hit_attributes, front_facing)
                    }
                    None => (
                        materials.len() as u32 - 1,
                        FragmentAttributes::default(),
                        false,
                    ),
                };
                let material = &materials[material_index as usize];
                let surface = material.sample(texels, hit_attributes.uv, &mut sites);
                if hit.is_some() {
                    let shading = Shading {
                        light,
                        depths,
                        camera_position: origin,
                        texels,
                    };
                    let (shaded, scattered) = crate::fragment::shade(
                        &shading,
                        material,
                        &surface,
                        &hit_attributes,
                        front_facing,
                        &mut sites,
                    );
                    radiance += weight * shaded;
                    if depth + 1 < BOUNCES {
                        // The next bounce leaves the side the ray arrived from.
                        normal = if front_facing {
                            hit_attributes.normal
                        } else {
                            -hit_attributes.normal
                        };
                        let position = hit_attributes.world_position;
                        origin = position + normal * (1e-4 * (1.0 + position.abs().max_element()));
                        weight *= scattered;
                    }
                }
                alive = hit.is_some();
                if covered {
                    let traced = hit.as_ref().map_or(u32::MAX, |_| material_index);
                    store_trace(&mut trace, slots + 1 + path as usize, sites, traced);
                }
            }
        }
        if covered {
            // Cosine-weighted sampling cancels the Lambertian cosine and 1/pi.
            let indirect = Vec3::from_array(albedo[sample]) * (radiance / BOUNCE_RAYS as f32);
            let color = unsafe { colors.get_unchecked_mut(sample) };
            color[0] += indirect.x;
            color[1] += indirect.y;
            color[2] += indirect.z;
        }
    }

    /// Resident blocks per SM for shading, capping registers at 65536 / (32 blocks): 128 for the
    /// tensor-core decoder, which otherwise takes 206, and 96 for dp4a, the fastest for each.
    const SHADE_BLOCKS: u32 = if crate::ntc::MMA { 16 } else { 20 };

    /// With `TRACE`, stores a lane's region bits and material; the caller owns `slot`.
    #[inline(always)]
    fn store_trace(trace: &mut DisjointSlice<u32>, slot: usize, sites: u32, material: u32) {
        if TRACE {
            unsafe {
                *trace.get_unchecked_mut(2 * slot) = sites;
                *trace.get_unchecked_mut(2 * slot + 1) = material;
            }
        }
    }

    fn cosine_direction(normal: Vec3, seed: u32) -> Vec3 {
        let mut state = seed;
        let u = uniform(&mut state);
        let v = uniform(&mut state);
        let (sin, cos) = (std::f32::consts::TAU * v).sin_cos();
        let radius = u.sqrt();
        // Branchless orthonormal basis (Duff et al. 2017).
        let sign = 1f32.copysign(normal.z);
        let a = -1.0 / (sign + normal.z);
        let b = normal.x * normal.y * a;
        let tangent = Vec3::new(
            1.0 + sign * normal.x * normal.x * a,
            sign * b,
            -sign * normal.x,
        );
        let bitangent = Vec3::new(b, sign + normal.y * normal.y * a, -normal.y);
        (tangent * (radius * cos) + bitangent * (radius * sin) + normal * (1.0 - u).sqrt())
            .normalize_or_zero()
    }
}

fn elapsed(events: &(CudaEvent, CudaEvent)) -> io::Result<Duration> {
    Ok(Duration::from_secs_f32(
        events.0.elapsed_ms(&events.1).map_err(io::Error::other)? * 0.001,
    ))
}

/// One 32-lane block per 8x4 pixel tile.
fn tile_launch(width: u32, height: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (width.div_ceil(8) * height.div_ceil(4), 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    }
}
