use std::{
    collections::HashMap,
    io,
    sync::Arc,
    time::{Duration, Instant},
};

use cuda_core::{
    CudaContext, CudaEvent, CudaStream, DeviceBuffer, DeviceCopy,
    simt::{LaunchConfig, memory},
};
use cuda_device::{
    atomic::{DeviceAtomicU32, DeviceAtomicU64},
    cuda_module,
};
use glam::{Mat4, Vec3, Vec4};

use crate::{
    config::ShadowConfig,
    framebuffer::{DepthBuffer, Framebuffer, RenderOutput},
    geometry::{
        ClipPrimitive, ClipScene, ClipVertex, PreparedScene, PreparedVertex, VertexTransform,
        check_view_projection,
    },
    lighting::{Light, Lighting, RenderLight, visible_lights},
    material::Material,
    pipeline::{Fragment, FragmentStage, Pipeline, PipelineStats, ShadowProjection, VertexStage},
    rasterization::{
        Polygon, ScreenVertex, Triangle, UvGradients, clip_triangle, interpolate_attributes,
        outcode, project_vertex,
    },
    shadow::{LightShadow, ShadowMap, prepare_lights},
    texture::Texture,
};

type PackedMaterial = Material<Texture<[usize; 2]>>;
type PackedLight = RenderLight<LightShadow<[usize; 2]>>;
type PackedMap = ShadowMap<[usize; 2]>;

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

    // Used only for integer atomic buffers.
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
    unsafe fn upload(&mut self, chunks: &[&[T]]) -> io::Result<()>
    where
        T: Copy,
    {
        let len = chunks
            .iter()
            .try_fold(0usize, |n, chunk| n.checked_add(chunk.len()))
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "CUDA buffer size overflow")
            })?;
        let mut dst = self.resize(len)?.cu_deviceptr();
        for chunk in chunks.iter().filter(|chunk| !chunk.is_empty()) {
            let bytes = std::mem::size_of_val(*chunk);
            // Chunks exactly fill the allocation's visible length; copies finish before return.
            unsafe { memory::memcpy_htod_sync(dst, chunk.as_ptr(), bytes) }
                .map_err(io::Error::other)?;
            dst += bytes as u64;
        }
        Ok(())
    }
}

#[derive(Default, Debug)]
pub struct CudaTimings {
    pub packing: Duration,
    /// Allocation and host-to-device copies.
    pub upload: Duration,
    /// CUDA event interval around dispatch, including any cold launch delay.
    pub dispatch: Duration,
    pub readback: Duration,
}

pub struct CudaDevice {
    stream: Arc<CudaStream>,
    module: kernels::LoadedModule,
}

impl CudaDevice {
    pub fn new(device: usize) -> io::Result<Arc<Self>> {
        let context = CudaContext::new(device).map_err(io::Error::other)?;
        Ok(Arc::new(Self {
            stream: context.default_stream(),
            module: kernels::load(&context).map_err(io::Error::other)?,
        }))
    }

    fn event(&self) -> io::Result<CudaEvent> {
        self.stream.record_event(Some(0)).map_err(io::Error::other)
    }
}

pub struct CudaFragment {
    device: Arc<CudaDevice>,
    timings: CudaTimings,
    resource_key: Option<(usize, usize, usize, usize)>,
    resident_key: Option<(usize, usize, usize, usize)>,
    fragments: Buffer<Fragment>,
    materials: Buffer<PackedMaterial>,
    pixels: Buffer<Vec4>,
    lights: Buffer<PackedLight>,
    maps: Buffer<PackedMap>,
    depths: Buffer<f32>,
    colors: Buffer<[f32; 3]>,
}

impl CudaFragment {
    pub fn new(device: usize) -> io::Result<Self> {
        Ok(Self::with_device(CudaDevice::new(device)?))
    }

    pub fn with_device(device: Arc<CudaDevice>) -> Self {
        let context = device.stream.context();
        Self {
            timings: CudaTimings::default(),
            resource_key: None,
            resident_key: None,
            fragments: Buffer::new(context),
            materials: Buffer::new(context),
            pixels: Buffer::new(context),
            lights: Buffer::new(context),
            maps: Buffer::new(context),
            depths: Buffer::new(context),
            colors: Buffer::new(context),
            device,
        }
    }

    pub fn timings(&self) -> &CudaTimings {
        &self.timings
    }

    fn upload_materials(&mut self, materials: &[Material]) -> io::Result<()> {
        let started = Instant::now();
        let mut pixels = Vec::new();
        let mut pixel_count = 0;
        let mut textures = HashMap::new();
        let materials: Vec<_> = materials
            .iter()
            .map(|material| {
                material.map_textures(|texture| {
                    let range = *textures.entry(texture.pixels.as_ptr()).or_insert_with(|| {
                        let start = pixel_count;
                        pixel_count += texture.pixels.len();
                        pixels.push(texture.pixels.as_ref());
                        [start, pixel_count]
                    });
                    texture.with_pixels(range)
                })
            })
            .collect();
        self.timings.packing += started.elapsed();
        let started = Instant::now();
        unsafe {
            self.materials.upload(&[&materials])?;
            self.pixels.upload(&pixels)?;
        }
        self.timings.upload += started.elapsed();
        Ok(())
    }

    fn upload_resources(
        &mut self,
        materials: &[Material],
        lights: &[RenderLight],
    ) -> io::Result<()> {
        let key = (
            materials.as_ptr() as usize,
            materials.len(),
            lights.as_ptr() as usize,
            lights.len(),
        );
        if self.resource_key == Some(key) {
            return Ok(());
        }
        self.invalidate_resources();
        self.upload_materials(materials)?;
        let started = Instant::now();
        let mut depths = Vec::new();
        let mut depth_count = 0;
        let mut maps = Vec::new();
        let lights: Vec<_> = lights
            .iter()
            .map(|light| RenderLight {
                light: light.light,
                shadow: light.shadow.as_ref().map(|shadow| {
                    let start = maps.len();
                    for map in shadow.maps() {
                        let offset = depth_count;
                        depth_count += map.depth.values.len();
                        depths.push(map.depth.values.as_slice());
                        maps.push(map.with_depth([offset, depth_count]));
                    }
                    shadow.with_maps([start, maps.len()])
                }),
            })
            .collect();
        self.timings.packing += started.elapsed();
        let started = Instant::now();
        // Packed descriptors contain offsets; pixel and depth slices contain values only.
        unsafe {
            self.lights.upload(&[&lights])?;
            self.maps.upload(&[&maps])?;
            self.depths.upload(&depths)?;
        }
        self.timings.upload += started.elapsed();
        self.resource_key = Some(key);
        Ok(())
    }

    fn shade_device(
        &mut self,
        count: u32,
        winners: &DeviceBuffer<DeviceAtomicU64>,
        camera_position: Vec3,
    ) -> io::Result<(CudaEvent, CudaEvent)> {
        let start = self.device.event()?;
        // One lane per output; buffers live through the synchronous readback.
        unsafe {
            self.device.module.shade(
                &self.device.stream,
                LaunchConfig::for_num_elems(count),
                &self.fragments.device,
                winners,
                &self.materials.device,
                &self.pixels.device,
                &self.lights.device,
                &self.maps.device,
                &self.depths.device,
                camera_position.to_array(),
                &mut self.colors.device,
            )
        }
        .map_err(io::Error::other)?;
        Ok((start, self.device.event()?))
    }

    fn read_colors(&self) -> io::Result<Vec<Vec3>> {
        Ok(self
            .colors
            .device
            .to_host_vec(&self.device.stream)
            .map_err(io::Error::other)?
            .into_iter()
            .map(Vec3::from_array)
            .collect())
    }
}

impl FragmentStage for CudaFragment {
    fn invalidate_resources(&mut self) {
        self.resource_key = None;
        self.resident_key = None;
    }

    fn shade(
        &mut self,
        fragments: &[Fragment],
        materials: &[Material],
        lighting: &Lighting<'_>,
    ) -> io::Result<Vec<Vec3>> {
        self.timings = CudaTimings::default();
        if fragments.is_empty() {
            return Ok(Vec::new());
        }
        let count = u32::try_from(fragments.len()).map_err(io::Error::other)?;
        if fragments
            .iter()
            .any(|f| f.material_index.is_some_and(|i| i >= materials.len()))
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Fragment material index is outside the material list",
            ));
        }
        self.device
            .stream
            .context()
            .bind_to_thread()
            .map_err(io::Error::other)?;
        self.upload_resources(materials, lighting.lights)?;
        let started = Instant::now();
        // Fragments contain only scalar/vector attributes and indices.
        unsafe { self.fragments.upload(&[fragments])? };
        self.colors.resize(fragments.len())?;
        self.timings.upload += started.elapsed();
        let events = self.shade_device(
            count,
            &Buffer::<DeviceAtomicU64>::new(self.device.stream.context()).device,
            lighting.camera_position,
        )?;
        events.1.synchronize().map_err(io::Error::other)?;
        self.timings.dispatch = elapsed(&events)?;
        let started = Instant::now();
        let colors = self.read_colors()?;
        self.timings.readback = started.elapsed();
        Ok(colors)
    }
}

// ClipVertex contains only FP32 vectors, including valid zero representations.
unsafe impl DeviceCopy for ClipVertex {}

#[derive(Clone, Copy)]
struct CudaTriangle {
    vertices: [u32; 3],
    primitive: u32,
    source_index: u32,
}
#[derive(Clone, Copy)]
struct CudaPrimitive {
    material_index: Option<usize>,
    mirrored: bool,
}

pub struct CudaVertex {
    device: Arc<CudaDevice>,
    timings: CudaTimings,
    resource_key: Option<(usize, usize)>,
    vertices: Buffer<PreparedVertex>,
    primitive_indices: Buffer<u32>,
    transforms: Buffer<VertexTransform>,
    output: Buffer<ClipVertex>,
    triangles: Buffer<CudaTriangle>,
    primitives: Buffer<CudaPrimitive>,
    error: Buffer<DeviceAtomicU64>,
    readback: Vec<ClipVertex>,
}

impl CudaVertex {
    pub fn new(device: usize) -> io::Result<Self> {
        Ok(Self::with_device(CudaDevice::new(device)?))
    }

    pub fn with_device(device: Arc<CudaDevice>) -> Self {
        let context = device.stream.context();
        Self {
            timings: CudaTimings::default(),
            resource_key: None,
            vertices: Buffer::new(context),
            primitive_indices: Buffer::new(context),
            transforms: Buffer::new(context),
            output: Buffer::new(context),
            triangles: Buffer::new(context),
            primitives: Buffer::new(context),
            error: Buffer::new(context),
            readback: Vec::new(),
            device,
        }
    }

    fn transform_device(
        &mut self,
        scene: &PreparedScene,
        view_projection: Mat4,
    ) -> io::Result<(CudaEvent, CudaEvent)> {
        self.timings = CudaTimings::default();
        check_view_projection(view_projection)?;
        let count = scene
            .primitives
            .iter()
            .try_fold(0u32, |count, primitive| {
                count.checked_add(u32::try_from(primitive.vertices.len()).ok()?)
            })
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Too many CUDA vertices"))?;
        self.device
            .stream
            .context()
            .bind_to_thread()
            .map_err(io::Error::other)?;
        let key = (scene.primitives.as_ptr() as usize, scene.primitives.len());
        if self.resource_key != Some(key) {
            self.invalidate_resources();
            let started = Instant::now();
            let triangle_count = scene
                .primitives
                .iter()
                .try_fold(0u32, |count, primitive| {
                    count.checked_add(u32::try_from(primitive.triangles.len()).ok()?)
                })
                .filter(|&n| n <= u32::MAX / 7)
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "Too many CUDA triangles")
                })?;
            let mut triangles = Vec::with_capacity(triangle_count as usize);
            let mut vertices = Vec::with_capacity(scene.primitives.len());
            let mut indices = Vec::with_capacity(count as usize);
            let mut primitives = Vec::with_capacity(scene.primitives.len());
            for (i, primitive) in scene.primitives.iter().enumerate() {
                let i = u32::try_from(i).map_err(io::Error::other)?;
                let base = indices.len() as u32;
                vertices.push(primitive.vertices.as_slice());
                indices.extend(std::iter::repeat_n(i, primitive.vertices.len()));
                primitives.push(CudaPrimitive {
                    material_index: primitive.source.material_index,
                    mirrored: primitive.mirrored,
                });
                for triangle in &primitive.triangles {
                    if triangle
                        .iter()
                        .any(|&v| v as usize >= primitive.vertices.len())
                    {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            "Triangle vertex index is outside the primitive",
                        ));
                    }
                    triangles.push(CudaTriangle {
                        vertices: triangle.map(|v| base + v),
                        primitive: i,
                        source_index: triangles.len() as u32,
                    });
                }
            }
            // Camera ranks index this stable material order; shadows retain source order.
            triangles.sort_by_key(|t| primitives[t.primitive as usize].material_index);
            self.timings.packing += started.elapsed();
            let started = Instant::now();
            // Geometry and descriptors contain values and indices, never host pointers.
            unsafe {
                self.vertices.upload(&vertices)?;
                self.primitive_indices.upload(&[&indices])?;
                self.triangles.upload(&[&triangles])?;
                self.primitives.upload(&[&primitives])?;
            }
            self.timings.upload += started.elapsed();
            self.resource_key = Some(key);
        }
        let started = Instant::now();
        let transforms: Vec<_> = scene
            .primitives
            .iter()
            .map(|primitive| VertexTransform::new(primitive, view_projection))
            .collect();
        self.timings.packing += started.elapsed();
        let started = Instant::now();
        // Transform descriptors contain only matrices and a scalar sign.
        unsafe { self.transforms.upload(&[&transforms])? };
        let output = self.output.resize(count as usize)?;
        self.error.resize(1)?;
        self.timings.upload += started.elapsed();
        let start = self.device.event()?;
        self.error.fill(255, &self.device.stream)?;
        if count != 0 {
            unsafe {
                self.device
                    .module
                    .transform(
                        &self.device.stream,
                        LaunchConfig::for_num_elems(count),
                        &self.vertices.device,
                        &self.primitive_indices.device,
                        &self.transforms.device,
                        &self.error.device,
                        output,
                    )
                    .map_err(io::Error::other)?;
            }
        }
        Ok((start, self.device.event()?))
    }

    pub fn timings(&self) -> &CudaTimings {
        &self.timings
    }
}

impl VertexStage for CudaVertex {
    fn invalidate_resources(&mut self) {
        self.resource_key = None;
    }

    fn transform(&mut self, scene: &PreparedScene, view_projection: Mat4) -> io::Result<ClipScene> {
        let events = self.transform_device(scene, view_projection)?;
        events.1.synchronize().map_err(io::Error::other)?;
        let count = self.output.device.len();
        let output = &self.output.device;
        let started = Instant::now();
        self.readback.resize(count, ClipVertex::default());
        if count != 0 {
            output
                .copy_to_host(&self.device.stream, &mut self.readback)
                .map_err(io::Error::other)?;
        }
        let mut vertices = self.readback.iter().copied();
        let primitives = scene
            .primitives
            .iter()
            .map(|primitive| {
                Ok(ClipPrimitive {
                    source: primitive.source,
                    mirrored: primitive.mirrored,
                    vertices: vertices
                        .by_ref()
                        .take(primitive.vertices.len())
                        .map(|vertex| vertex.checked(primitive))
                        .collect::<io::Result<_>>()?,
                    triangles: primitive.triangles.clone(),
                })
            })
            .collect::<io::Result<_>>()?;
        self.timings.readback = started.elapsed();
        self.timings.dispatch = elapsed(&events)?;
        Ok(ClipScene {
            scene_index: scene.scene_index,
            primitives,
        })
    }
}

#[cuda_module]
mod kernels {
    use super::*;
    use cuda_device::{DisjointSlice, SharedArray, atomic::AtomicOrdering, kernel, thread};

    #[kernel]
    pub fn resolve_depth(
        winners: &[DeviceAtomicU64],
        offset: usize,
        mut depth: DisjointSlice<f32>,
    ) {
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
                *depth.get_unchecked_mut(offset + index) = if key == u64::MAX {
                    f32::INFINITY
                } else {
                    f32::from_bits(bits)
                };
            }
        }
    }

    const SMALL_SAMPLES: usize = 256;
    const MAX_PARTS: usize = 16;

    // Call with every lane active, including lanes beyond the input length.
    fn count_warp(counter: &DeviceAtomicU64, mut count: u32) {
        // The pinned compiler misclassifies integer redux in test builds.
        for offset in [16, 8, 4, 2, 1] {
            count = count.wrapping_add(cuda_device::warp::shuffle_down(count, offset));
        }
        if cuda_device::warp::lane_id() == 0 && count != 0 {
            counter.fetch_add(u64::from(count), AtomicOrdering::Relaxed);
        }
    }

    fn mark_vertex(used: &[DeviceAtomicU32], source: Option<u32>) -> u32 {
        if used.is_empty() {
            return 0;
        }
        match source {
            Some(i) => {
                let bit = 1 << (i % 32);
                u32::from(used[i as usize / 32].fetch_or(bit, AtomicOrdering::Relaxed) & bit == 0)
            }
            None => 1,
        }
    }

    fn project_faces(
        vertices: &[ClipVertex],
        triangle: &CudaTriangle,
        width: u32,
        height: u32,
        used: &[DeviceAtomicU32],
        mut emit: impl FnMut(&[ScreenVertex; 3], u32),
    ) -> Result<[u32; 2], crate::rasterization::RasterError> {
        let codes = triangle
            .vertices
            .map(|i| outcode(vertices[i as usize].position));
        if codes[0] & codes[1] & codes[2] != 0 {
            return Ok([0, 0]);
        }
        // Most triangles avoid the clipping scratch entirely.
        if codes[0] | codes[1] | codes[2] == 0 {
            let [a, b, c] = triangle
                .vertices
                .map(|i| project_vertex(vertices[i as usize], width as f32, height as f32));
            let points = [a?, b?, c?];
            let count = triangle
                .vertices
                .iter()
                .map(|&i| mark_vertex(used, Some(i)))
                .sum();
            emit(&points, 0);
            return Ok([count, 1]);
        }
        let mut polygon = Polygon::default();
        clip_triangle(vertices, triangle.vertices, codes, &mut polygon)?;
        if polygon.len < 3 {
            return Ok([0, 0]);
        }
        let first = project_vertex(polygon.vertices[0].0, width as f32, height as f32)?;
        let mut previous = project_vertex(polygon.vertices[1].0, width as f32, height as f32)?;
        for fan in 0..polygon.len - 2 {
            let current = project_vertex(polygon.vertices[fan + 2].0, width as f32, height as f32)?;
            emit(&[first, previous, current], fan as u32);
            previous = current;
        }
        let count = polygon.vertices[..polygon.len]
            .iter()
            .map(|&(_, source)| mark_vertex(used, source))
            .sum();
        Ok([count, (polygon.len - 2) as u32])
    }

    fn raster_samples(
        setup: &Triangle,
        width: u32,
        height: u32,
        rank: u32,
        first: usize,
        stride: usize,
        winners: &[DeviceAtomicU64],
    ) -> u32 {
        let (xs, ys) = setup.bounds(width, 0..height);
        let columns = (xs.end - xs.start) as usize;
        let count = columns * (ys.end - ys.start) as usize;
        let mut sample = first;
        let mut covered = 0;
        while sample < count {
            let x = xs.start + (sample % columns) as u32;
            let y = ys.start + (sample / columns) as u32;
            sample += stride;
            if let Some(barycentric) = setup.coverage(x, y) {
                covered += 1;
                if let Some(z) = setup.depth(barycentric) {
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
        covered
    }

    #[kernel]
    pub fn rasterize_small(
        vertices: &[ClipVertex],
        triangles: &[CudaTriangle],
        primitives: &[CudaPrimitive],
        materials: &[PackedMaterial],
        width: u32,
        height: u32,
        shadow: Option<ShadowProjection>,
        winners: &[DeviceAtomicU64],
        used: &[DeviceAtomicU32],
        stats: &[DeviceAtomicU64],
        error: &[DeviceAtomicU64],
        mut large: DisjointSlice<[u32; 2]>,
    ) {
        let index = thread::index_1d().get();
        let mut counts = [0u32; 3];
        if let Some(triangle) = triangles.get(index)
            && error[0].load(AtomicOrdering::Relaxed) == u64::MAX
        {
            let primitive = primitives[triangle.primitive as usize];
            let double_sided = primitive
                .material_index
                .is_some_and(|i| materials[i].double_sided);
            let rank = if shadow.is_some() {
                triangle.source_index
            } else {
                index as u32
            } * 7;
            let mut parts = 0;
            match project_faces(vertices, triangle, width, height, used, |points, fan| {
                if let Some(setup) = Triangle::new(points, primitive.mirrored, double_sided, shadow)
                {
                    let (xs, ys) = setup.bounds(width, 0..height);
                    let count = (xs.end - xs.start) as usize * (ys.end - ys.start) as usize;
                    if count > SMALL_SAMPLES {
                        parts = parts.max(count.div_ceil(2048).min(MAX_PARTS));
                    } else {
                        counts[0] +=
                            raster_samples(&setup, width, height, rank + fan, 0, 1, winners);
                    }
                }
            }) {
                Ok([vertices, triangles]) => {
                    counts[1] = vertices;
                    counts[2] = triangles;
                }
                Err(reason) => {
                    error[0].fetch_min(
                        ((triangle.source_index as u64) << 32) | (10 + reason as u64),
                        AtomicOrdering::Relaxed,
                    );
                }
            }
            if parts != 0 {
                let slot = stats[4].fetch_add(1, AtomicOrdering::Relaxed) as usize;
                // Each input triangle reserves at most one unique slot.
                unsafe {
                    *large.get_unchecked_mut(slot) = [index as u32, parts as u32];
                }
            }
        }
        for i in 0..3 {
            count_warp(&stats[i], counts[i]);
        }
    }

    #[kernel]
    pub fn rasterize_large(
        vertices: &[ClipVertex],
        triangles: &[CudaTriangle],
        primitives: &[CudaPrimitive],
        materials: &[PackedMaterial],
        width: u32,
        height: u32,
        shadow: Option<ShadowProjection>,
        winners: &[DeviceAtomicU64],
        stats: &[DeviceAtomicU64],
        error: &[DeviceAtomicU64],
        large: &[[u32; 2]],
    ) {
        static mut JOB: SharedArray<u64, 1> = SharedArray::UNINIT;
        static mut LENGTH: SharedArray<u32, 1> = SharedArray::UNINIT;
        static mut SETUPS: SharedArray<Option<Triangle>, 7> = SharedArray::UNINIT;
        let lane = thread::threadIdx_x() as usize;
        loop {
            if lane == 0 {
                unsafe {
                    JOB[0] = stats[5].fetch_add(1, AtomicOrdering::Relaxed);
                    LENGTH[0] = 0;
                }
            }
            thread::sync_threads();
            let job = unsafe { JOB[0] as usize };
            if job >= stats[4].load(AtomicOrdering::Relaxed) as usize * MAX_PARTS {
                return;
            }
            let [index, parts] = large[job / MAX_PARTS];
            if job % MAX_PARTS >= parts as usize {
                thread::sync_threads();
                continue;
            }
            let index = index as usize;
            let triangle = triangles[index];
            if lane == 0 && error[0].load(AtomicOrdering::Relaxed) == u64::MAX {
                let primitive = primitives[triangle.primitive as usize];
                let double_sided = primitive
                    .material_index
                    .is_some_and(|i| materials[i].double_sided);
                if let Err(reason) = project_faces(
                    vertices,
                    &triangle,
                    width,
                    height,
                    &[],
                    |points, fan| unsafe {
                        SETUPS[fan as usize] =
                            Triangle::new(points, primitive.mirrored, double_sided, shadow);
                        LENGTH[0] = fan + 1;
                    },
                ) {
                    error[0].fetch_min(
                        ((triangle.source_index as u64) << 32) | (10 + reason as u64),
                        AtomicOrdering::Relaxed,
                    );
                }
            }
            thread::sync_threads();
            let mut covered = 0;
            for fan in 0..unsafe { LENGTH[0] } {
                if let Some(setup) = unsafe { SETUPS[fan as usize] } {
                    let (xs, ys) = setup.bounds(width, 0..height);
                    let count = (xs.end - xs.start) as usize * (ys.end - ys.start) as usize;
                    if count > SMALL_SAMPLES {
                        let rank = if shadow.is_some() {
                            triangle.source_index
                        } else {
                            index as u32
                        } * 7
                            + fan;
                        covered += raster_samples(
                            &setup,
                            width,
                            height,
                            rank,
                            (job % MAX_PARTS) * thread::blockDim_x() as usize + lane,
                            parts as usize * thread::blockDim_x() as usize,
                            winners,
                        );
                    }
                }
            }
            count_warp(&stats[0], covered);
            thread::sync_threads();
        }
    }

    #[kernel]
    pub fn interpolate(
        vertices: &[ClipVertex],
        triangles: &[CudaTriangle],
        primitives: &[CudaPrimitive],
        width: u32,
        height: u32,
        winners: &[DeviceAtomicU64],
        stats: &[DeviceAtomicU64],
        error: &[DeviceAtomicU64],
        mut fragments: DisjointSlice<Fragment>,
        mut depth: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d().get();
        let mut shaded = 0;
        if let Some(fragment) = fragments.get_mut(thread::index_1d()) {
            *fragment = Fragment::default();
            let z = depth.get_mut(thread::index_1d()).unwrap();
            *z = f32::INFINITY;
            let key = winners[index].load(AtomicOrdering::Relaxed);
            if key != u64::MAX && error[0].load(AtomicOrdering::Relaxed) == u64::MAX {
                let rank = key as u32;
                let triangle = triangles[(rank / 7) as usize];
                let primitive = primitives[triangle.primitive as usize];
                if let Err(reason) =
                    project_faces(vertices, &triangle, width, height, &[], |points, fan| {
                        if fan != rank % 7 {
                            return;
                        }
                        let Some(setup) = Triangle::new(points, primitive.mirrored, true, None)
                        else {
                            return;
                        };
                        let Some(barycentric) = setup.coverage(
                            (index % width as usize) as u32,
                            (index / width as usize) as u32,
                        ) else {
                            return;
                        };
                        *fragment = Fragment {
                            sample_index: index,
                            material_index: primitive.material_index,
                            attributes: interpolate_attributes(
                                points,
                                barycentric,
                                &UvGradients::new(points),
                            ),
                            front_facing: setup.front_facing,
                        };
                        *z = setup.depth(barycentric).unwrap();
                        shaded = 1;
                    })
                {
                    error[0].fetch_min(
                        ((triangle.source_index as u64) << 32) | (10 + reason as u64),
                        AtomicOrdering::Relaxed,
                    );
                }
            }
        }
        count_warp(&stats[3], shaded);
    }

    #[kernel]
    pub fn transform(
        vertices: &[PreparedVertex],
        primitive_indices: &[u32],
        transforms: &[VertexTransform],
        error: &[DeviceAtomicU64],
        mut output: DisjointSlice<ClipVertex>,
    ) {
        let index = thread::index_1d();
        let offset = index.get();
        if let Some(vertex) = output.get_mut(index) {
            let transform = &transforms[primitive_indices[offset] as usize];
            *vertex = transform.apply(&vertices[offset]);
            let invalid = vertex.invalid(transform.flat_shaded);
            if invalid != 0 {
                error[0].fetch_min(
                    ((offset as u64) << 32) | u64::from(invalid),
                    cuda_device::atomic::AtomicOrdering::Relaxed,
                );
            }
        }
    }

    #[kernel]
    pub fn shade(
        fragments: &[Fragment],
        winners: &[DeviceAtomicU64],
        materials: &[PackedMaterial],
        pixels: &[Vec4],
        lights: &[PackedLight],
        maps: &[PackedMap],
        depths: &[f32],
        camera: [f32; 3],
        mut colors: DisjointSlice<[f32; 3]>,
    ) {
        let index = thread::index_1d();
        let offset = index.get();
        if let Some(color) = colors.get_mut(index) {
            if !winners.is_empty()
                && winners[offset].load(cuda_device::atomic::AtomicOrdering::Relaxed) == u64::MAX
            {
                *color = [0.0; 3];
                return;
            }
            let fragment = &fragments[offset];
            let material = fragment
                .material_index
                .map_or_else(Material::default, |index| {
                    materials[index].map_textures(|texture| {
                        let [start, end] = texture.pixels;
                        texture.with_pixels(&pixels[start..end])
                    })
                });
            let position = fragment.attributes.world_position;
            let samples = visible_lights(lights, position, |shadow, position| {
                let [start, end] = *shadow.maps();
                shadow
                    .with_maps(&maps[start..end])
                    .visibility_with(position, |&[start, end]| &depths[start..end])
            });
            *color = crate::fragment::shade(
                Vec3::from_array(camera),
                &material,
                &fragment.attributes,
                fragment.front_facing,
                samples,
            )
            .to_array();
        }
    }
}

fn elapsed(events: &(CudaEvent, CudaEvent)) -> io::Result<Duration> {
    Ok(Duration::from_secs_f32(
        events.0.elapsed_ms(&events.1).map_err(io::Error::other)? * 0.001,
    ))
}

pub struct CudaRasterization {
    device: Arc<CudaDevice>,
    winners: Buffer<DeviceAtomicU64>,
    used: Buffer<DeviceAtomicU32>,
    counters: Buffer<DeviceAtomicU64>,
    depth: Buffer<f32>,
    large: Buffer<[u32; 2]>,
}

impl CudaRasterization {
    pub fn with_device(device: Arc<CudaDevice>) -> Self {
        let context = device.stream.context();
        Self {
            winners: Buffer::new(context),
            used: Buffer::new(context),
            counters: Buffer::new(context),
            depth: Buffer::new(context),
            large: Buffer::new(context),
            device,
        }
    }

    fn rasterize_device(
        &mut self,
        vertex: &CudaVertex,
        fragment: &mut CudaFragment,
        width: u32,
        height: u32,
        shadow: Option<ShadowProjection>,
    ) -> io::Result<(CudaEvent, CudaEvent)> {
        let count = sample_count(width, height)?;
        self.winners.resize(count as usize)?;
        self.used.resize(vertex.output.device.len().div_ceil(32))?;
        // Coverage, projected vertices/triangles, winners, queue length/cursor.
        self.counters.resize(6)?;
        self.large.resize(vertex.triangles.device.len())?;
        if shadow.is_none() {
            fragment.fragments.resize(count as usize)?;
            self.depth.resize(count as usize)?;
        }
        let stream = &self.device.stream;
        let start = self.device.event()?;
        self.winners.fill(255, stream)?;
        self.used.fill(0, stream)?;
        self.counters.fill(0, stream)?;
        if !vertex.triangles.device.is_empty() {
            unsafe {
                self.device
                    .module
                    .rasterize_small(
                        stream,
                        LaunchConfig::for_num_elems(vertex.triangles.device.len() as u32),
                        &vertex.output.device,
                        &vertex.triangles.device,
                        &vertex.primitives.device,
                        &fragment.materials.device,
                        width,
                        height,
                        shadow,
                        &self.winners.device,
                        &self.used.device,
                        &self.counters.device,
                        &vertex.error.device,
                        &mut self.large.device,
                    )
                    .map_err(io::Error::other)?;
                self.device
                    .module
                    .rasterize_large(
                        stream,
                        LaunchConfig {
                            grid_dim: (256, 1, 1),
                            block_dim: (128, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        &vertex.output.device,
                        &vertex.triangles.device,
                        &vertex.primitives.device,
                        &fragment.materials.device,
                        width,
                        height,
                        shadow,
                        &self.winners.device,
                        &self.counters.device,
                        &vertex.error.device,
                        &self.large.device,
                    )
                    .map_err(io::Error::other)?;
            }
        }
        if shadow.is_none() {
            unsafe {
                self.device
                    .module
                    .interpolate(
                        stream,
                        LaunchConfig::for_num_elems(count),
                        &vertex.output.device,
                        &vertex.triangles.device,
                        &vertex.primitives.device,
                        width,
                        height,
                        &self.winners.device,
                        &self.counters.device,
                        &vertex.error.device,
                        &mut fragment.fragments.device,
                        &mut self.depth.device,
                    )
                    .map_err(io::Error::other)?;
            }
        }
        Ok((start, self.device.event()?))
    }
}

impl Pipeline<CudaVertex, CudaRasterization, CudaFragment> {
    fn check_cuda(&self, scene: &PreparedScene, materials: &[Material]) -> io::Result<()> {
        if !Arc::ptr_eq(&self.vertex.device, &self.rasterization.device)
            || !Arc::ptr_eq(&self.vertex.device, &self.fragment.device)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "CUDA stages must share one CudaDevice",
            ));
        }
        if scene.primitives.iter().any(|p| {
            p.source
                .material_index
                .is_some_and(|i| i >= materials.len())
        }) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Primitive material index is outside the material list",
            ));
        }
        self.vertex
            .device
            .stream
            .context()
            .bind_to_thread()
            .map_err(io::Error::other)
    }

    /// Prepare textures, lights and GPU shadow maps; repeat after resource invalidation.
    pub fn prepare_cuda_lights(
        &mut self,
        scene: &PreparedScene,
        materials: &[Material],
        lights: Vec<Light>,
        config: &ShadowConfig,
        unshadowed_lights: &[Light],
    ) -> io::Result<()> {
        self.check_cuda(scene, materials)?;
        self.fragment.invalidate_resources();
        self.fragment.timings = CudaTimings::default();
        self.fragment.upload_materials(materials)?;
        let mut depth_count = 0usize;
        let positions = if config.enabled && !lights.is_empty() {
            scene.world_positions()?
        } else {
            Vec::new()
        };
        let layout = prepare_lights(lights, &positions, config, |map| {
            let start = depth_count;
            depth_count = depth_count
                .checked_add(sample_count(map.depth.width, map.depth.height)? as usize)
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "CUDA shadow size overflow")
                })?;
            Ok([start, depth_count])
        })?;
        let mut maps = Vec::new();
        let mut lights: Vec<PackedLight> = layout
            .iter()
            .map(|light| RenderLight {
                light: light.light,
                shadow: light.shadow.as_ref().map(|shadow| {
                    let start = maps.len();
                    maps.extend_from_slice(shadow.maps());
                    shadow.with_maps([start, maps.len()])
                }),
            })
            .collect();
        lights.extend(unshadowed_lights.iter().map(|&light| RenderLight {
            light,
            shadow: None,
        }));
        self.fragment.depths.resize(depth_count)?;
        unsafe {
            self.fragment.maps.upload(&[&maps])?;
            self.fragment.lights.upload(&[&lights])?;
        }
        for map in &maps {
            self.vertex.transform_device(scene, map.view_projection)?;
            self.rasterization.rasterize_device(
                &self.vertex,
                &mut self.fragment,
                map.depth.width,
                map.depth.height,
                Some(map.projection),
            )?;
            unsafe {
                self.vertex.device.module.resolve_depth(
                    &self.vertex.device.stream,
                    LaunchConfig::for_num_elems(sample_count(map.depth.width, map.depth.height)?),
                    &self.rasterization.winners.device,
                    map.depth.values[0],
                    &mut self.fragment.depths.device,
                )
            }
            .map_err(io::Error::other)?;
            self.vertex.check_error(scene)?;
        }
        self.fragment.resident_key = Some(resident_key(scene, materials));
        Ok(())
    }

    /// Keep all intermediate camera and shadow data on the device.
    pub fn render_cuda(
        &mut self,
        scene: &PreparedScene,
        view_projection: Mat4,
        materials: &[Material],
        width: u32,
        height: u32,
        camera_position: Vec3,
    ) -> io::Result<RenderOutput> {
        self.check_cuda(scene, materials)?;
        let count = sample_count(width, height)?;
        if !camera_position.is_finite() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Non-finite camera position",
            ));
        }
        if self.fragment.resident_key != Some(resident_key(scene, materials)) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Call prepare_cuda_lights after changing or invalidating scene resources",
            ));
        }
        self.stats = PipelineStats::default();
        let vertex_events = self.vertex.transform_device(scene, view_projection)?;
        let raster_events = self.rasterization.rasterize_device(
            &self.vertex,
            &mut self.fragment,
            width,
            height,
            None,
        )?;
        self.fragment.colors.resize(count as usize)?;
        let fragment_events = self.fragment.shade_device(
            count,
            &self.rasterization.winners.device,
            camera_position,
        )?;
        let stream = &self.vertex.device.stream;
        let color = self.fragment.read_colors()?;
        let depth = self
            .rasterization
            .depth
            .device
            .to_host_vec(stream)
            .map_err(io::Error::other)?;
        self.vertex.check_error(scene)?;
        let counters = read_counters(&self.rasterization.counters.device, stream)?;
        self.stats.vertex_time = elapsed(&vertex_events)?;
        self.stats.rasterization_time = elapsed(&raster_events)?;
        self.stats.fragment_time = elapsed(&fragment_events)?;
        self.stats.vertices_shaded = self.vertex.output.device.len();
        self.stats.fragments_shaded = counters[3] as usize;
        Ok(RenderOutput {
            framebuffer: Framebuffer {
                color,
                depth: DepthBuffer {
                    width,
                    height,
                    values: depth,
                },
            },
            covered_fragments: counters[0],
            screen_vertices: counters[1] as usize,
            screen_triangles: counters[2] as usize,
        })
    }
}

fn sample_count(width: u32, height: u32) -> io::Result<u32> {
    width
        .checked_mul(height)
        .filter(|&n| n != 0)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "CUDA framebuffer dimensions must be nonzero and fit u32",
            )
        })
}

fn resident_key(scene: &PreparedScene, materials: &[Material]) -> (usize, usize, usize, usize) {
    (
        scene.primitives.as_ptr() as usize,
        scene.primitives.len(),
        materials.as_ptr() as usize,
        materials.len(),
    )
}

fn read_counters(
    buffer: &DeviceBuffer<DeviceAtomicU64>,
    stream: &CudaStream,
) -> io::Result<Vec<u64>> {
    let mut values = vec![0u64; buffer.len()];
    unsafe {
        memory::memcpy_dtoh_async(
            values.as_mut_ptr(),
            buffer.cu_deviceptr(),
            values.len() * size_of::<u64>(),
            stream.cu_stream(),
        )
    }
    .map_err(io::Error::other)?;
    stream.synchronize().map_err(io::Error::other)?;
    Ok(values)
}

impl CudaVertex {
    fn check_error(&self, scene: &PreparedScene) -> io::Result<()> {
        let error = read_counters(&self.error.device, &self.device.stream)?[0];
        if error == u64::MAX {
            return Ok(());
        }
        let code = error as u32;
        let mut index = (error >> 32) as usize;
        let primitive = scene
            .primitives
            .iter()
            .find(|p| {
                let count = if code < 10 {
                    p.vertices.len()
                } else {
                    p.triangles.len()
                };
                if index < count {
                    true
                } else {
                    index -= count;
                    false
                }
            })
            .ok_or_else(|| io::Error::other("Invalid CUDA error index"))?;
        let message = match code {
            1 => "Non-finite vertex position",
            2 => "Invalid vertex normal",
            11 => crate::rasterization::RasterError::ClipOverflow.message(),
            12 => crate::rasterization::RasterError::InvalidProjection.message(),
            _ => crate::rasterization::RasterError::ProjectionOverflow.message(),
        };
        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Node {}, mesh {}, primitive {}: {message}",
                primitive.source.node_index,
                primitive.source.mesh_index,
                primitive.source.primitive_index
            ),
        ))
    }
}
