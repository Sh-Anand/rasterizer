use std::{collections::HashMap, io, sync::Arc, time::Duration};

use cuda_core::{
    CudaContext, CudaEvent, CudaStream, DeviceBuffer,
    simt::{LaunchConfig, memory},
};
use cuda_device::{atomic::DeviceAtomicU64, cuda_module};
use glam::{Mat4, Vec2, Vec3, Vec4};

use crate::{
    config::ShadowConfig,
    framebuffer::{DepthBuffer, Framebuffer, RenderOutput},
    geometry::{
        ClipVertex, PreparedScene, PreparedVertex, VertexTransform, check_view_projection,
    },
    lighting::{Light, RenderLight, visible_lights},
    material::Material,
    pipeline::{FragmentAttributes, Pipeline, PipelineStats, ShadowProjection},
    rasterization::{
        Polygon, ScreenVertex, Triangle, UvGradients, clip_triangle, interpolate_attributes,
        outcode, project_vertex,
    },
    shadow::{LightShadow, ShadowMap, prepare_lights},
    texture::{Texture, UvSample},
};

type PackedMaterial = Material<Texture<[usize; 2]>>;
type PackedLight = RenderLight<LightShadow<[usize; 2]>>;
type PackedMap = ShadowMap<[usize; 2]>;

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

soa!(PreparedVertex, 18,
    position: Vec3 = 0, uv: Vec2 = 3, normal_uv: Vec2 = 5,
    metallic_roughness_uv: Vec2 = 7, emissive_uv: Vec2 = 9,
    normal: Vec3 = 11, tangent: Vec4 = 14,
);
soa!(ClipVertex, 22,
    position: Vec4 = 0, world_position: Vec3 = 4,
    uv: Vec2 = 7, normal_uv: Vec2 = 9,
    metallic_roughness_uv: Vec2 = 11, emissive_uv: Vec2 = 13,
    normal: Vec3 = 15, tangent: Vec4 = 18,
);
soa!(FragmentAttributes, 34,
    world_position: Vec3 = 0, uv: UvSample = 3, normal_uv: UvSample = 9,
    metallic_roughness_uv: UvSample = 15, emissive_uv: UvSample = 21,
    normal: Vec3 = 27, tangent: Vec4 = 30,
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
    resident_key: Option<(usize, usize, usize, usize)>,
    fragments: Buffer<f32>,
    fragment_meta: Buffer<u32>,
    materials: Buffer<PackedMaterial>,
    pixels: Buffer<Vec4>,
    lights: Buffer<PackedLight>,
    maps: Buffer<PackedMap>,
    depths: Buffer<f32>,
    colors: Buffer<[f32; 3]>,
}

impl CudaFragment {
    pub fn with_device(device: Arc<CudaDevice>) -> Self {
        let context = device.stream.context();
        Self {
            resident_key: None,
            fragments: Buffer::new(context),
            fragment_meta: Buffer::new(context),
            materials: Buffer::new(context),
            pixels: Buffer::new(context),
            lights: Buffer::new(context),
            maps: Buffer::new(context),
            depths: Buffer::new(context),
            colors: Buffer::new(context),
            device,
        }
    }

    fn upload_materials(&mut self, materials: &[Material]) -> io::Result<()> {
        u32::try_from(materials.len()).map_err(io::Error::other)?;
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
        unsafe {
            self.materials.upload(&[&materials])?;
            self.pixels.upload(&pixels)?;
        }
        Ok(())
    }

    fn shade_device(
        &mut self,
        count: u32,
        winners: &DeviceBuffer<DeviceAtomicU64>,
        camera_position: Vec3,
    ) -> io::Result<(CudaEvent, CudaEvent)> {
        let block = 32;
        let start = self.device.event()?;
        // One lane per output; buffers live through the synchronous readback.
        unsafe {
            self.device.module.shade(
                &self.device.stream,
                LaunchConfig {
                    grid_dim: (count.div_ceil(block), 1, 1),
                    block_dim: (block, 1, 1),
                    shared_mem_bytes: 0,
                },
                &self.fragments.device,
                &self.fragment_meta.device,
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

    fn read_colors(&self, count: usize) -> io::Result<Vec<Vec3>> {
        let mut colors = self
            .colors
            .device
            .to_host_vec(&self.device.stream)
            .map_err(io::Error::other)?;
        colors.truncate(count);
        Ok(colors.into_iter().map(Vec3::from_array).collect())
    }
}

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
    resource_key: Option<(usize, usize)>,
    vertices: Buffer<f32>,
    primitive_indices: Buffer<u32>,
    transforms: Buffer<VertexTransform>,
    output: Buffer<f32>,
    triangles: Buffer<CudaTriangle>,
    primitives: Buffer<CudaPrimitive>,
    error: Buffer<DeviceAtomicU64>,
}

impl CudaVertex {
    pub fn with_device(device: Arc<CudaDevice>) -> Self {
        let context = device.stream.context();
        Self {
            resource_key: None,
            vertices: Buffer::new(context),
            primitive_indices: Buffer::new(context),
            transforms: Buffer::new(context),
            output: Buffer::new(context),
            triangles: Buffer::new(context),
            primitives: Buffer::new(context),
            error: Buffer::new(context),
            device,
        }
    }

    fn transform_device(
        &mut self,
        scene: &PreparedScene,
        view_projection: Mat4,
    ) -> io::Result<(CudaEvent, CudaEvent)> {
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
            self.resource_key = None;
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
            let stride = padded_count(count as usize);
            let mut vertices = vec![0.0; stride * PreparedVertex::COMPONENTS];
            let mut indices = Vec::with_capacity(count as usize);
            let mut primitives = Vec::with_capacity(scene.primitives.len());
            for (i, primitive) in scene.primitives.iter().enumerate() {
                let i = u32::try_from(i).map_err(io::Error::other)?;
                let base = indices.len() as u32;
                for (index, vertex) in primitive.vertices.iter().enumerate() {
                    vertex.store_soa(base as usize + index, |offset, v| vertices[offset] = v);
                }
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
            // Geometry and descriptors contain values and indices, never host pointers.
            unsafe {
                self.vertices.upload(&[&vertices])?;
                self.primitive_indices.upload(&[&indices])?;
                self.triangles.upload(&[&triangles])?;
                self.primitives.upload(&[&primitives])?;
            }
            self.resource_key = Some(key);
        }
        let transforms: Vec<_> = scene
            .primitives
            .iter()
            .map(|primitive| VertexTransform::new(primitive, view_projection))
            .collect();
        // Transform descriptors contain only matrices and a scalar sign.
        unsafe { self.transforms.upload(&[&transforms])? };
        let output = self
            .output
            .resize(padded_count(count as usize) * ClipVertex::COMPONENTS)?;
        self.error.resize(1)?;
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

    fn project_faces(
        vertices: &[f32],
        triangle: &CudaTriangle,
        width: u32,
        height: u32,
        mut emit: impl FnMut(&[ScreenVertex; 3], u32),
    ) -> Result<u32, crate::rasterization::RasterError> {
        let codes = triangle.vertices.map(|i| {
            outcode(Vec4::from_array(std::array::from_fn(|c| {
                vertices[soa_offset::<22>(i as usize, c)]
            })))
        });
        if codes[0] & codes[1] & codes[2] != 0 {
            return Ok(0);
        }
        // Most triangles avoid the clipping scratch entirely.
        if codes[0] | codes[1] | codes[2] == 0 {
            let [a, b, c] = triangle.vertices.map(|i| {
                project_vertex(
                    ClipVertex::load_soa(vertices, i as usize),
                    width as f32,
                    height as f32,
                )
            });
            emit(&[a?, b?, c?], 0);
            return Ok(1);
        }
        let mut polygon = Polygon::default();
        clip_triangle(
            |i| ClipVertex::load_soa(vertices, i as usize),
            triangle.vertices,
            codes,
            &mut polygon,
        )?;
        if polygon.len < 3 {
            return Ok(0);
        }
        let first = project_vertex(polygon.vertices[0].0, width as f32, height as f32)?;
        let mut previous = project_vertex(polygon.vertices[1].0, width as f32, height as f32)?;
        for fan in 0..polygon.len - 2 {
            let current = project_vertex(polygon.vertices[fan + 2].0, width as f32, height as f32)?;
            emit(&[first, previous, current], fan as u32);
            previous = current;
        }
        Ok((polygon.len - 2) as u32)
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
        vertices: &[f32],
        triangles: &[CudaTriangle],
        primitives: &[CudaPrimitive],
        materials: &[PackedMaterial],
        width: u32,
        height: u32,
        shadow: Option<ShadowProjection>,
        winners: &[DeviceAtomicU64],
        stats: &[DeviceAtomicU64],
        error: &[DeviceAtomicU64],
        mut large: DisjointSlice<[u32; 2]>,
    ) {
        let index = thread::index_1d().get();
        let mut counts = [0u32; 2];
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
            match project_faces(vertices, triangle, width, height, |points, fan| {
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
                Ok(triangles) => counts[1] = triangles,
                Err(reason) => {
                    error[0].fetch_min(
                        ((triangle.source_index as u64) << 32) | (10 + reason as u64),
                        AtomicOrdering::Relaxed,
                    );
                }
            }
            if parts != 0 {
                let slot = stats[3].fetch_add(1, AtomicOrdering::Relaxed) as usize;
                // Each input triangle reserves at most one unique slot.
                unsafe {
                    *large.get_unchecked_mut(slot) = [index as u32, parts as u32];
                }
            }
        }
        for i in 0..2 {
            count_warp(&stats[i], counts[i]);
        }
    }

    #[kernel]
    pub fn rasterize_large(
        vertices: &[f32],
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
                    JOB[0] = stats[4].fetch_add(1, AtomicOrdering::Relaxed);
                    LENGTH[0] = 0;
                }
            }
            thread::sync_threads();
            let job = unsafe { JOB[0] as usize };
            if job >= stats[3].load(AtomicOrdering::Relaxed) as usize * MAX_PARTS {
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
        vertices: &[f32],
        triangles: &[CudaTriangle],
        primitives: &[CudaPrimitive],
        width: u32,
        height: u32,
        winners: &[DeviceAtomicU64],
        stats: &[DeviceAtomicU64],
        error: &[DeviceAtomicU64],
        mut fragments: DisjointSlice<f32>,
        mut fragment_meta: DisjointSlice<u32>,
        mut depth: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d().get();
        let mut shaded = 0;
        if let Some(z) = depth.get_mut(thread::index_1d()) {
            *z = f32::INFINITY;
            let key = winners[index].load(AtomicOrdering::Relaxed);
            if key != u64::MAX && error[0].load(AtomicOrdering::Relaxed) == u64::MAX {
                let rank = key as u32;
                let triangle = triangles[(rank / 7) as usize];
                let primitive = primitives[triangle.primitive as usize];
                if let Err(reason) =
                    project_faces(vertices, &triangle, width, height, |points, fan| {
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
                        let attributes =
                            interpolate_attributes(points, barycentric, &UvGradients::new(points));
                        // Each lane owns its index in every component plane.
                        attributes.store_soa(index, |offset, v| unsafe {
                            *fragments.get_unchecked_mut(offset) = v;
                        });
                        unsafe {
                            *fragment_meta.get_unchecked_mut(soa_offset::<2>(index, 0)) =
                                primitive.material_index.map_or(u32::MAX, |i| i as u32);
                            *fragment_meta.get_unchecked_mut(soa_offset::<2>(index, 1)) =
                                u32::from(setup.front_facing);
                        }
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
            if key != u64::MAX && shaded == 0 {
                winners[index].store(u64::MAX, AtomicOrdering::Relaxed);
            }
        }
        count_warp(&stats[2], shaded);
    }

    #[kernel]
    pub fn transform(
        vertices: &[f32],
        primitive_indices: &[u32],
        transforms: &[VertexTransform],
        error: &[DeviceAtomicU64],
        mut output: DisjointSlice<f32>,
    ) {
        let offset = thread::index_1d().get();
        if let Some(&primitive) = primitive_indices.get(offset) {
            let transform = &transforms[primitive as usize];
            let vertex = transform.apply(&PreparedVertex::load_soa(vertices, offset));
            let invalid = vertex.invalid(transform.flat_shaded);
            if invalid != 0 {
                error[0].fetch_min(
                    ((offset as u64) << 32) | u64::from(invalid),
                    AtomicOrdering::Relaxed,
                );
            }
            // Each lane owns its index in every component plane.
            vertex.store_soa(offset, |index, v| unsafe {
                *output.get_unchecked_mut(index) = v;
            });
        }
    }

    #[kernel]
    pub fn shade(
        fragments: &[f32],
        fragment_meta: &[u32],
        winners: &[DeviceAtomicU64],
        materials: &[PackedMaterial],
        pixels: &[Vec4],
        lights: &[PackedLight],
        maps: &[PackedMap],
        depths: &[f32],
        camera: [f32; 3],
        mut colors: DisjointSlice<[f32; 3]>,
    ) {
        let offset = thread::index_1d().get();
        let stride = colors.len();
        if offset < stride {
            let mut color = Vec3::ZERO;
            if winners
                .get(offset)
                .is_some_and(|winner| winner.load(AtomicOrdering::Relaxed) != u64::MAX)
            {
                let attributes = FragmentAttributes::load_soa(fragments, offset);
                let material_index = fragment_meta[soa_offset::<2>(offset, 0)];
                let material = if material_index == u32::MAX {
                    Material::default()
                } else {
                    materials[material_index as usize].map_textures(|texture| {
                        let [start, end] = texture.pixels;
                        texture.with_pixels(&pixels[start..end])
                    })
                };
                let position = attributes.world_position;
                let samples = visible_lights(lights, position, |shadow, position| {
                    let [start, end] = *shadow.maps();
                    shadow
                        .with_maps(&maps[start..end])
                        .visibility_with(position, |&[start, end]| &depths[start..end])
                });
                color = crate::fragment::shade(
                    Vec3::from_array(camera),
                    &material,
                    &attributes,
                    fragment_meta[soa_offset::<2>(offset, 1)] != 0,
                    samples,
                );
            }
            // Transpose in the warp: contiguous stores, CPU RGB order, no repacking.
            let lane = offset % 32;
            for plane in 0..3 {
                let slot = plane * 32 + lane;
                let source = (slot / 3) as u32;
                let r = cuda_device::warp::shuffle_f32(color.x, source);
                let g = cuda_device::warp::shuffle_f32(color.y, source);
                let b = cuda_device::warp::shuffle_f32(color.z, source);
                let value = match slot % 3 {
                    0 => r,
                    1 => g,
                    _ => b,
                };
                // Padded warps participate fully and each lane writes a distinct scalar.
                unsafe {
                    colors
                        .as_mut_ptr()
                        .cast::<f32>()
                        .add((offset / 32) * 96 + slot)
                        .write(value);
                }
            }
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
    counters: Buffer<DeviceAtomicU64>,
    depth: Buffer<f32>,
    large: Buffer<[u32; 2]>,
}

impl CudaRasterization {
    pub fn with_device(device: Arc<CudaDevice>) -> Self {
        let context = device.stream.context();
        Self {
            winners: Buffer::new(context),
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
        // Coverage, projected triangles, winners, queue length/cursor.
        self.counters.resize(5)?;
        self.large.resize(vertex.triangles.device.len())?;
        if shadow.is_none() {
            fragment
                .fragments
                .resize(padded_count(count as usize) * FragmentAttributes::COMPONENTS)?;
            fragment
                .fragment_meta
                .resize(padded_count(count as usize) * 2)?;
            self.depth.resize(count as usize)?;
        }
        let stream = &self.device.stream;
        let start = self.device.event()?;
        self.winners.fill(255, stream)?;
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
                        &mut fragment.fragment_meta.device,
                        &mut self.depth.device,
                    )
                    .map_err(io::Error::other)?;
            }
        }
        Ok((start, self.device.event()?))
    }
}

impl Pipeline<CudaVertex, CudaRasterization, CudaFragment> {
    /// Invalidate cached geometry and shading resources; prepare_cuda_lights must run again.
    pub fn invalidate_resources(&mut self) {
        self.vertex.resource_key = None;
        self.fragment.resident_key = None;
    }

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
        self.fragment.resident_key = None;
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
        self.fragment.colors.resize(padded_count(count as usize))?;
        let fragment_events = self.fragment.shade_device(
            count,
            &self.rasterization.winners.device,
            camera_position,
        )?;
        let stream = &self.vertex.device.stream;
        let color = self.fragment.read_colors(count as usize)?;
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
        self.stats.vertices_shaded = self.vertex.primitive_indices.device.len();
        self.stats.fragments_shaded = counters[2] as usize;
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
            screen_vertices: None,
            screen_triangles: counters[1] as usize,
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
