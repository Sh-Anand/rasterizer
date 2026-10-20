use std::{io, sync::Arc, time::Duration};

use cuda_core::{
    CudaContext, CudaEvent, CudaStream, DeviceBuffer,
    simt::{LaunchConfig, memory},
};
use cuda_device::{atomic::DeviceAtomicU64, cuda_module};
use glam::{Mat4, Vec2, Vec3, Vec4};

use crate::{
    config::ShadowConfig,
    framebuffer::{DepthBuffer, Framebuffer, RenderOutput},
    geometry::{ClipVertex, PreparedScene, PreparedVertex, VertexTransform, check_view_projection},
    lighting::{Light, RenderLight},
    material::Material,
    pipeline::{FragmentAttributes, Pipeline, PipelineStats, Shading, ShadowProjection},
    rasterization::{
        Polygon, ScreenVertex, Triangle, UvGradients, clip_triangle, interpolate_attributes,
        outcode, project_vertex,
    },
    shadow::{ShadowMap, prepare_lights},
    texture::{Texel, UvSample},
};

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
    materials: Buffer<Material>,
    texels: Buffer<Texel>,
    lights: Buffer<RenderLight>,
    maps: Buffer<ShadowMap>,
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
            texels: Buffer::new(context),
            lights: Buffer::new(context),
            maps: Buffer::new(context),
            depths: Buffer::new(context),
            colors: Buffer::new(context),
            device,
        }
    }

    fn shade_device(
        &mut self,
        width: u32,
        height: u32,
        winners: &DeviceBuffer<DeviceAtomicU64>,
        camera_position: Vec3,
    ) -> io::Result<(CudaEvent, CudaEvent)> {
        let start = self.device.event()?;
        // One warp per 8x4 pixel block; buffers live through the synchronous readback.
        unsafe {
            self.device.module.shade(
                &self.device.stream,
                LaunchConfig {
                    grid_dim: (width.div_ceil(8) * height.div_ceil(4), 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                },
                width,
                height,
                &self.fragments.device,
                &self.fragment_meta.device,
                winners,
                &self.materials.device,
                &self.texels.device,
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
}
#[derive(Clone, Copy)]
struct CudaPrimitive {
    /// The material count selects the default material appended after the scene's.
    material_index: u32,
    mirrored: bool,
    double_sided: bool,
}

pub struct CudaVertex {
    device: Arc<CudaDevice>,
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

    /// Material indices must be checked against the material list.
    fn upload_geometry(&mut self, scene: &PreparedScene, materials: &[Material]) -> io::Result<()> {
        let count = scene
            .primitives
            .iter()
            .try_fold(0u32, |count, primitive| {
                count.checked_add(u32::try_from(primitive.vertices.len()).ok()?)
            })
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Too many CUDA vertices"))?;
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
        let mut vertices = vec![0.0; padded_count(count as usize) * PreparedVertex::COMPONENTS];
        let mut indices = Vec::with_capacity(count as usize);
        let mut primitives = Vec::with_capacity(scene.primitives.len());
        for (i, primitive) in scene.primitives.iter().enumerate() {
            let i = u32::try_from(i).map_err(io::Error::other)?;
            let base = indices.len() as u32;
            for (index, vertex) in primitive.vertices.iter().enumerate() {
                vertex.store_soa(base as usize + index, |offset, v| vertices[offset] = v);
            }
            indices.extend(std::iter::repeat_n(i, primitive.vertices.len()));
            let material_index = primitive.source.material_index;
            primitives.push(CudaPrimitive {
                material_index: material_index.unwrap_or(materials.len()) as u32,
                mirrored: primitive.mirrored,
                double_sided: material_index.is_some_and(|i| materials[i].double_sided),
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
                });
            }
        }
        // Ranks index this stable material order, with the default material first.
        triangles.sort_by_key(|t| scene.primitives[t.primitive as usize].source.material_index);
        // Geometry and descriptors contain values and indices, never host pointers.
        unsafe {
            self.vertices.upload(&vertices)?;
            self.primitive_indices.upload(&indices)?;
            self.triangles.upload(&triangles)?;
            self.primitives.upload(&primitives)?;
        }
        Ok(())
    }

    fn transform_device(
        &mut self,
        scene: &PreparedScene,
        view_projection: Mat4,
    ) -> io::Result<(CudaEvent, CudaEvent)> {
        check_view_projection(view_projection)?;
        let count = self.primitive_indices.device.len();
        let transforms: Vec<_> = scene
            .primitives
            .iter()
            .map(|primitive| VertexTransform::new(primitive, view_projection))
            .collect();
        // Transform descriptors contain only matrices and a scalar sign.
        unsafe { self.transforms.upload(&transforms)? };
        let output = self
            .output
            .resize(padded_count(count) * ClipVertex::COMPONENTS)?;
        self.error.resize(1)?;
        let start = self.device.event()?;
        self.error.fill(255, &self.device.stream)?;
        if count != 0 {
            unsafe {
                self.device
                    .module
                    .transform(
                        &self.device.stream,
                        LaunchConfig::for_num_elems(count as u32),
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
            let rank = index as u32 * 7;
            let mut parts = 0;
            match project_faces(vertices, triangle, width, height, |points, fan| {
                if let Some(setup) =
                    Triangle::new(points, primitive.mirrored, primitive.double_sided, shadow)
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
                        ((triangle.primitive as u64) << 32) | (10 + reason as u64),
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
                if let Err(reason) =
                    project_faces(vertices, &triangle, width, height, |points, fan| unsafe {
                        SETUPS[fan as usize] = Triangle::new(
                            points,
                            primitive.mirrored,
                            primitive.double_sided,
                            shadow,
                        );
                        LENGTH[0] = fan + 1;
                    })
                {
                    error[0].fetch_min(
                        ((triangle.primitive as u64) << 32) | (10 + reason as u64),
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
                        covered += raster_samples(
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
                                primitive.material_index;
                            *fragment_meta.get_unchecked_mut(soa_offset::<2>(index, 1)) =
                                u32::from(setup.front_facing);
                        }
                        *z = setup.depth(barycentric).unwrap();
                        shaded = 1;
                    })
                {
                    error[0].fetch_min(
                        ((triangle.primitive as u64) << 32) | (10 + reason as u64),
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
                    ((primitive as u64) << 32) | u64::from(invalid),
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
        width: u32,
        height: u32,
        fragments: &[f32],
        fragment_meta: &[u32],
        winners: &[DeviceAtomicU64],
        materials: &[Material],
        texels: &[Texel],
        lights: &[RenderLight],
        maps: &[ShadowMap],
        depths: &[f32],
        camera: [f32; 3],
        mut colors: DisjointSlice<[f32; 3]>,
    ) {
        // Each warp shades an 8x4 pixel block, one 8-pixel row per group of 8 lanes.
        let index = thread::index_1d().get();
        let (block, lane) = (index / 32, index % 32);
        let blocks = (width as usize).div_ceil(8);
        let row_start = block % blocks * 8;
        let x = row_start + lane % 8;
        let y = block / blocks * 4 + lane / 8;
        let sample = y * width as usize + x;
        let mut color = Vec3::ZERO;
        if x < width as usize
            && y < height as usize
            && winners[sample].load(AtomicOrdering::Relaxed) != u64::MAX
        {
            let shading = Shading {
                texels,
                lights,
                maps,
                depths,
                camera_position: Vec3::from_array(camera),
            };
            color = crate::fragment::shade(
                &shading,
                &materials[fragment_meta[soa_offset::<2>(sample, 0)] as usize],
                &FragmentAttributes::load_soa(fragments, sample),
                fragment_meta[soa_offset::<2>(sample, 1)] != 0,
            );
        }
        // Transpose each row's colors so its lanes store contiguous floats in CPU RGB order.
        let valid = (width as usize - row_start).min(8) * 3;
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
    /// Invalidate uploaded resources; prepare_cuda_lights must run again.
    pub fn invalidate_resources(&mut self) {
        self.fragment.resident_key = None;
    }

    fn check_cuda(&self) -> io::Result<()> {
        if !Arc::ptr_eq(&self.vertex.device, &self.rasterization.device)
            || !Arc::ptr_eq(&self.vertex.device, &self.fragment.device)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "CUDA stages must share one CudaDevice",
            ));
        }
        self.vertex
            .device
            .stream
            .context()
            .bind_to_thread()
            .map_err(io::Error::other)
    }

    /// Upload geometry, textures, lights and GPU shadow maps; repeat after scene changes.
    pub fn prepare_cuda_lights(
        &mut self,
        scene: &PreparedScene,
        materials: &[Material],
        texels: &[Texel],
        lights: Vec<Light>,
        config: &ShadowConfig,
        unshadowed_lights: &[Light],
    ) -> io::Result<()> {
        self.check_cuda()?;
        u32::try_from(materials.len()).map_err(io::Error::other)?;
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
        self.fragment.resident_key = None;
        self.vertex.upload_geometry(scene, materials)?;
        let positions = if config.enabled && !lights.is_empty() {
            scene.world_positions()?
        } else {
            Vec::new()
        };
        let (mut lights, maps) = prepare_lights(lights, &positions, config)?;
        lights.extend(unshadowed_lights.iter().map(|&light| RenderLight {
            light,
            shadow: None,
        }));
        self.fragment
            .depths
            .resize(maps.last().map_or(0, ShadowMap::end))?;
        // The default material follows the scene materials, so every index reads device memory.
        let table: Vec<_> = materials
            .iter()
            .copied()
            .chain([Material::default()])
            .collect();
        // Materials, lights and maps hold values and array offsets, never host pointers.
        unsafe {
            self.fragment.materials.upload(&table)?;
            self.fragment.texels.upload(texels)?;
            self.fragment.lights.upload(&lights)?;
            self.fragment.maps.upload(&maps)?;
        }
        for map in &maps {
            self.vertex.transform_device(scene, map.view_projection)?;
            self.rasterization.rasterize_device(
                &self.vertex,
                &mut self.fragment,
                map.width,
                map.height,
                Some(map.projection),
            )?;
            unsafe {
                self.vertex.device.module.resolve_depth(
                    &self.vertex.device.stream,
                    LaunchConfig::for_num_elems(sample_count(map.width, map.height)?),
                    &self.rasterization.winners.device,
                    map.offset,
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
        self.check_cuda()?;
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
            width,
            height,
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
        let primitive = scene
            .primitives
            .get((error >> 32) as usize)
            .ok_or_else(|| io::Error::other("Invalid CUDA error primitive"))?;
        let message = match error as u32 {
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
