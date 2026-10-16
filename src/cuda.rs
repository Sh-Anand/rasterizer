use std::{
    collections::HashMap,
    io,
    sync::Arc,
    time::{Duration, Instant},
};

use cuda_core::{
    CudaContext, CudaStream, DeviceBuffer,
    simt::{LaunchConfig, memory},
};
use cuda_device::cuda_module;
use glam::{Vec3, Vec4};

use crate::{
    lighting::{Lighting, RenderLight, visible_lights},
    material::Material,
    pipeline::{Fragment, FragmentStage},
    shadow::{LightShadow, ShadowMap},
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
    /// Allocation and host-to-device copies; scene resources are uploaded on refresh only.
    pub upload: Duration,
    /// CUDA event interval around dispatch, including any cold launch delay.
    pub dispatch: Duration,
    pub readback: Duration,
}

pub struct CudaFragment {
    stream: Arc<CudaStream>,
    module: kernels::LoadedModule,
    timings: CudaTimings,
    resource_key: Option<(usize, usize, usize, usize)>,
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
        let context = CudaContext::new(device).map_err(io::Error::other)?;
        Ok(Self {
            stream: context.default_stream(),
            module: kernels::load(&context).map_err(io::Error::other)?,
            timings: CudaTimings::default(),
            resource_key: None,
            fragments: Buffer::new(&context),
            materials: Buffer::new(&context),
            pixels: Buffer::new(&context),
            lights: Buffer::new(&context),
            maps: Buffer::new(&context),
            depths: Buffer::new(&context),
            colors: Buffer::new(&context),
        })
    }

    pub fn timings(&self) -> &CudaTimings {
        &self.timings
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
        self.timings.packing = started.elapsed();
        let started = Instant::now();
        // Packed descriptors contain offsets; pixel and depth slices contain values only.
        unsafe {
            self.materials.upload(&[&materials])?;
            self.pixels.upload(&pixels)?;
            self.lights.upload(&[&lights])?;
            self.maps.upload(&[&maps])?;
            self.depths.upload(&depths)?;
        }
        self.timings.upload = started.elapsed();
        self.resource_key = Some(key);
        Ok(())
    }
}

impl FragmentStage for CudaFragment {
    fn invalidate_resources(&mut self) {
        self.resource_key = None;
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
        self.stream
            .context()
            .bind_to_thread()
            .map_err(io::Error::other)?;
        self.upload_resources(materials, lighting.lights)?;
        let started = Instant::now();
        let stream = &self.stream;
        // Fragments contain only scalar/vector attributes and indices.
        unsafe { self.fragments.upload(&[fragments])? };
        let colors = self.colors.resize(fragments.len())?;
        self.timings.upload += started.elapsed();
        let start = stream.record_event(Some(0)).map_err(io::Error::other)?;
        // One lane per output; all indexed buffers live through the synchronous readback.
        unsafe {
            self.module
                .shade(
                    stream,
                    LaunchConfig::for_num_elems(count),
                    &self.fragments.device,
                    &self.materials.device,
                    &self.pixels.device,
                    &self.lights.device,
                    &self.maps.device,
                    &self.depths.device,
                    lighting.camera_position.to_array(),
                    colors,
                )
                .map_err(io::Error::other)?;
        }
        let end = stream.record_event(Some(0)).map_err(io::Error::other)?;
        end.synchronize().map_err(io::Error::other)?;
        self.timings.dispatch =
            Duration::from_secs_f32(start.elapsed_ms(&end).map_err(io::Error::other)? * 0.001);
        let started = Instant::now();
        let colors = colors.to_host_vec(stream).map_err(io::Error::other)?;
        let colors = colors.into_iter().map(Vec3::from_array).collect();
        self.timings.readback = started.elapsed();
        Ok(colors)
    }
}

#[cuda_module]
mod kernels {
    use super::*;
    use cuda_device::{DisjointSlice, kernel, thread};

    #[kernel]
    pub fn shade(
        fragments: &[Fragment],
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
