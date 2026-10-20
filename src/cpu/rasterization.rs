use std::io;

use rayon::prelude::*;

use crate::{
    framebuffer::DepthBuffer,
    geometry::{ClipPrimitive, ClipVertex},
    material::Material,
    pipeline::{
        Fragment, FragmentAttributes, RasterizationStage, RasterizedScene, ShadowProjection,
    },
    rasterization::{
        Polygon, ScreenVertex, Triangle, UvGradients, clip_triangle, interpolate_attributes,
        outcode, project_vertex, sample_range,
    },
};

pub(super) struct DepthSlice<'a> {
    pub width: u32,
    pub y_start: u32,
    pub values: &'a mut [f32],
}

pub(super) fn rasterize(
    primitive: &ScreenPrimitive,
    triangles: &[[u32; 3]],
    double_sided: bool,
    depth: DepthSlice<'_>,
    shadow: Option<ShadowProjection>,
    mut emit: impl FnMut(usize, &FragmentAttributes, bool),
) -> u64 {
    let DepthSlice {
        width,
        y_start,
        values: depth_buffer,
    } = depth;
    let y_end = y_start + (depth_buffer.len() / width as usize) as u32;
    let first_sample = y_start as usize * width as usize;
    let mut covered_fragments = 0;
    for triangle in triangles {
        let vertices = triangle.map(|index| primitive.vertices[index as usize]);
        let Some(setup) = Triangle::new(&vertices, primitive.mirrored, double_sided, shadow) else {
            continue;
        };
        let gradients = shadow.is_none().then(|| UvGradients::new(&vertices));
        let (xs, ys) = setup.bounds(width, y_start..y_end);
        for y in ys {
            for x in xs.clone() {
                let Some(barycentric) = setup.coverage(x, y) else {
                    continue;
                };
                covered_fragments += 1;
                let Some(depth) = setup.depth(barycentric) else {
                    continue;
                };
                let index = (y - y_start) as usize * width as usize + x as usize;
                if depth >= depth_buffer[index] {
                    continue;
                }
                depth_buffer[index] = depth;
                if let Some(gradients) = &gradients {
                    let attributes = interpolate_attributes(&vertices, barycentric, gradients);
                    emit(first_sample + index, &attributes, setup.front_facing);
                }
            }
        }
    }
    covered_fragments
}

pub struct CpuRasterization;

impl RasterizationStage for CpuRasterization {
    fn rasterize(
        &mut self,
        primitives: &[&ClipPrimitive],
        materials: &[Material],
        depth: &mut DepthBuffer,
        shadow: Option<ShadowProjection>,
    ) -> io::Result<RasterizedScene> {
        let width = depth.width;
        let band_height = (depth.height as usize).div_ceil(rayon::current_num_threads());
        let band_samples = band_height * width as usize;
        let screens = primitives
            .par_iter()
            .map(|primitive| {
                let screen = project(primitive, width, depth.height)?;
                let bins = screen.bins(depth.height, band_height);
                Ok((screen, bins))
            })
            .collect::<io::Result<Vec<_>>>()?;
        let mut output = RasterizedScene::default();
        for (screen, _) in &screens {
            output.screen_vertices += screen.vertices.len();
            output.screen_triangles += screen.triangles.len();
        }
        let default_material = Material::default();
        let bands = depth
            .values
            .par_chunks_mut(band_samples)
            .enumerate()
            .map(|(band, values)| {
                let mut output = RasterizedScene::default();
                let mut slot_for_sample = Vec::new();
                if shadow.is_none() {
                    slot_for_sample
                        .try_reserve_exact(values.len())
                        .map_err(io::Error::other)?;
                    slot_for_sample.resize(values.len(), usize::MAX);
                }
                for (primitive, (screen, bins)) in primitives.iter().zip(&screens) {
                    let material_index = primitive.source.material_index;
                    let material =
                        material_index.map_or(&default_material, |index| &materials[index]);
                    output.covered_fragments += rasterize(
                        screen,
                        &bins[band],
                        material.double_sided,
                        DepthSlice {
                            width,
                            y_start: (band * band_height) as u32,
                            values,
                        },
                        shadow,
                        |sample_index, attributes, front_facing| {
                            let fragment = Fragment {
                                sample_index,
                                material_index,
                                attributes: *attributes,
                                front_facing,
                            };
                            let slot = &mut slot_for_sample[sample_index - band * band_samples];
                            if *slot == usize::MAX {
                                *slot = output.fragments.len();
                                output.fragments.push(fragment);
                            } else {
                                output.fragments[*slot] = fragment;
                            }
                        },
                    );
                }
                Ok(output)
            })
            .collect::<io::Result<Vec<_>>>()?;
        output
            .fragments
            .try_reserve_exact(bands.iter().map(|band| band.fragments.len()).sum())
            .map_err(io::Error::other)?;
        for band in bands {
            output.covered_fragments += band.covered_fragments;
            output.fragments.extend(band.fragments);
        }
        Ok(output)
    }
}

pub(super) fn project(
    primitive: &ClipPrimitive,
    width: u32,
    height: u32,
) -> io::Result<ScreenPrimitive> {
    let mut output = ScreenPrimitive {
        mirrored: primitive.mirrored,
        vertices: Vec::new(),
        triangles: Vec::new(),
    };
    let codes: Vec<_> = primitive
        .vertices
        .iter()
        .map(|v| outcode(v.position))
        .collect();
    let mut projected = vec![u32::MAX; primitive.vertices.len()];
    let mut project = |(vertex, source): (ClipVertex, Option<u32>)| -> io::Result<u32> {
        if let Some(source) = source {
            let index = projected[source as usize];
            if index != u32::MAX {
                return Ok(index);
            }
        }
        let index = u32::try_from(output.vertices.len() + 1)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Too many clipped vertices"))?
            - 1;
        output.vertices.push(
            project_vertex(vertex, width as f32, height as f32)
                .map_err(|message| io::Error::new(io::ErrorKind::InvalidData, message.message()))?,
        );
        if let Some(source) = source {
            projected[source as usize] = index;
        }
        Ok(index)
    };
    let mut polygon = Polygon::default();
    for triangle in &primitive.triangles {
        clip_triangle(
            |i| primitive.vertices[i as usize],
            *triangle,
            triangle.map(|i| codes[i as usize]),
            &mut polygon,
        )
        .map_err(|message| io::Error::new(io::ErrorKind::InvalidData, message.message()))?;
        if polygon.len < 3 {
            continue;
        }
        let first = project(polygon.vertices[0])?;
        let mut previous = project(polygon.vertices[1])?;
        for &vertex in &polygon.vertices[2..polygon.len] {
            let current = project(vertex)?;
            output.triangles.push([first, previous, current]);
            previous = current;
        }
    }
    Ok(output)
}

#[derive(Debug)]
pub(super) struct ScreenPrimitive {
    pub mirrored: bool,
    pub vertices: Vec<ScreenVertex>,
    pub triangles: Vec<[u32; 3]>,
}

impl ScreenPrimitive {
    fn bins(&self, height: u32, band_height: usize) -> Vec<Vec<[u32; 3]>> {
        let mut bins = vec![Vec::new(); (height as usize).div_ceil(band_height)];
        for &triangle in &self.triangles {
            let y = triangle.map(|index| self.vertices[index as usize].position.y);
            let rows = sample_range(
                y[0].min(y[1]).min(y[2]),
                y[0].max(y[1]).max(y[2]),
                0..height,
            );
            if !rows.is_empty() {
                for bin in &mut bins
                    [rows.start as usize / band_height..=(rows.end as usize - 1) / band_height]
                {
                    bin.push(triangle);
                }
            }
        }
        bins
    }
}
