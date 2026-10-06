use std::{io, sync::Mutex, time::Instant};

use glam::{Vec2, Vec3, camera::rh};
use rayon::prelude::*;
use xatlas_rs_v2::{ChartOptions, IndexData, MeshData, MeshDecl, PackOptions, Xatlas};

use crate::{
    asset::GltfAsset,
    backend::cpu,
    config::{BakeConfig, Config},
    coverage::rasterize_triangle,
    lighting::collect_lights,
    lightmap::{BakedLighting, Lightmap, MeshLayout, cache_key, invalid},
    material::{Material, load_materials},
    vertex::ClipScene,
};

const PADDING: u32 = 2;

#[derive(Clone, Copy)]
struct Sample {
    position: Vec3,
    normal: Vec3,
    geometric_normal: Vec3,
    double_sided: bool,
}

pub fn bake(
    scene: &mut ClipScene,
    asset: &GltfAsset,
    config: &Config,
) -> io::Result<BakedLighting> {
    let settings = &config.bake;
    if !(8..=4096).contains(&settings.resolution)
        || !(2..=256).contains(&settings.hemicube_resolution)
        || !settings.hemicube_resolution.is_multiple_of(2)
        || !settings.bias.is_finite()
        || settings.bias <= 0.0
        || settings.bias >= 0.01
    {
        return Err(invalid(
            "Bake needs resolution 8..4096, even hemicube_resolution 2..256, and 0 < bias < 0.01",
        ));
    }
    let key = cache_key(asset, config)?;
    let materials = load_materials(asset)?;
    eprintln!("Preparing direct-light shadows...");
    let lights = cpu::prepare_lights(
        collect_lights(scene, &materials, &config.light)?,
        scene,
        &materials,
        &config.shadow,
    )?;
    eprintln!("Generating lightmap UVs...");
    let (layouts, width, height, owners) = unwrap(scene, settings.resolution)?;
    let mut baked = BakedLighting {
        key,
        layouts,
        lightmap: Lightmap::new(width, height)?,
    };
    baked.apply(scene)?;
    let samples = surface_samples(scene, &materials, width, height);
    let valid: Vec<_> = samples.iter().map(Option::is_some).collect();
    let count = valid.iter().filter(|&&v| v).count();
    if count == 0 {
        return Ok(baked);
    }
    eprintln!("Baking {width}x{height} lightmap ({count} surface samples)...");
    for (i, sample) in samples.iter().enumerate() {
        if let Some(sample) = sample {
            baked.lightmap.front[i] =
                cpu::direct_lighting(&lights, sample.position, sample.normal).to_array();
            if sample.double_sided {
                baked.lightmap.back[i] =
                    cpu::direct_lighting(&lights, sample.position, -sample.normal).to_array();
            }
        }
    }
    drop(lights);
    dilate(&mut baked.lightmap, &valid, &owners);

    let mut indirect = bake_indirect(scene, &materials, &baked.lightmap, &samples, settings)?;
    dilate(&mut indirect, &valid, &owners);
    for (dst, src) in baked
        .lightmap
        .front
        .iter_mut()
        .chain(&mut baked.lightmap.back)
        .flatten()
        .zip(indirect.front.iter().chain(&indirect.back).flatten())
    {
        *dst += src;
    }
    Ok(baked)
}

fn surface_samples(
    scene: &ClipScene,
    materials: &[Material<'_>],
    width: u32,
    height: u32,
) -> Vec<Option<Sample>> {
    let size = Vec2::new(width as f32, height as f32);
    let mut samples = vec![None; width as usize * height as usize];
    let default_material = Material::default();
    for primitive in &scene.primitives {
        let material = primitive
            .source
            .material_index
            .map_or(&default_material, |i| &materials[i]);
        for triangle in &primitive.triangles {
            let v = triangle.indices.map(|i| primitive.vertices[i as usize]);
            let [a, b, c] = v.map(|v| v.world_position.as_dvec3());
            let mut geometric_normal = (b - a).cross(c - a).normalize_or_zero().as_vec3();
            if primitive.mirrored {
                geometric_normal = -geometric_normal;
            }
            if geometric_normal == Vec3::ZERO {
                continue;
            }
            let sample = |weights: [f32; 3]| {
                let position: Vec3 = v
                    .iter()
                    .zip(weights)
                    .map(|(v, w)| v.world_position * w)
                    .sum();
                let normal: Vec3 = v.iter().zip(weights).map(|(v, w)| v.normal * w).sum();
                let uv: Vec2 = v.iter().zip(weights).map(|(v, w)| v.uv * w).sum();
                if material
                    .alpha_cutoff
                    .is_some_and(|cutoff| material.base_color(uv).w < cutoff)
                {
                    return None;
                }
                Some(Sample {
                    position,
                    normal: normal.try_normalize().unwrap_or(geometric_normal),
                    geometric_normal,
                    double_sided: material.double_sided,
                })
            };
            let mut covered = false;
            rasterize_triangle(v.map(|v| v.lightmap_uv * size), width, height, |fragment| {
                let [x, y] = fragment.pixel;
                if let Some(sample) = sample(fragment.barycentric) {
                    samples[(y * width + x) as usize] = Some(sample);
                    covered = true;
                }
            });
            // Seed sub-texel triangles at their centroid; ordinary coverage takes priority.
            if !covered {
                let uv = v.iter().map(|v| v.lightmap_uv).sum::<Vec2>() / 3.0 * size;
                let x = (uv.x as u32).min(width - 1);
                let y = (uv.y as u32).min(height - 1);
                let index = (y * width + x) as usize;
                if samples[index].is_none() {
                    samples[index] = sample([1.0 / 3.0; 3]);
                }
            }
        }
    }
    samples
}

fn bake_indirect(
    scene: &ClipScene,
    materials: &[Material<'_>],
    direct: &Lightmap,
    samples: &[Option<Sample>],
    settings: &BakeConfig,
) -> io::Result<Lightmap> {
    let mut indirect = Lightmap::new(direct.width, direct.height)?;
    let [min, max] = scene.primitives.iter().flat_map(|p| &p.vertices).fold(
        [Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)],
        |[min, max], v| [min.min(v.world_position), max.max(v.world_position)],
    );
    let diagonal = (max - min).length();
    if !diagonal.is_finite() || diagonal <= 0.0 {
        return Err(invalid("Invalid bake scene bounds"));
    }
    let offset = diagonal * settings.bias;
    let projection = rh::proj::opengl::perspective(
        std::f32::consts::FRAC_PI_2,
        1.0,
        offset * 0.5,
        diagonal * 1.01,
    );
    let weights = hemicube_weights(settings.hemicube_resolution);
    let capture_scene = cpu::CaptureScene::new(scene);
    let progress = Mutex::new((0, Instant::now()));
    let count = samples.iter().flatten().count();
    indirect
        .front
        .par_iter_mut()
        .zip(&mut indirect.back)
        .zip(samples)
        .try_for_each(|((front, back), sample)| -> io::Result<()> {
            let Some(sample) = sample else {
                return Ok(());
            };
            for (sign, output) in [(1.0, front), (-1.0, back)] {
                if sign < 0.0 && !sample.double_sided {
                    continue;
                }
                let position = sample.position + sample.geometric_normal * (sign * offset);
                let mut incoming = Vec3::ZERO;
                for ((direction, up), weights) in hemicube_axes(sample.normal * sign)
                    .into_iter()
                    .zip(&weights)
                {
                    let view = rh::view::look_to_mat4(position, direction, up);
                    let color = capture_scene.render(
                        materials,
                        direct,
                        projection * view,
                        settings.hemicube_resolution,
                    )?;
                    incoming += color.iter().zip(weights).map(|(c, w)| c * *w).sum::<Vec3>();
                }
                if !incoming.is_finite() {
                    return Err(invalid("Non-finite indirect lighting"));
                }
                *output = incoming.to_array();
            }
            let (done, last_progress) = &mut *progress.lock().unwrap();
            *done += 1;
            if last_progress.elapsed().as_secs() >= 5 || *done == count {
                eprintln!("Indirect lighting: {done}/{count}");
                *last_progress = Instant::now();
            }
            Ok(())
        })?;
    Ok(indirect)
}

fn unwrap(scene: &ClipScene, resolution: u32) -> io::Result<(Vec<MeshLayout>, u32, u32, Vec<u32>)> {
    let positions: Vec<Vec<f32>> = scene
        .primitives
        .iter()
        .map(|p| {
            p.vertices
                .iter()
                .flat_map(|v| v.world_position.to_array())
                .collect()
        })
        .collect();
    let indices: Vec<Vec<u32>> = scene
        .primitives
        .iter()
        .map(|p| p.triangles.iter().flat_map(|t| t.indices).collect())
        .collect();
    if indices.iter().any(Vec::is_empty) {
        return Err(invalid("Cannot unwrap an empty mesh primitive"));
    }
    let declarations: Vec<_> = positions
        .iter()
        .zip(&indices)
        .map(|(p, i)| MeshDecl {
            vertex_position_data: MeshData::Contiguous(p),
            index_data: Some(IndexData::U32(i)),
            ..MeshDecl::default()
        })
        .collect();
    let area: f64 = scene
        .primitives
        .iter()
        .flat_map(|p| {
            p.triangles.iter().map(|t| {
                let [a, b, c] = t
                    .indices
                    .map(|i| p.vertices[i as usize].world_position.as_dvec3());
                (b - a).cross(c - a).length() * 0.5
            })
        })
        .sum();
    if !area.is_finite() || area <= 0.0 {
        return Err(invalid("No nondegenerate geometry to bake"));
    }
    let mut atlas = Xatlas::new();
    for decl in &declarations {
        atlas
            .add_mesh(decl)
            .map_err(|e| invalid(&format!("Lightmap UV generation failed: {e:?}")))?;
    }
    atlas.generate(
        &ChartOptions::default(),
        &PackOptions {
            resolution: 0,
            texels_per_unit: (f64::from(resolution) / area.sqrt()) as f32,
            padding: PADDING,
            create_image: true,
            ..PackOptions::default()
        },
    );
    if atlas.atlas_count() != 1 {
        return Err(invalid("Expected a single lightmap atlas"));
    }
    let (width, height) = (atlas.width(), atlas.height());
    let layouts = atlas
        .meshes()
        .iter()
        .map(|mesh| MeshLayout {
            indices: mesh.index_array.to_vec(),
            vertices: mesh
                .vertex_array
                .iter()
                .map(|v| {
                    (
                        v.xref,
                        if v.atlas_index < 0 {
                            [0.0; 2]
                        } else {
                            [v.uv[0] / width as f32, v.uv[1] / height as f32]
                        },
                    )
                })
                .collect(),
        })
        .collect();
    let owners = atlas
        .image()
        .ok_or_else(|| invalid("Missing lightmap chart ownership"))?
        .to_vec();
    Ok((layouts, width, height, owners))
}

fn hemicube_axes(normal: Vec3) -> [(Vec3, Vec3); 5] {
    let up = if normal.y.abs() > 0.99 {
        Vec3::X
    } else {
        Vec3::Y
    };
    let right = up.cross(normal).normalize();
    let up = normal.cross(right);
    [
        (right, up),
        (-right, up),
        (up, -normal),
        (-up, normal),
        (normal, up),
    ]
}

fn hemicube_weights(resolution: u32) -> [Vec<f32>; 5] {
    let mut weights = hemicube_axes(Vec3::Z).map(|(direction, up)| {
        let right = direction.cross(up);
        (0..resolution)
            .flat_map(|y| {
                (0..resolution).map(move |x| {
                    let u = 2.0 * (x as f32 + 0.5) / resolution as f32 - 1.0;
                    let v = 1.0 - 2.0 * (y as f32 + 0.5) / resolution as f32;
                    let d = direction + right * u + up * v;
                    // cos(theta) * solid angle; normalize the quadrature so a white hemisphere gives 1.
                    d.z.max(0.0) / d.length_squared().powi(2)
                })
            })
            .collect::<Vec<_>>()
    });
    let sum: f32 = weights.iter().flatten().sum();
    for w in weights.iter_mut().flatten() {
        *w /= sum;
    }
    weights
}

fn dilate(map: &mut Lightmap, valid: &[bool], owners: &[u32]) {
    let mut valid = valid.to_vec();
    // xatlas tags chart coverage, bilinear borders and padding in the high three bits.
    let chart = |i: usize| owners[i] & 0x1fff_ffff;
    for _ in 0..PADDING + 1 {
        let mut next = valid.clone();
        for i in 0..valid.len() {
            if valid[i] || owners[i] == 0 {
                continue;
            }
            let (x, y) = (i % map.width as usize, i / map.width as usize);
            for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                let nx = x as i64 + dx;
                let ny = y as i64 + dy;
                if nx < 0 || ny < 0 || nx >= i64::from(map.width) || ny >= i64::from(map.height) {
                    continue;
                }
                let j = ny as usize * map.width as usize + nx as usize;
                if valid[j] && chart(i) == chart(j) {
                    map.front[i] = map.front[j];
                    map.back[i] = map.back[j];
                    next[i] = true;
                    break;
                }
            }
        }
        valid = next;
    }
}
