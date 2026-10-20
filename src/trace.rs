use std::{
    collections::BTreeMap,
    fs::File,
    io::{self, BufWriter, Write},
    path::PathBuf,
};

use glam::Vec3;
use rayon::prelude::*;

use crate::{
    fragment::{Ops, Site, Trace, ops, shade},
    material::Material,
    pipeline::{Fragment, FragmentStage, Shading},
};

// Work the CUDA shading kernel does outside the shared shader: fragment attributes,
// material index, facing, visibility winner and material descriptor; RGB stores.
const INPUT: Ops = ops(38, 0, 0, 0, 10);
const OUTPUT: Ops = ops(0, 3, 0, 0, 5);

/// Fragment stage that writes the shader operations of each 32-lane warp, which covers an
/// 8x4 pixel block as in the CUDA backend. The colors it returns match the CPU stage.
///
/// The little-endian file holds `b"RTRC"`, version 1 and the render width as u32, then
/// one record per operation and active mask: warp u32, lane mask u32, `Site` number u8,
/// light index u32, then loads, stores, fp, sfu and int counts per lane as u16.
pub struct TraceFragment {
    width: u32,
    path: PathBuf,
}

impl TraceFragment {
    pub fn new(width: u32, path: impl Into<PathBuf>) -> Self {
        Self {
            width,
            path: path.into(),
        }
    }
}

#[derive(Default)]
struct Lane(Vec<(Site, u32, Ops)>);

impl Trace for Lane {
    fn op(&mut self, site: Site, light: u32, ops: Ops) {
        self.0.push((site, light, ops));
    }
}

impl FragmentStage for TraceFragment {
    fn shade(
        &mut self,
        fragments: &[Fragment],
        materials: &[Material],
        shading: &Shading<'_>,
    ) -> io::Result<Vec<Vec3>> {
        let width = self.width as usize;
        let mut warps = BTreeMap::<u32, Vec<(u32, usize)>>::new();
        for (index, fragment) in fragments.iter().enumerate() {
            let (x, y) = (fragment.sample_index % width, fragment.sample_index / width);
            let warp =
                u32::try_from(y / 4 * width.div_ceil(8) + x / 8).map_err(io::Error::other)?;
            warps
                .entry(warp)
                .or_default()
                .push((y as u32 % 4 * 8 + x as u32 % 8, index));
        }
        let default_material = Material::default();
        let warps: Vec<_> = warps
            .into_par_iter()
            .map(|(warp, lanes)| {
                // Lanes reaching the same region together form one active mask.
                let mut regions = BTreeMap::new();
                let mut colors = Vec::with_capacity(lanes.len());
                for (lane, index) in lanes {
                    let fragment = &fragments[index];
                    let material = fragment
                        .material_index
                        .map_or(&default_material, |index| &materials[index]);
                    let mut trace = Lane::default();
                    trace.op(Site::Input, 0, INPUT);
                    let color = shade(
                        shading,
                        material,
                        &fragment.attributes,
                        fragment.front_facing,
                        &mut trace,
                    );
                    trace.op(Site::Output, 0, OUTPUT);
                    colors.push((index, color));
                    for (site, light, ops) in trace.0 {
                        let phase = match site {
                            Site::Light | Site::Shadow | Site::Brdf => 1,
                            _ if site > Site::Brdf => 2,
                            _ => 0,
                        };
                        regions.entry((phase, light, site)).or_insert((0u32, ops)).0 |=
                            1u32 << lane;
                    }
                }
                let mut records = Vec::new();
                for ((_, light, site), (mask, ops)) in regions {
                    records.extend(warp.to_le_bytes());
                    records.extend(mask.to_le_bytes());
                    records.push(site as u8);
                    records.extend(light.to_le_bytes());
                    for count in [ops.loads, ops.stores, ops.fp, ops.sfu, ops.int] {
                        records.extend(count.to_le_bytes());
                    }
                }
                (records, colors)
            })
            .collect();
        let mut file = BufWriter::new(File::create(&self.path)?);
        file.write_all(b"RTRC")?;
        file.write_all(&1u32.to_le_bytes())?;
        file.write_all(&self.width.to_le_bytes())?;
        let mut colors = vec![Vec3::ZERO; fragments.len()];
        for (records, warp_colors) in warps {
            file.write_all(&records)?;
            for (index, color) in warp_colors {
                colors[index] = color;
            }
        }
        file.flush()?;
        Ok(colors)
    }
}
