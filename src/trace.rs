use std::{collections::BTreeMap, fs, io, path::Path};

use crate::{cuda::BOUNCE_RAYS, material::Material, texture::Sampling};

/// Set at build time by the `TRACE` environment variable; normal builds record nothing.
pub const TRACE: bool = option_env!("TRACE").is_some();

/// Straight-line shader regions in program order; a lane marks the bit of each region it
/// executes, and trace records store their numbers.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Site {
    Input,
    BaseColor,
    BaseColorLevel,
    Normal,
    NormalLevel,
    MetallicRoughness,
    MetallicRoughnessLevel,
    Emissive,
    EmissiveLevel,
    Neural,
    NormalMap,
    Surface,
    Light,
    Shadow,
    Brdf,
    Output,
    Traverse,
    Hit,
}

const SITES: [Site; 18] = [
    Site::Input,
    Site::BaseColor,
    Site::BaseColorLevel,
    Site::Normal,
    Site::NormalLevel,
    Site::MetallicRoughness,
    Site::MetallicRoughnessLevel,
    Site::Emissive,
    Site::EmissiveLevel,
    Site::Neural,
    Site::NormalMap,
    Site::Surface,
    Site::Light,
    Site::Shadow,
    Site::Brdf,
    Site::Output,
    Site::Traverse,
    Site::Hit,
];

#[inline(always)]
pub(crate) fn mark(sites: &mut u32, site: Site) {
    if TRACE {
        *sites |= 1 << site as u32;
    }
}

/// One lane's loads, stores, FP32, special-function and integer operations in a region.
/// Source-level estimates of the shared shader, not compiled instruction counts.
fn ops(site: Site, sampling: Sampling, bounce: bool) -> [u16; 5] {
    match site {
        // Fragment attributes, material index, facing, visibility winner and descriptor.
        Site::Input => [20, 0, 0, 0, 10],
        Site::BaseColor | Site::Normal | Site::MetallicRoughness | Site::Emissive => match sampling
        {
            Sampling::None => [1, 0, 29, 1, 30],
            Sampling::Bilinear => [4, 0, 52, 0, 40],
            Sampling::Trilinear => [4, 0, 69, 1, 57],
        },
        Site::BaseColorLevel
        | Site::NormalLevel
        | Site::MetallicRoughnessLevel
        | Site::EmissiveLevel => [4, 0, 64, 0, 52],
        // The dp4a decoder: descriptor, 8 latent taps, then 504 weight and 78 scale/bias 16-byte
        // loads; the integer count includes 2016 dp4a. Latent filtering runs on ALUs, not
        // texture units. On tensor cores each record is instead one warp-wide network pass.
        Site::Neural => [605, 0, 850, 21, 2575],
        Site::NormalMap => [0, 0, 67, 4, 0],
        Site::Surface => [0, 0, 47, 2, 0],
        // The light and its shadow map arrive as kernel parameters.
        Site::Light => [0, 0, 6, 0, 0],
        Site::Shadow => [4, 0, 76, 2, 35],
        Site::Brdf => [0, 0, 112, 7, 0],
        // RGB, plus the albedo the bounce reads.
        Site::Output => [0, if bounce { 6 } else { 3 }, 0, 0, 5],
        // Traversal cost varies per ray and is not modeled.
        Site::Traverse => [0, 0, 0, 0, 0],
        // Triangle, primitive and three vertices; barycentric interpolation and facing.
        Site::Hit => [53, 0, 79, 1, 10],
    }
}

fn is_texture(site: Site) -> bool {
    matches!(
        site,
        Site::BaseColor
            | Site::BaseColorLevel
            | Site::Normal
            | Site::NormalLevel
            | Site::MetallicRoughness
            | Site::MetallicRoughnessLevel
            | Site::Emissive
            | Site::EmissiveLevel
    )
}

/// Writes one record per 32-lane warp (an 8x4 pixel block), pass, region and active mask;
/// texture regions also split by material, since each material reads its own textures, and
/// neural regions by texture set, since each set has its own network.
/// `lanes` holds, per pixel and pass (primary, then each of `BOUNCE_RAYS` bounce rays), the
/// region bits and material of that lane.
///
/// The little-endian file holds `b"RTRC"`, version 5, then the width, height and pass count as
/// u32, then records: warp u32, pass u8, `Site` number u8, material u32 (`u32::MAX` outside
/// texture regions; in neural regions the first material decoding the set), lane mask u32, then
/// loads, stores, fp, sfu and int counts per lane as u16.
pub fn write(
    path: impl AsRef<Path>,
    width: u32,
    height: u32,
    lanes: &[u32],
    materials: &[Material],
    sampling: Sampling,
    bounce: bool,
) -> io::Result<()> {
    // Neural records key each texture set by the first material decoding it.
    let first = |m: &Material| materials.iter().position(|n| n.neural == m.neural).unwrap();
    let sets: Vec<_> = materials.iter().map(first).collect();
    let (width, height) = (width as usize, height as usize);
    let stride = 1 + BOUNCE_RAYS as usize;
    let passes = if bounce { stride } else { 1 };
    let mut bytes = Vec::new();
    for value in [*b"RTRC", 5u32.to_le_bytes()] {
        bytes.extend(value);
    }
    for value in [width, height, passes] {
        bytes.extend((value as u32).to_le_bytes());
    }
    let blocks = width.div_ceil(8);
    for warp in 0..blocks * height.div_ceil(4) {
        for pass in 0..passes {
            let mut masks = BTreeMap::<(usize, u32), u32>::new();
            for lane in 0..32 {
                let (x, y) = (warp % blocks * 8 + lane % 8, warp / blocks * 4 + lane / 8);
                if x >= width || y >= height {
                    continue;
                }
                let slot = 2 * ((y * width + x) * stride + pass);
                let (sites, material) = (lanes[slot], lanes[slot + 1]);
                for (bit, &site) in SITES.iter().enumerate() {
                    if sites >> bit & 1 != 0 {
                        let material = match site {
                            Site::Neural => sets[material as usize] as u32,
                            _ if is_texture(site) => material,
                            _ => u32::MAX,
                        };
                        *masks.entry((bit, material)).or_default() |= 1 << lane;
                    }
                }
            }
            for ((bit, material), mask) in masks {
                bytes.extend((warp as u32).to_le_bytes());
                bytes.extend([pass as u8, bit as u8]);
                bytes.extend(material.to_le_bytes());
                bytes.extend(mask.to_le_bytes());
                for count in ops(SITES[bit], sampling, bounce) {
                    bytes.extend(count.to_le_bytes());
                }
            }
        }
    }
    fs::write(path, bytes)
}
