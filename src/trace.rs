use std::{collections::BTreeMap, fs, io, path::Path};

use crate::{
    cuda::{BOUNCES, SLOTS},
    material::Material,
    texture::Sampling,
};

/// Set at build time by the `TRACE` environment variable; normal builds record nothing.
pub const TRACE: bool = option_env!("TRACE").is_some();

/// Straight-line shader regions in program order; a lane marks the bit of each region it
/// executes, and trace records store their numbers.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
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
    Latents,
    Network,
    Appearance,
    AppearanceNetwork,
    AppearanceScatter,
    AppearanceLight,
    NormalMap,
    Surface,
    Light,
    Shadow,
    Brdf,
    Output,
    Traverse,
    Hit,
}

const SITES: u8 = Site::Hit as u8 + 1;

#[inline(always)]
pub(crate) fn mark(sites: &mut u32, site: Site) {
    if TRACE {
        *sites |= 1 << site as u32;
    }
}

/// Operation types per region: loads, stores, FP32, special-function and integer operations,
/// then matrix multiplies, one type per shape M x N x K (samples, outputs, inputs): int8 with
/// int32 accumulation for the NTC network's layers, 1x64x48, 1x48x64, 1x32x48 and 1x16x32, then
/// FP16 for the neural material's, 1x12x8 (frames), 1x16x11, 1x16x16 and 1x9x16 (sampler, whose
/// middle shape the decoder shares), 1x16x24 and 1x3x16 (decoder).
const OPS: usize = 15;

/// Pads a region's scalar counts with no matrix multiplies.
const fn scalar(ops: [u16; 5]) -> [u16; OPS] {
    let mut row = [0; OPS];
    let mut i = 0;
    while i < 5 {
        row[i] = ops[i];
        i += 1;
    }
    row
}

/// One lane's operations in a region, by type: all of them, then those on its longest chain of
/// dependent operations (the critical path, the region's inputs being ready at its start), as
/// hand counts of the shared shader's source after inlining, not compiled instruction counts.
/// The critical path is the longest at reference latencies of 30 cycles per load, 20 per 16x8x32
/// matrix multiply, 15 per special function and 4 per other operation. A load or store is one
/// value accessed (a scalar, texel, or vector of up to 16 bytes). FP32 counts float adds,
/// multiplies, FMAs, min, max, compares, selects, floor and trunc; SFU counts square roots,
/// reciprocals, log2 and exp2; a division is a reciprocal and a multiply, or one multiply by a
/// constant. Int counts integer arithmetic, compares and selects, conversions, and one address
/// per access not at a constant offset from another. Free: float negation and abs, arithmetic on
/// constants and kernel parameters, selecting a constant, repeated values after the first,
/// bounds checks, loop control and shuffles that only rearrange data. Branches inside a region
/// count their common path, and mip-offset loops two levels.
fn ops(site: Site, sampling: Sampling, bounce: bool) -> ([u16; OPS], [u16; OPS]) {
    match site {
        // Visibility winner, material index, 16 attributes and facing; pixel and SoA indexing.
        // Path: pixel index, winner load, coverage test, then the predicated attribute loads.
        Site::Input => (scalar([19, 0, 0, 0, 27]), scalar([2, 0, 0, 0, 10])),
        // 7 descriptor fields and the texels; LOD, repeat wrapping, filtering. Path: descriptor,
        // LOD and mip offsets, wrapped texel address, texel, filter.
        Site::BaseColor | Site::Normal | Site::MetallicRoughness | Site::Emissive => match sampling
        {
            Sampling::None => (scalar([8, 0, 34, 1, 41]), scalar([2, 0, 11, 1, 15])),
            Sampling::Bilinear => (scalar([11, 0, 62, 0, 49]), scalar([2, 0, 13, 0, 9])),
            Sampling::Trilinear => (scalar([11, 0, 79, 1, 65]), scalar([2, 0, 16, 1, 16])),
        },
        // The next level's bilinear sample and the blend between levels.
        Site::BaseColorLevel
        | Site::NormalLevel
        | Site::MetallicRoughnessLevel
        | Site::EmissiveLevel => (scalar([4, 0, 59, 0, 60]), scalar([1, 0, 9, 0, 18])),
        // An NTC sample outside its network, once per lane: 15 descriptor fields and 8 latent
        // taps, filtering and nibble unpacking (on ALUs, not texture units), positional encoding,
        // and the outputs' sRGB. Path: size, LOD, mip, mip and latent descriptors, a latent tap,
        // a feature's 4-tap filter, then sRGB.
        Site::Latents => (scalar([23, 0, 338, 9, 668]), scalar([4, 0, 21, 3, 21])),
        // The NTC network, once per texture set: one multiply per layer, 78 scale/bias loads,
        // activation of 144 hidden and 9 output values. Path: the first bias, then each layer's
        // multiply and activation.
        Site::Network => (
            [78, 0, 585, 0, 300, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 13, 0, 8, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        ),
        // Gram-Schmidt tangent frame, bitangent, two normalizations, all on the path.
        Site::NormalMap => (scalar([0, 0, 64, 4, 0]), scalar([0, 0, 22, 4, 0])),
        // Texture presence, material factors, facing, view vector and BRDF inputs. Path: the
        // view vector's normalization.
        Site::Surface => (scalar([10, 0, 55, 2, 6]), scalar([0, 0, 7, 2, 0])),
        // The light and its shadow map arrive as kernel parameters.
        Site::Light => (scalar([0, 0, 6, 0, 0]), scalar([0, 0, 4, 0, 0])),
        // Light-space projection and 2x2 PCF over 4 depths. Path: projection, perspective
        // divide, texel address, depth, comparison and blend.
        Site::Shadow => (scalar([4, 0, 76, 3, 24]), scalar([1, 0, 21, 1, 7])),
        // GGX distribution, height-correlated Smith visibility, Schlick Fresnel, accumulation.
        // Path: half vector, GGX distribution, then the products into the sum.
        Site::Brdf => (scalar([0, 0, 109, 6, 0]), scalar([0, 0, 23, 3, 0])),
        // Color, plus the albedo the bounce reads; the store transpose only rearranges data.
        Site::Output => {
            let values = 1 + bounce as u16;
            (scalar([0, values, 0, 0, values]), scalar([0, 1, 0, 0, 1]))
        }
        // Traversal cost varies per ray and is not modeled.
        Site::Traverse => ([0; OPS], [0; OPS]),
        // A neural material outside its networks, once per lane: facing, view vector, tangent
        // frame and incident direction, the latent texel and the material's two words, 3
        // random numbers; then the learned frames, the sampler's lobes, a diffuse-lobe sample
        // (the common path) and the mixture density, the scatter decode's inputs and weight.
        // Path: the latent texel, then the diffuse slope offset through the sampled direction to
        // the decoder's inputs, and the weight.
        Site::Appearance => (scalar([3, 0, 416, 46, 46]), scalar([1, 0, 38, 8, 7])),
        // The frame layer and the sampler, once per material among lanes seeing the surface
        // from above: 11 bias loads and activation of 32 hidden values. Path: a bias, then the
        // sampler's layers.
        Site::AppearanceNetwork => (
            [11, 0, 128, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
            [1, 0, 6, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
        ),
        // The decode at the sampled direction, once per material among lanes sampling above the
        // surface: 9 bias loads, activation of 32 hidden values and 3 outputs. Path: the layers,
        // their biases loaded during the per-lane work before them.
        Site::AppearanceScatter => (
            [9, 0, 335, 6, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1],
            [0, 0, 16, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1],
        ),
        // The decode toward the light, once per material among lit lanes: its inputs, 9 bias
        // loads, the decoder and the light it reflects. Path: the inputs, layers, then the sum.
        Site::AppearanceLight => (
            [9, 0, 468, 9, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1],
            [0, 0, 35, 5, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1],
        ),
        // Triangle, primitive and the 12 attributes used of each vertex; barycentric
        // interpolation and facing. Path: triangle, vertex attributes, normal and its
        // normalization.
        Site::Hit => (scalar([40, 0, 85, 2, 19]), scalar([2, 0, 9, 2, 5])),
    }
}

/// Writes one record per 32-lane warp (an 8x4 pixel block), pass, region and active mask;
/// texture regions also split by material, since each material reads its own textures, and
/// network regions by texture set, or by material for neural materials, since each network runs
/// separately on matrix hardware. Latent and other per-lane regions do not split: lanes with
/// different sets or materials run them together.
/// `lanes` holds, per pixel and pass (primary, then each bounce ray's hits by depth), the region
/// bits and material of that lane.
///
/// The little-endian file holds `b"RTRC"`, version 9, then the width, height, pass count,
/// bounces per path (0 without the bounce; pass 1 + r b + d is ray r at depth d), `Site` count
/// and op-type count as u32; then per `Site`, its per-lane op counts in `OPS` order and its
/// critical path's (see `ops`) as u16; then records: warp u32, pass u8, `Site` number u8,
/// material u32 (`u32::MAX` outside texture and network regions; in NTC network regions the
/// first material decoding the set), lane mask u32.
pub fn write(
    path: impl AsRef<Path>,
    width: u32,
    height: u32,
    lanes: &[u32],
    materials: &[Material],
    sampling: Sampling,
    bounce: bool,
) -> io::Result<()> {
    // Network records key each texture set by the first material decoding it.
    let first = |m: &Material| {
        materials
            .iter()
            .position(|n| n.network == m.network)
            .unwrap()
    };
    let sets: Vec<_> = materials.iter().map(first).collect();
    let (width, height) = (width as usize, height as usize);
    let passes = if bounce { SLOTS } else { 1 };
    let bounces = if bounce { BOUNCES as usize } else { 0 };
    let mut bytes = Vec::new();
    for value in [*b"RTRC", 9u32.to_le_bytes()] {
        bytes.extend(value);
    }
    for value in [width, height, passes, bounces, SITES as usize, OPS] {
        bytes.extend((value as u32).to_le_bytes());
    }
    // Sites number 0..SITES in declaration order.
    let sites: Vec<Site> = (0..SITES)
        .map(|bit| unsafe { std::mem::transmute(bit) })
        .collect();
    for &site in &sites {
        let (work, path) = ops(site, sampling, bounce);
        for count in work.into_iter().chain(path) {
            bytes.extend(count.to_le_bytes());
        }
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
                let slot = 2 * ((y * width + x) * SLOTS + pass);
                let (bits, material) = (lanes[slot], lanes[slot + 1]);
                for (bit, &site) in sites.iter().enumerate() {
                    if bits >> bit & 1 != 0 {
                        let material = match site {
                            Site::Network => sets[material as usize] as u32,
                            Site::AppearanceNetwork
                            | Site::AppearanceScatter
                            | Site::AppearanceLight => material,
                            // Texture regions.
                            _ if (Site::BaseColor as usize..=Site::EmissiveLevel as usize)
                                .contains(&bit) =>
                            {
                                material
                            }
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
            }
        }
    }
    fs::write(path, bytes)
}
