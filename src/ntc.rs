//! RTXNTC neural texture sets, decoded on sample with the int8 network of LibNTC's
//! `Inference.hlsli`.

use std::{fs, io, path::Path};

use cuda_device::{
    dotprod::dp4a_s32,
    float::{add_rm_f32, add_rn_sat_f32, fma_rn_f32, fma_rn_sat_f32},
    prmt::prmt,
    thread::__unroll_config,
    warp,
    wmma::{mma_m16n8k16_s32_s8, mma_m16n8k32_s32_s8},
};
use glam::{Vec2, Vec4};
use serde::Deserialize;

use crate::texture::{Texel, UvSample, decode_srgb, lod};

/// RTXNTC's network: 48 inputs, hidden layers of 64, 48 and 32 outputs, then 16 outputs.
const LAYERS: [(usize, usize); 4] = [(48, 64), (64, 48), (48, 32), (32, 16)];
/// Each layer's first output.
const OUTPUTS: [usize; 4] = [0, 64, 112, 144];
/// Each layer's first weight word. dp4a rows hold a layer's int8 inputs in order; tensor-core
/// rows hold each lane's fragment words together, padded to whole 16-byte loads (see `load`).
const WEIGHTS: [usize; 4] = if MMA {
    [0, 1024, 1792, 2304]
} else {
    [0, 768, 1536, 1920]
};
/// Weight words, then scale and bias words: dp4a sets keep all scales, then all biases;
/// tensor-core sets keep each column pair's two scales and two biases together.
const WEIGHT_WORDS: usize = if MMA { 2432 } else { 2048 };
/// The scales, and the biases, of all layers in a dp4a set.
const VECTOR_WORDS: usize = 160;
const MAX_COLOR_MIPS: usize = 16;
const MAX_LATENTS: usize = 8;
/// Inputs in [-1, 1] map to int8 by this scale.
const INPUT_SCALE: f32 = 127.5;
/// 1.5 * 2^23. A float below 2^22 in magnitude plus this holds its rounded integer in the low
/// mantissa bits, which replaces conversion-unit instructions with an add.
const MAGIC: f32 = 12_582_912.0;
/// Decode on tensor cores unless the build sets `NTC_DP4A`, which decodes each lane with dp4a.
pub const MMA: bool = option_env!("NTC_DP4A").is_none();

#[derive(Clone, Copy, Default, PartialEq)]
struct ColorMip {
    latent: u32,
    position_lod: f32,
    position_scale: f32,
}

#[derive(Clone, Copy, Default, PartialEq)]
struct Latent {
    /// Words from the set's first word; each texel packs its 16 4-bit features in two words.
    offset: u32,
    width: u32,
    height: u32,
}

/// One texture set; its weights and latents are words in the texel array.
#[derive(Clone, Copy, PartialEq)]
pub struct NeuralSet {
    /// First word: weights, scales, biases, then the loaded latent levels.
    offset: u32,
    width: u32,
    height: u32,
    /// Finer color mips read GDeflate-compressed latents, which are not loaded.
    first_mip: u32,
    mips: u32,
    color_mips: [ColorMip; MAX_COLOR_MIPS],
    latents: [Latent; MAX_LATENTS],
    /// sRGB outputs, numbered after reordering to base color, ORM and normal.
    srgb: u32,
}

impl NeuralSet {
    /// The network inputs for the nearest texel of the nearest decodable mip level, as
    /// inference on sample evaluates one texel: 16 features from each of two latent levels,
    /// the positional encoding and the mip's LOD constant, as packed int8.
    #[inline(always)]
    fn inputs(&self, texels: &[Texel], sample: UvSample) -> [u32; 12] {
        let level = lod(Vec2::new(self.width as f32, self.height as f32), sample);
        let mip = ((level + 0.5) as u32).clamp(self.first_mip, self.mips - 1);
        let (width, height) = ((self.width >> mip).max(1), (self.height >> mip).max(1));
        let uv = sample.uv - sample.uv.floor();
        let x = ((uv.x * width as f32) as u32).min(width - 1);
        let y = ((uv.y * height as f32) as u32).min(height - 1);
        let color = self.color_mips[mip as usize];
        let center = Vec2::new(
            (x as f32 + 0.5) / width as f32,
            (y as f32 + 0.5) / height as f32,
        );
        let [a0, a1, a2, a3] = self.features(texels, color.latent, center);
        let [b0, b1, b2, b3] = self.features(texels, color.latent + 1, center);
        let mut position = Vec2::new(x as f32, y as f32) * color.position_scale;
        let mut waves = [0; 3];
        for wave in 0..3 {
            waves[wave] = pack(
                [position.x, position.y, position.x + 0.25, position.y + 0.25]
                    .map(|v| ((v - v.floor()) * 2.0 - 1.0) * INPUT_SCALE)
                    .map(|v| v as i32 as u32),
            );
            position *= 2.0;
        }
        let position_lod = (color.position_lod * INPUT_SCALE) as i32 as u32;
        [
            a0,
            a1,
            a2,
            a3,
            b0,
            b1,
            b2,
            b3,
            waves[0],
            waves[1],
            waves[2],
            pack([position_lod, position_lod, 0, 0]),
        ]
    }

    /// Bilinearly filters a latent level's features with repeat wrapping, as the latent sampler
    /// does, and quantizes them to packed int8.
    #[inline(always)]
    fn features(&self, texels: &[Texel], level: u32, uv: Vec2) -> [u32; 4] {
        let latent = self.latents[level as usize];
        let (width, height) = (latent.width as i32, latent.height as i32);
        let p = uv * Vec2::new(width as f32, height as f32) - 0.5;
        let t = p - p.floor();
        let (x, y) = (p.x.floor() as i32, p.y.floor() as i32);
        // Texel centers keep taps within one texel of the level.
        let wrap = |i: i32, n: i32| {
            if i < 0 {
                i + n
            } else if i >= n {
                i - n
            } else {
                i
            }
        };
        let level = Words::new(texels, (self.offset + latent.offset) as usize);
        let tap =
            |dx, dy| level.pair(2 * (wrap(y + dy, height) * width + wrap(x + dx, width)) as usize);
        let taps = [tap(0, 0), tap(1, 0), tap(0, 1), tap(1, 1)];
        let weights = [
            (1.0 - t.x) * (1.0 - t.y),
            t.x * (1.0 - t.y),
            (1.0 - t.x) * t.y,
            t.x * t.y,
        ];
        // Texel centers of power-of-two levels make every weight, product and sum below exact.
        let mut bytes = [0; 16];
        for feature in 0..16 {
            // Unrolled so the features stay in registers.
            __unroll_config::<0>();
            // Nibble n as the float 1 + n / 16: the weights sum to one, so `sum` is 1 + n / 16
            // for the filtered nibble n.
            let mut sum = 0.0;
            for k in 0..4 {
                let n = taps[k][feature / 8] >> (4 * (feature % 8)) & 15;
                sum = fma_rn_f32(weights[k], f32::from_bits(0x3F80_0000 | n << 19), sum);
            }
            // The UNORM4 value n / 15 becomes floor(17 n - 128), rounded down by the add.
            bytes[feature] = add_rm_f32(fma_rn_f32(sum, 272.0, -400.0), MAGIC).to_bits();
        }
        [0, 4, 8, 12].map(|i| pack([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]))
    }

    /// The network on one lane with dp4a: base color, ORM and normal outputs.
    #[inline(always)]
    fn network(&self, texels: &[Texel], inputs: &[u32; 12]) -> [f32; 9] {
        let hidden = self.hidden::<12, 16>(texels, WEIGHTS[0], OUTPUTS[0], inputs);
        let hidden = self.hidden::<16, 12>(texels, WEIGHTS[1], OUTPUTS[1], &hidden);
        let hidden = self.hidden::<12, 8>(texels, WEIGHTS[2], OUTPUTS[2], &hidden);
        // No channel reads the last four outputs.
        let output = [0, 1, 2].map(|g| {
            let (sums, scale) = self.outputs(texels, WEIGHTS[3], OUTPUTS[3], g, &hidden);
            [0, 1, 2, 3].map(|k| sums[k] as f32 * scale[k])
        });
        [0, 1, 2, 3, 4, 5, 6, 7, 8].map(|c| output[c / 4][c % 4])
    }

    /// Int32 sums and scales for outputs `4 group..4 group + 4` of a layer with `4 N` int8
    /// inputs.
    #[inline(always)]
    fn outputs<const N: usize>(
        &self,
        texels: &[Texel],
        weights: usize,
        outputs: usize,
        group: usize,
        input: &[u32; N],
    ) -> ([i32; 4], [f32; 4]) {
        let set = Words::new(texels, self.offset as usize);
        let first = outputs + 4 * group;
        let scale = set.quad(WEIGHT_WORDS + first).map(f32::from_bits);
        let bias = set.quad(WEIGHT_WORDS + VECTOR_WORDS + first);
        let mut sums = [0; 4];
        for k in 0..4 {
            let row = weights + (4 * group + k) * N;
            let mut sum = bias[k] as i32;
            for chunk in 0..N / 4 {
                let w = set.quad(row + 4 * chunk);
                for j in 0..4 {
                    sum = dp4a_s32(input[4 * chunk + j], w[j], sum);
                }
            }
            sums[k] = sum;
        }
        (sums, scale)
    }

    /// A hidden layer, quantized to int8 for the next layer.
    #[inline(always)]
    fn hidden<const N: usize, const G: usize>(
        &self,
        texels: &[Texel],
        weights: usize,
        outputs: usize,
        input: &[u32; N],
    ) -> [u32; G] {
        let mut packed = [0; G];
        // The group loop stays rolled, as in Inference.hlsli: unrolling it left both shading
        // kernels at 255 registers with spills, and slower.
        for group in 0..G {
            let (sums, scale) = self.outputs(texels, weights, outputs, group, input);
            packed[group] = pack([0, 1, 2, 3].map(|k| activate(hidden_sum(sums[k]) * scale[k])));
        }
        packed
    }
}

/// Decodes a lane's texture set texel, returning linear base color, occlusion, roughness,
/// metallic, then the normal, each in [0, 1]. With tensor cores every lane of the warp must
/// call this together, with or without a set.
#[inline(always)]
pub fn sample(texels: &[Texel], set: Option<&NeuralSet>, sample: UvSample) -> [f32; 9] {
    if MMA {
        return sample_warp(texels, set, sample);
    }
    match set {
        Some(set) => finish(set.srgb, set.network(texels, &set.inputs(texels, sample))),
        None => [0.0; 9],
    }
}

/// Tensor-core decode. Each texture set in the warp costs one pass of the whole warp through
/// its network, as on a warp-wide tensor core; rows of lanes with other sets are discarded.
#[inline(always)]
fn sample_warp(texels: &[Texel], set: Option<&NeuralSet>, sample: UvSample) -> [f32; 9] {
    let lane = warp::lane_id();
    let (group, thread) = (lane / 4, lane % 4);
    let key = set.map_or(u32::MAX, |set| set.offset);
    let mut pending = warp::ballot(key != u32::MAX);
    if pending == 0 {
        return [0.0; 9];
    }
    let (inputs, srgb) = match set {
        Some(set) => (set.inputs(texels, sample), set.srgb),
        None => ([0; 12], 0),
    };
    // Layer-0 A fragments: for row tile m, 16-input chunk c and row half h, input word
    // 4c + thread of the lane 16m + 8h + group.
    let mut a = [[[0; 2]; 3]; 2];
    for m in 0..2 {
        __unroll_config::<0>();
        for c in 0..3 {
            __unroll_config::<0>();
            for h in 0..2 {
                __unroll_config::<0>();
                for s in 0..4 {
                    __unroll_config::<0>();
                    let word = warp::shuffle(inputs[4 * c + s], 16 * m + 8 * h + group);
                    if s as u32 == thread {
                        a[m as usize][c][h as usize] = word;
                    }
                }
            }
        }
    }
    let mut output = [[[0.0; 4]; 2]; 2];
    while pending != 0 {
        let key_of_pass = warp::shuffle(key, pending.trailing_zeros());
        let members = warp::ballot(key == key_of_pass);
        let set = Words::new(texels, key_of_pass as usize);
        // A 16-row half without members skips its matrix operations and activations.
        match (members & 0xffff != 0, members >> 16 != 0) {
            (true, true) => pass::<3>(set, &a, members, &mut output),
            (true, false) => pass::<1>(set, &a, members, &mut output),
            _ => pass::<2>(set, &a, members, &mut output),
        }
        pending &= !members;
    }
    // Each lane collects its own row's channels from the four lanes holding them.
    let (m, h) = (lane / 16, lane / 8 % 2);
    let mut channels = [0.0; 9];
    for c in 0..9 {
        __unroll_config::<0>();
        let source = 4 * (lane % 8) + c as u32 % 8 / 2;
        for mm in 0..2 {
            __unroll_config::<0>();
            for hh in 0..2 {
                __unroll_config::<0>();
                let value = warp::shuffle_f32(output[mm as usize][c / 8][2 * hh + c % 2], source);
                if (mm, hh as u32) == (m, h) {
                    channels[c] = value;
                }
            }
        }
    }
    finish(srgb, channels)
}

/// One pass of the warp through a set's network for the 16-row halves in the `HALVES` bits;
/// the output rows of `members` land in `output`.
#[inline(always)]
fn pass<const HALVES: usize>(
    set: Words,
    a: &[[[u32; 2]; 3]; 2],
    members: u32,
    output: &mut [[[f32; 4]; 2]; 2],
) {
    let group = warp::lane_id() / 4;
    let layer = |l: usize| (set.at(WEIGHTS[l]), set.at(WEIGHT_WORDS + 2 * OUTPUTS[l]));
    let hidden = hidden_layer::<3, 4, HALVES>(layer(0), a);
    let hidden = hidden_layer::<4, 3, HALVES>(layer(1), &hidden);
    let hidden = hidden_layer::<3, 2, HALVES>(layer(2), &hidden);
    for j in 0..2 {
        __unroll_config::<0>();
        let (sums, scale) = tile::<2, HALVES>(layer(3), j, &hidden);
        for m in 0..2 {
            __unroll_config::<0>();
            for r in 0..4 {
                __unroll_config::<0>();
                if members >> (16 * m + 8 * (r / 2) + group) & 1 != 0 {
                    output[m as usize][j][r as usize] =
                        sums[m as usize][r as usize] as f32 * scale[r as usize % 2];
                }
            }
        }
    }
}

/// A hidden layer with `16 KC` inputs and `16 NC` outputs on tensor cores. Outputs stay in
/// accumulator order: the packed word for chunk c holds columns 16c + 2 thread + {0, 1, 8, 9},
/// which loading made the next layer's inputs 16c + 4 thread + {0, 1, 2, 3}.
#[inline(always)]
fn hidden_layer<const KC: usize, const NC: usize, const HALVES: usize>(
    layer: (Words, Words),
    a: &[[[u32; 2]; KC]; 2],
) -> [[[u32; 2]; NC]; 2] {
    let mut packed = [[[0; 2]; NC]; 2];
    for c in 0..NC {
        __unroll_config::<0>();
        let (low, low_scale) = tile::<KC, HALVES>(layer, 2 * c, a);
        let (high, high_scale) = tile::<KC, HALVES>(layer, 2 * c + 1, a);
        for m in 0..2 {
            __unroll_config::<0>();
            if HALVES >> m & 1 == 0 {
                continue;
            }
            for h in 0..2 {
                __unroll_config::<0>();
                packed[m][c][h] = pack([
                    activate(hidden_sum(low[m][2 * h]) * low_scale[0]),
                    activate(hidden_sum(low[m][2 * h + 1]) * low_scale[1]),
                    activate(hidden_sum(high[m][2 * h]) * high_scale[0]),
                    activate(hidden_sum(high[m][2 * h + 1]) * high_scale[1]),
                ]);
            }
        }
    }
    packed
}

/// Int32 sums of 8-output tile `tile` of a layer with `16 KC` inputs for the 16-row halves in
/// `HALVES`, and the scales of this lane's two columns: register r holds row group + 8 (r / 2)
/// and column 8 tile + 2 thread + r % 2. `layer` holds the layer's weight and scale-bias words.
#[inline(always)]
fn tile<const KC: usize, const HALVES: usize>(
    (weights, vectors): (Words, Words),
    tile: usize,
    a: &[[[u32; 2]; KC]; 2],
) -> ([[i32; 4]; 2], [f32; 2]) {
    let lane = warp::lane_id() as usize;
    let (group, thread) = (lane / 4, lane % 4);
    // 32-bit loads: wider ones need aligned register quads, which spilled more and ran slower.
    let [s0, s1, b0, b1] = vectors.words(16 * tile + 4 * thread);
    // B fragments of output 8 tile + group: one word per 16-input chunk.
    let lanes = lane_words(KC);
    let w: [u32; KC] = weights.words((8 * tile + group) * 4 * lanes + thread * lanes);
    let mut sums = [[b0 as i32, b1 as i32, b0 as i32, b1 as i32]; 2];
    for m in 0..2 {
        __unroll_config::<0>();
        if HALVES >> m & 1 == 0 {
            continue;
        }
        for step in 0..KC / 2 {
            let [lo, hi] = [a[m][2 * step], a[m][2 * step + 1]];
            let b = [w[2 * step], w[2 * step + 1]];
            sums[m] = unsafe { mma_m16n8k32_s32_s8(sums[m], [lo[0], lo[1], hi[0], hi[1]], b) };
        }
        if KC % 2 == 1 {
            sums[m] = unsafe { mma_m16n8k16_s32_s8(sums[m], a[m][KC - 1], w[KC - 1]) };
        }
    }
    (sums, [f32::from_bits(s0), f32::from_bits(s1)])
}

/// Words per lane in a tensor-core weight row of `chunks` 16-input chunks, padded so each
/// lane's words start 16-byte aligned (8-byte for two chunks).
const fn lane_words(chunks: usize) -> usize {
    if chunks == 2 { 2 } else { 4 }
}

/// Clamped HGELU of a hidden output v, given 80 v (loading scales hidden layers by 80),
/// quantized to int8 for the next layer in the low byte.
#[inline(always)]
fn activate(v80: f32) -> u32 {
    // min(80 v, 240): integer order matches float order when either side is a positive float.
    let low = f32::from_bits((v80.to_bits() as i32).min(240f32.to_bits() as i32) as u32);
    // round(80 min(v, 3) sat(v / 3 + 1 / 2)) - 113 in the low byte. Ties round up rather than
    // to even as in HLSL, which needs the product to land exactly on a half.
    fma_rn_f32(
        low,
        fma_rn_sat_f32(v80, 0.333333 / 80.0, 0.5),
        MAGIC - 113.0,
    )
    .to_bits()
}

/// A hidden layer's sum. Loading adds `MAGIC`'s bit pattern to hidden biases, so the int32 sum
/// is the bit pattern of `MAGIC` plus the sum, exact while the sum stays below 2^22.
#[inline(always)]
fn hidden_sum(sum: i32) -> f32 {
    f32::from_bits(sum as u32) - MAGIC
}

/// Clamps outputs to [0, 1] and linearizes sRGB ones.
#[inline(always)]
fn finish(srgb: u32, values: [f32; 9]) -> [f32; 9] {
    let channel = |c: usize| {
        let value = add_rn_sat_f32(values[c], 0.0);
        if srgb >> c & 1 != 0 {
            decode_srgb(value)
        } else {
            value
        }
    };
    [0, 1, 2, 3, 4, 5, 6, 7, 8].map(channel)
}

/// Packs the low bytes of four words.
#[inline(always)]
fn pack(words: [u32; 4]) -> u32 {
    prmt(
        prmt(words[0], words[1], 0x40),
        prmt(words[2], words[3], 0x40),
        0x5410,
    )
}

/// A texture set's words in the texel array.
#[derive(Clone, Copy)]
struct Words(*const u32);

impl Words {
    #[inline(always)]
    fn new(texels: &[Texel], offset: usize) -> Self {
        // Loading checked the set's extent.
        Self(unsafe { texels.as_ptr().cast::<u32>().add(offset) })
    }

    #[inline(always)]
    fn at(self, word: usize) -> Self {
        Self(unsafe { self.0.add(word) })
    }

    #[inline(always)]
    fn words<const N: usize>(self, word: usize) -> [u32; N] {
        std::array::from_fn(|i| unsafe { self.0.add(word + i).read() })
    }

    /// Two words starting at an even word.
    #[inline(always)]
    fn pair(self, word: usize) -> [u32; 2] {
        let pair = unsafe { self.0.add(word).cast::<u64>().read() };
        [pair as u32, (pair >> 32) as u32]
    }

    /// Four words starting at a multiple of four, in one 16-byte load (a texel-typed read
    /// splits into four).
    #[inline(always)]
    fn quad(self, word: usize) -> [u32; 4] {
        let quad = unsafe { self.0.add(word).cast::<u128>().read() };
        [0, 32, 64, 96].map(|shift| (quad >> shift) as u32)
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Document {
    width: u32,
    height: u32,
    num_color_mips: usize,
    latent_shape: LatentShape,
    mlp_versions: Vec<Mlp>,
    textures: Vec<TextureInfo>,
    channels: Vec<Channel>,
    latents: Vec<LatentImage>,
    color_mips: Vec<ColorMipInfo>,
    views: Vec<View>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct LatentShape {
    num_features: usize,
}

#[derive(Deserialize)]
struct Mlp {
    layers: Vec<Layer>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Layer {
    input_channels: usize,
    output_channels: usize,
    weight_view: usize,
    /// Absent from FP8 layers, which are not loaded.
    #[serde(default)]
    scale_view: usize,
    bias_view: usize,
    weight_type: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct TextureInfo {
    name: String,
    first_channel: usize,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Channel {
    color_space: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct LatentImage {
    width: u32,
    height: u32,
    layer_views: Vec<usize>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ColorMipInfo {
    latent_mip: Option<usize>,
    #[serde(default)]
    position_lod: f32,
    #[serde(default)]
    position_scale: f32,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct View {
    offset: usize,
    stored_size: usize,
    compression: Option<String>,
}

/// Loads an RTXNTC texture set file, appending its int8 network and uncompressed latent levels
/// to `texels`. The set needs RTXNTC's network shape, 16 latent features, and BaseColor, ORM
/// and Normal textures.
pub(crate) fn load(path: &Path, texels: &mut Vec<Texel>) -> io::Result<NeuralSet> {
    let invalid = |message: &str| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{}: {message}", path.display()),
        )
    };
    let file = fs::read(path)?;
    if !file.starts_with(b"NTEX") || file.len() < 40 {
        return Err(invalid("not an NTC texture set"));
    }
    let field = |at: usize| u64::from_le_bytes(file[at..at + 8].try_into().unwrap()) as usize;
    let (json, data) = (&file[field(8)..][..field(16)], field(24));
    // The descriptor is padded with zeros.
    let json = json.split(|&b| b == 0).next().unwrap();
    let doc: Document =
        serde_json::from_slice(json).map_err(|error| invalid(&error.to_string()))?;
    // GDeflate-compressed views read as None.
    let view = |index: usize| {
        let view = &doc.views[index];
        (view.compression.as_deref().unwrap_or("None") == "None")
            .then(|| &file[data + view.offset..][..view.stored_size])
    };

    // Outputs are reordered to base color, ORM and normal so shading reads fixed channels.
    let first = |name: &str| {
        doc.textures
            .iter()
            .find(|t| t.name == name)
            .map(|t| t.first_channel)
    };
    let (Some(base_color), Some(orm), Some(normal)) =
        (first("BaseColor"), first("ORM"), first("Normal"))
    else {
        return Err(invalid("needs BaseColor, ORM and Normal textures"));
    };
    let mut order: Vec<usize> = [base_color, orm, normal]
        .iter()
        .flat_map(|&c| c..c + 3)
        .collect();
    order.extend((0..16).filter(|c| !order.contains(c)).collect::<Vec<_>>());
    let srgb = (0..9)
        .filter(|&o| doc.channels[order[o]].color_space.as_deref() == Some("sRGB"))
        .fold(0, |mask, o| mask | 1 << o);

    let mlp = doc
        .mlp_versions
        .iter()
        .find(|mlp| mlp.layers.iter().all(|l| l.weight_type == "Int8"))
        .ok_or_else(|| invalid("no int8 network"))?;
    if mlp
        .layers
        .iter()
        .map(|l| (l.input_channels, l.output_channels))
        .ne(LAYERS)
    {
        return Err(invalid("unsupported network shape"));
    }
    let (mut weights, mut scales, mut biases) = (Vec::new(), Vec::new(), Vec::new());
    for (index, (layer, (inputs, outputs))) in mlp.layers.iter().zip(LAYERS).enumerate() {
        let [Some(w), Some(s), Some(b)] =
            [layer.weight_view, layer.scale_view, layer.bias_view].map(view)
        else {
            return Err(invalid("compressed network"));
        };
        let last = index == LAYERS.len() - 1;
        for row in 0..outputs {
            let row = if last { order[row] } else { row };
            let w = &w[row * inputs..][..inputs];
            let scale = f32::from_le_bytes(s[4 * row..][..4].try_into().unwrap());
            scales.extend_from_slice(&(if last { scale } else { 80.0 * scale }).to_le_bytes());
            let bias = i32::from_le_bytes(b[4 * row..][..4].try_into().unwrap());
            if !last && bias.unsigned_abs() >= (1 << 22) - (inputs << 14) as u32 {
                return Err(invalid("hidden sums can exceed 2^22"));
            }
            let offset = if last { 0 } else { MAGIC.to_bits() as i32 };
            biases.extend_from_slice(&(bias + offset).to_le_bytes());
            if !MMA {
                weights.extend_from_slice(w);
                continue;
            }
            // Inputs in the previous layer's accumulator order; see `hidden_layer`.
            let source = |k: usize| k / 16 * 16 + k % 4 / 2 * 8 + k % 16 / 4 * 2 + k % 2;
            let w: Vec<u8> = (0..inputs)
                .map(|k| w[if index == 0 { k } else { source(k) }])
                .collect();
            // Lane t's fragment words, inputs 16c + 4t.. of each chunk c, padded; see `tile`.
            let chunks = inputs / 16;
            for t in 0..4 {
                for c in 0..lane_words(chunks) {
                    weights.extend(if c < chunks {
                        &w[16 * c + 4 * t..][..4]
                    } else {
                        &[0; 4]
                    });
                }
            }
        }
    }
    let vectors = if MMA {
        scales
            .chunks_exact(8)
            .zip(biases.chunks_exact(8))
            .flat_map(|(s, b)| [s, b].concat())
            .collect()
    } else {
        [scales, biases].concat()
    };
    let mut bytes = [weights, vectors].concat();

    if doc.latent_shape.num_features != 16 {
        return Err(invalid("needs 16 latent features"));
    }
    let mut latents = [Latent::default(); MAX_LATENTS];
    for (level, image) in doc.latents.iter().enumerate() {
        let Some(layers) = image
            .layer_views
            .iter()
            .map(|&v| view(v))
            .collect::<Option<Vec<_>>>()
        else {
            continue;
        };
        latents[level] = Latent {
            offset: (bytes.len() / 4) as u32,
            width: image.width,
            height: image.height,
        };
        // Feature f of a texel lands in bits 4f..4f+4 of its 64-bit word.
        for texel in 0..(image.width * image.height) as usize {
            for layer in &layers {
                bytes.extend_from_slice(&layer[2 * texel..][..2]);
            }
        }
    }

    // Finer mips than the last one with an unloaded latent level stay undecoded.
    let mips = doc.num_color_mips;
    let mut color_mips = [ColorMip::default(); MAX_COLOR_MIPS];
    let mut first_mip = 0;
    for (mip, info) in doc.color_mips[..mips].iter().enumerate() {
        match info.latent_mip {
            Some(latent)
                if latents[latent..]
                    .iter()
                    .take(2)
                    .filter(|l| l.width > 0)
                    .count()
                    == 2 =>
            {
                color_mips[mip] = ColorMip {
                    latent: latent as u32,
                    position_lod: info.position_lod,
                    position_scale: info.position_scale,
                }
            }
            _ => first_mip = mip + 1,
        }
    }
    if first_mip == mips {
        return Err(invalid("no decodable color mip"));
    }

    // Word offsets fit in u32 for any texel array that fits GPU memory.
    let offset = texels.len() as u32 * 4;
    bytes.resize(bytes.len().next_multiple_of(16), 0);
    texels.extend(bytes.chunks_exact(16).map(|chunk| {
        Texel(Vec4::from_array(std::array::from_fn(|i| {
            f32::from_bits(u32::from_le_bytes(chunk[4 * i..][..4].try_into().unwrap()))
        })))
    }));
    Ok(NeuralSet {
        offset,
        width: doc.width,
        height: doc.height,
        first_mip: first_mip as u32,
        mips: mips as u32,
        color_mips,
        latents,
        srgb,
    })
}
