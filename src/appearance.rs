//! Neural materials after Zeltner et al. 2024 ("Real-Time Neural Appearance Models"), with the
//! default shapes of NVIDIA's neuralappearance code: an 8-channel latent texture, a frame layer
//! predicting two shading frames, an importance sampler and a BSDF decoder, one network per
//! material. Weights and latents are seeded stand-ins, not trained, so images are wrong but
//! every lane does a trained material's work.

use glam::{Vec2, Vec3};

use crate::{
    fragment::uniform,
    rasterization::FragmentAttributes,
    texture::Texel,
    trace::{Site, mark},
};

const LATENTS: usize = 8;
/// Side of the stand-in latent texture all materials share; its values change no work.
const LATENT_SIZE: u32 = 256;
/// Texels of each layer, as rows of 4-weight texels then, with biases, 4-bias texels.
const fn layer(inputs: usize, outputs: usize, bias: bool) -> usize {
    outputs * inputs.div_ceil(4) + if bias { outputs.div_ceil(4) } else { 0 }
}
const SAMPLER: usize = layer(8, 12, false);
const DECODER: usize = SAMPLER + layer(11, 16, true) + layer(16, 16, true) + layer(16, 9, true);
const TEXELS: usize = DECODER + layer(24, 16, true) + layer(16, 16, true) + layer(16, 3, true);
/// The decoder's output activation bound.
const SCALE: f32 = 1e5;

/// A material's network weights and latents, as texels of the texel array.
#[derive(Clone, Copy, PartialEq)]
pub struct Appearance {
    weights: u32,
    latents: u32,
}

/// One shading point after sampling: its frame, latents, learned frames and the scatter weight.
pub(crate) struct Point {
    basis: [Vec3; 3],
    wi: Vec3,
    latents: [f32; LATENTS],
    frames: [[Vec3; 3]; 2],
    /// The sampled direction's BSDF times cosine over its density.
    pub weight: Vec3,
}

/// Appends the stand-in latent texture and returns its first texel.
pub fn load_latents(texels: &mut Vec<Texel>) -> u32 {
    let first = texels.len() as u32;
    let mut state = 0x9e37_79b9;
    for _ in 0..2 * LATENT_SIZE * LATENT_SIZE {
        texels.push(Texel(
            std::array::from_fn(|_| 2.0 * uniform(&mut state) - 1.0).into(),
        ));
    }
    first
}

impl Appearance {
    /// Appends seeded stand-in weights: uniform in +-1 / sqrt(inputs), small for the frame
    /// layer so frames stay near the surface's, and decoder output biases of -2, so the BSDF
    /// times cosine starts near e^-2.
    pub fn load(seed: u32, latents: u32, texels: &mut Vec<Texel>) -> Self {
        let weights = texels.len() as u32;
        let mut state = seed.wrapping_mul(0x85eb_ca6b) ^ 0xc2b2_ae35;
        let mut words = Vec::new();
        let layers: [(usize, usize, bool, f32, f32); 7] = [
            (8, 12, false, 0.1, 0.0),
            (11, 16, true, 1.0, 0.0),
            (16, 16, true, 1.0, 0.0),
            (16, 9, true, 1.0, 0.0),
            (24, 16, true, 1.0, 0.0),
            (16, 16, true, 1.0, 0.0),
            (16, 3, true, 1.0, -2.0),
        ];
        for (inputs, outputs, bias, gain, offset) in layers {
            let range = gain / (inputs as f32).sqrt();
            for _ in 0..outputs {
                for i in 0..inputs.div_ceil(4) * 4 {
                    let weight = range * (2.0 * uniform(&mut state) - 1.0);
                    words.push(if i < inputs { weight } else { 0.0 });
                }
            }
            if bias {
                words.extend((0..outputs.div_ceil(4) * 4).map(|_| offset));
            }
        }
        texels.extend(
            words
                .chunks(4)
                .map(|w| Texel([w[0], w[1], w[2], w[3]].into())),
        );
        debug_assert_eq!(texels.len() as u32 - weights, TEXELS as u32);
        Self { weights, latents }
    }

    /// Samples the latents, predicts the frames, samples a direction from the sampler's
    /// mixture and decodes the BSDF there, as a hit does to continue its path.
    #[inline(always)]
    pub(crate) fn sample(
        &self,
        texels: &[Texel],
        attributes: &FragmentAttributes,
        normal: Vec3,
        view: Vec3,
        sites: &mut u32,
    ) -> Point {
        mark(sites, Site::Appearance);
        let tangent = attributes.tangent.truncate();
        let t = (tangent - normal * normal.dot(tangent))
            .try_normalize()
            .unwrap_or_else(|| normal.any_orthonormal_vector());
        let basis = [t, normal.cross(t), normal];
        let wi = local(&basis, view);
        let uv = attributes.uv.uv - attributes.uv.uv.floor();
        let x = ((uv.x * LATENT_SIZE as f32) as u32).min(LATENT_SIZE - 1);
        let y = ((uv.y * LATENT_SIZE as f32) as u32).min(LATENT_SIZE - 1);
        let texel = (self.latents + 2 * (y * LATENT_SIZE + x)) as usize;
        let (a, b) = (texels[texel].0, texels[texel + 1].0);
        let latents = [a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w];
        // Random numbers from the position, so no caller passes a seed.
        let p = attributes.world_position;
        let mut state =
            p.x.to_bits() ^ p.y.to_bits().rotate_left(11) ^ p.z.to_bits().rotate_left(22);
        let u = Vec3::new(
            uniform(&mut state),
            uniform(&mut state),
            uniform(&mut state),
        );

        mark(sites, Site::AppearanceNetwork);
        let base = self.weights as usize;
        let shift: [f32; 12] = dense(texels, base, &latents, false);
        let frames = [0, 1].map(|i| {
            let n = (Vec3::from_slice(&shift[3 * i..]) + Vec3::Z).normalize();
            let t = (Vec3::from_slice(&shift[3 * i + 6..]) + Vec3::X).normalize();
            [t, n.cross(t), n]
        });
        let mut input = [0.0; 11];
        input[..3].copy_from_slice(&wi.to_array());
        input[3..].copy_from_slice(&latents);
        let hidden = dense::<11, 16>(texels, base + SAMPLER, &input, true).map(leaky_relu);
        let hidden = dense::<16, 16>(texels, base + SAMPLER + layer(11, 16, true), &hidden, true)
            .map(leaky_relu);
        let o: [f32; 9] = dense(texels, base + DECODER - layer(16, 9, true), &hidden, true);
        let lobes = Lobes::new(o);
        let wo = lobes.sample(wi, u);
        let pdf = lobes.pdf(wi, wo);
        let f = self.decode(texels, &latents, &frames, wi, wo);
        let weight = if wi.z > 0.0 && wo.z > 0.0 {
            f / pdf
        } else {
            Vec3::ZERO
        };
        Point {
            basis,
            wi,
            latents,
            frames,
            weight,
        }
    }

    /// The BSDF times cosine toward a light, for a lit point.
    #[inline(always)]
    pub(crate) fn eval(
        &self,
        texels: &[Texel],
        point: &Point,
        light: Vec3,
        sites: &mut u32,
    ) -> Vec3 {
        mark(sites, Site::AppearanceLight);
        let wo = local(&point.basis, light);
        if point.wi.z <= 0.0 || wo.z <= 0.0 {
            return Vec3::ZERO;
        }
        self.decode(texels, &point.latents, &point.frames, point.wi, wo)
    }

    /// The decoder: half and difference vectors (the stable Rusinkiewicz form), each direction
    /// in both learned frames, then a 24-16-16-3 SmeLU network.
    #[inline(always)]
    fn decode(
        &self,
        texels: &[Texel],
        latents: &[f32; LATENTS],
        frames: &[[Vec3; 3]; 2],
        wi: Vec3,
        wo: Vec3,
    ) -> Vec3 {
        let wh = (wi + wo).normalize_or_zero();
        let v = Vec3::new(wh.y, -wh.x, 0.0);
        let wd = wi * wh.z + v.cross(wi) + v * (v.dot(wi) / (1.0 + wh.z));
        let mut input = [0.0; 24];
        for (f, frame) in frames.iter().enumerate() {
            input[3 * f..3 * f + 3].copy_from_slice(&local(frame, wh).to_array());
            input[6 + 3 * f..9 + 3 * f].copy_from_slice(&local(frame, wd).to_array());
            input[12 + f] = frame[2].dot(wi);
            input[14 + f] = frame[2].dot(wo);
        }
        input[16..].copy_from_slice(latents);
        let base = self.weights as usize + DECODER;
        let hidden = dense::<24, 16>(texels, base, &input, true).map(smelu);
        let hidden = dense::<16, 16>(texels, base + layer(24, 16, true), &hidden, true).map(smelu);
        let o: [f32; 3] = dense(
            texels,
            base + layer(24, 16, true) + layer(16, 16, true),
            &hidden,
            true,
        );
        Vec3::from_array(o.map(scaled_sigmoid))
    }
}

/// The sampler's diffuse and anisotropic GGX lobes, each with a slope offset.
struct Lobes {
    alpha: Vec2,
    rho: f32,
    mu: Vec2,
    diffuse_mu: Vec2,
    specular: f32,
    diffuse: f32,
}

impl Lobes {
    #[inline(always)]
    fn new(o: [f32; 9]) -> Self {
        let tanh = |x: f32| x / (1.0 + x * x).sqrt();
        let sinh = |x: f32| x * (1.0 + x * x).sqrt();
        let m = o[7].max(o[8]).max(0.0);
        let (specular, diffuse) = ((o[7] - m).exp(), (o[8] - m).exp());
        let norm = 1.0 / (specular + diffuse);
        Self {
            alpha: 1e-4 + 0.5 * (1.0 + Vec2::new(tanh(o[0]), tanh(o[1]))),
            rho: tanh(o[2]),
            mu: Vec2::new(sinh(o[3]), sinh(o[4])),
            diffuse_mu: Vec2::new(sinh(o[5]), sinh(o[6])),
            specular: specular * norm,
            diffuse: diffuse * norm,
        }
    }

    #[inline(always)]
    fn sample(&self, wi: Vec3, u: Vec3) -> Vec3 {
        if u.x < self.specular {
            let s = (u.y / (1.0 - u.y)).sqrt();
            let (sin, cos) = (std::f32::consts::TAU * u.z).sin_cos();
            let standard = Vec2::new(s * cos, s * sin);
            let correlated = (1.0 - self.rho * self.rho).max(1e-4).sqrt();
            let slope = self.mu
                + Vec2::new(
                    self.alpha.x * standard.x,
                    self.alpha.y * (self.rho * standard.x + correlated * standard.y),
                );
            let wh = safe_normalize(Vec3::new(-slope.x, -slope.y, 1.0));
            2.0 * wi.dot(wh) * wh - wi
        } else {
            // Shirley's concentric disk, lifted to the cosine lobe around the offset normal.
            let d = 2.0 * Vec2::new(u.y, u.z) - 1.0;
            let (r, phi) = if d.x.abs() > d.y.abs() {
                (d.x, std::f32::consts::FRAC_PI_4 * (d.y / d.x))
            } else if d.y != 0.0 {
                (
                    d.y,
                    std::f32::consts::FRAC_PI_2 - std::f32::consts::FRAC_PI_4 * (d.x / d.y),
                )
            } else {
                (0.0, 0.0)
            };
            let (sin, cos) = phi.sin_cos();
            let disk = r * Vec2::new(cos, sin);
            let z = (1.0 - disk.length_squared()).max(0.0).sqrt();
            let n = safe_normalize(Vec3::new(-self.diffuse_mu.x, -self.diffuse_mu.y, 1.0));
            let s = safe_normalize(Vec3::X - n * n.x);
            disk.x * s + disk.y * n.cross(s) + z * n
        }
    }

    /// The mixture's density, kept above zero far from both lobes.
    #[inline(always)]
    fn pdf(&self, wi: Vec3, wo: Vec3) -> f32 {
        let mut pdf = 0.0;
        if wi.z > 0.0 && wo.z > 0.0 {
            let wh = safe_normalize(wi + wo);
            let wh = wh * wh.z.signum();
            let correlated = (1.0 - self.rho * self.rho).max(1e-4).sqrt();
            let slope = -Vec2::new(wh.x, wh.y) / wh.z.max(1e-4) - self.mu;
            let x = slope.x / self.alpha.x;
            let norm = 1.0 / (self.alpha.x * self.alpha.y * correlated).max(1e-6);
            let y = (self.alpha.x * slope.y - self.rho * self.alpha.y * slope.x) * norm;
            let r2 = x * x + y * y;
            let p22 = norm / (std::f32::consts::PI * (1.0 + r2) * (1.0 + r2)).max(1e-4);
            let ndf = p22 / (wh.z * wh.z * wh.z * wh.z).max(1e-4);
            pdf += self.specular * ndf * wh.z / (4.0 * wi.dot(wh).abs()).max(1e-4);
        }
        if wo.z >= 0.0 {
            let n = safe_normalize(Vec3::new(-self.diffuse_mu.x, -self.diffuse_mu.y, 1.0));
            pdf += self.diffuse * wo.dot(n).max(0.0) * std::f32::consts::FRAC_1_PI;
        }
        pdf.max(1e-4)
    }
}

/// A fully connected layer of texel rows, with each output's bias after the rows.
#[inline(always)]
fn dense<const I: usize, const O: usize>(
    texels: &[Texel],
    first: usize,
    input: &[f32; I],
    bias: bool,
) -> [f32; O] {
    let row = I.div_ceil(4);
    std::array::from_fn(|o| {
        let mut sum = if bias {
            texels[first + O * row + o / 4].0[o % 4]
        } else {
            0.0
        };
        for i in 0..I {
            sum += texels[first + o * row + i / 4].0[i % 4] * input[i];
        }
        sum
    })
}

#[inline(always)]
fn local(frame: &[Vec3; 3], v: Vec3) -> Vec3 {
    Vec3::new(v.dot(frame[0]), v.dot(frame[1]), v.dot(frame[2]))
}

#[inline(always)]
fn safe_normalize(v: Vec3) -> Vec3 {
    v / v.length().max(1e-4)
}

#[inline(always)]
fn leaky_relu(x: f32) -> f32 {
    x.max(0.0) + x.min(0.0) * 0.01
}

/// Leaky smooth ReLU: slope 0.01 below -0.5, 1 above 0.5, quadratic between.
#[inline(always)]
fn smelu(x: f32) -> f32 {
    let (alpha, beta) = (0.5, 0.01);
    let (a, b, c) = (
        (1.0 - beta) / (4.0 * alpha),
        (1.0 + beta) / 2.0,
        alpha * (1.0 - beta) / 4.0,
    );
    (beta * x)
        .max(x + alpha * (1.0 - beta))
        .min((a * x + b) * x + c)
        .min(x.max(alpha))
}

/// Sigmoid scaled to `SCALE`, near e^x below zero.
#[inline(always)]
fn scaled_sigmoid(x: f32) -> f32 {
    let e = (-x.abs()).exp();
    if x > 0.0 {
        SCALE / (SCALE * e + 1.0)
    } else {
        SCALE * e / (e + SCALE)
    }
}
