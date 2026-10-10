use std::{
    fs,
    io::{self, BufReader, BufWriter, Write},
    path::Path,
};

use bincode::{Decode, Encode};
use glam::{Vec2, Vec3};

use crate::{asset::GltfAsset, config::Config, vertex::ClipScene};

#[derive(Encode, Decode)]
pub struct Lightmap {
    pub width: u32,
    pub height: u32,
    pub front: Vec<[f32; 3]>,
    pub back: Vec<[f32; 3]>,
}

impl Lightmap {
    pub fn new(width: u32, height: u32) -> io::Result<Self> {
        let len =
            usize::try_from(u64::from(width) * u64::from(height)).map_err(io::Error::other)?;
        if len == 0 {
            return Err(invalid("Empty lightmap"));
        }
        let mut front = Vec::new();
        front.try_reserve_exact(len).map_err(io::Error::other)?;
        front.resize(len, [0.0; 3]);
        Ok(Self {
            width,
            height,
            back: front.clone(),
            front,
        })
    }

    pub fn sample(&self, uv: Vec2, back: bool) -> Vec3 {
        let p = uv * Vec2::new(self.width as f32, self.height as f32) - Vec2::splat(0.5);
        let base = p.floor();
        let fraction = p - base;
        let (x, y) = (base.x as i64, base.y as i64);
        let values = if back { &self.back } else { &self.front };
        let texel = |x: i64, y: i64| {
            Vec3::from_array(
                values[y.clamp(0, i64::from(self.height) - 1) as usize * self.width as usize
                    + x.clamp(0, i64::from(self.width) - 1) as usize],
            )
        };
        texel(x, y).lerp(texel(x + 1, y), fraction.x).lerp(
            texel(x, y + 1).lerp(texel(x + 1, y + 1), fraction.x),
            fraction.y,
        )
    }
}

#[derive(Encode, Decode)]
pub(crate) struct MeshLayout {
    pub vertices: Vec<(u32, [f32; 2])>,
    pub indices: Vec<u32>,
}

#[derive(Encode, Decode)]
pub struct BakedLighting {
    pub(crate) key: [u8; 32],
    pub(crate) layouts: Vec<MeshLayout>,
    pub lightmap: Lightmap,
}

impl BakedLighting {
    pub fn save(&self, path: &Path) -> io::Result<()> {
        let temporary = path.with_extension("lightmap.tmp");
        let mut writer = BufWriter::new(fs::File::create(&temporary)?);
        bincode::encode_into_std_write(self, &mut writer, bincode::config::standard())
            .map_err(io::Error::other)?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
        fs::rename(temporary, path)
    }

    pub fn load(path: &Path, key: [u8; 32]) -> io::Result<Self> {
        let mut reader = BufReader::new(fs::File::open(path)?);
        let result: Self = bincode::decode_from_std_read(
            &mut reader,
            bincode::config::standard().with_limit::<{ 1024 * 1024 * 1024 }>(),
        )
        .map_err(|error| invalid(&format!("Invalid lighting bake: {error}")))?;
        if result.key != key {
            return Err(invalid("Lighting bake is stale; bake the scene again"));
        }
        let map = &result.lightmap;
        let len = u64::from(map.width) * u64::from(map.height);
        if len == 0
            || len != map.front.len() as u64
            || len != map.back.len() as u64
            || map
                .front
                .iter()
                .chain(&map.back)
                .flatten()
                .any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err(invalid("Invalid lightmap dimensions or values"));
        }
        Ok(result)
    }

    pub fn apply(&self, scene: &mut ClipScene) -> io::Result<()> {
        if scene.primitives.len() != self.layouts.len() {
            return Err(invalid("Lighting bake does not match scene primitives"));
        }
        for (primitive, layout) in scene.primitives.iter_mut().zip(&self.layouts) {
            if layout.indices.len() != primitive.triangles.len() * 3
                || layout
                    .indices
                    .iter()
                    .any(|&i| i as usize >= layout.vertices.len())
                || layout.vertices.iter().any(|&(i, uv)| {
                    i as usize >= primitive.vertices.len()
                        || uv
                            .iter()
                            .any(|u| !u.is_finite() || !(0.0..=1.0).contains(u))
                })
            {
                return Err(invalid("Invalid lightmap mesh layout"));
            }
            primitive.vertices = layout
                .vertices
                .iter()
                .map(|&(index, uv)| {
                    let mut vertex = primitive.vertices[index as usize];
                    vertex.lightmap_uv = Vec2::from_array(uv);
                    vertex
                })
                .collect();
            for (triangle, indices) in primitive
                .triangles
                .iter_mut()
                .zip(layout.indices.chunks_exact(3))
            {
                triangle.indices.copy_from_slice(indices);
            }
        }
        Ok(())
    }
}

pub fn cache_key(asset: &GltfAsset, config: &Config) -> io::Result<[u8; 32]> {
    let mut hash = blake3::Hasher::new();
    hash.update(b"rasterizer-lightmap-v5");
    hash.update(
        &gltf::json::serialize::to_vec(asset.document.as_json()).map_err(io::Error::other)?,
    );
    for buffer in &asset.buffers {
        hash.update(&(buffer.0.len() as u64).to_le_bytes());
        hash.update(&buffer.0);
    }
    for image in &asset.images {
        hash.update(&image.width.to_le_bytes());
        hash.update(&image.height.to_le_bytes());
        hash.update(&(image.format as u32).to_le_bytes());
        hash.update(&(image.pixels.len() as u64).to_le_bytes());
        hash.update(&image.pixels);
    }
    hash.update(
        &gltf::json::serialize::to_vec(&(
            &config.light,
            &config.shadow,
            &config.bake,
            config.sampling,
        ))
        .map_err(io::Error::other)?,
    );
    Ok(*hash.finalize().as_bytes())
}

pub(crate) fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}
