use std::{collections::HashMap, io};

use glam::{Mat3, Vec2, Vec3, Vec4};

use super::ClipVertex;
use crate::geometry::Triangle;

pub(super) fn from_vertices(
    tangents: &[Vec4],
    vertex_count: usize,
    triangles: &[Triangle],
) -> io::Result<Vec<[Vec4; 3]>> {
    let invalid = |message| io::Error::new(io::ErrorKind::InvalidData, message);
    if tangents.len() != vertex_count
        || tangents
            .iter()
            .any(|t| !t.is_finite() || t.w.abs() != 1.0 || t.truncate().try_normalize().is_none())
    {
        return Err(invalid("Invalid vertex tangents"));
    }
    triangles
        .iter()
        .map(|triangle| {
            let [a, b, c] = triangle.indices.map(|i| tangents[i as usize]);
            if a.w != b.w || a.w != c.w {
                return Err(invalid("Triangle tangent handedness must be uniform"));
            }
            Ok([a, b, c])
        })
        .collect()
}

struct Geometry<'a> {
    positions: &'a [Vec3],
    normals: Option<&'a [Vec3]>,
    uvs: &'a [Vec2],
    triangles: &'a [Triangle],
    tangents: Vec<[Vec4; 3]>,
}

impl mikktspace::Geometry for Geometry<'_> {
    fn num_faces(&self) -> usize {
        self.triangles.len()
    }

    fn num_vertices_of_face(&self, _: usize) -> usize {
        3
    }

    fn position(&self, face: usize, vert: usize) -> [f32; 3] {
        self.positions[self.triangles[face].indices[vert] as usize].to_array()
    }

    fn normal(&self, face: usize, vert: usize) -> [f32; 3] {
        let triangle = &self.triangles[face];
        let normal = if let Some(normals) = self.normals {
            normals[triangle.indices[vert] as usize]
        } else {
            let [a, b, c] = triangle.indices.map(|i| self.positions[i as usize]);
            (b - a).cross(c - a)
        };
        normal.normalize_or_zero().to_array()
    }

    fn tex_coord(&self, face: usize, vert: usize) -> [f32; 2] {
        self.uvs[self.triangles[face].indices[vert] as usize].to_array()
    }

    fn set_tangent_encoded(&mut self, tangent: [f32; 4], face: usize, vert: usize) {
        self.tangents[face][vert] = Vec4::from_array(tangent);
    }
}

pub(super) fn generate(
    positions: &[Vec3],
    normals: Option<&[Vec3]>,
    uvs: &[Vec2],
    triangles: &[Triangle],
) -> Vec<[Vec4; 3]> {
    let mut geometry = Geometry {
        positions,
        normals,
        uvs,
        triangles,
        tangents: vec![[Vec4::ZERO; 3]; triangles.len()],
    };
    let has_surface = triangles.iter().any(|triangle| {
        let [a, b, c] = triangle.indices.map(|i| positions[i as usize]);
        (b - a).cross(c - a).try_normalize().is_some()
    });
    if has_surface && !mikktspace::generate_tangents(&mut geometry) {
        geometry.tangents.fill([Vec4::ZERO; 3]);
    }
    geometry.tangents
}

pub(super) fn apply(
    vertices: &mut Vec<ClipVertex>,
    triangles: &mut [Triangle],
    tangents: &[[Vec4; 3]],
    linear: Mat3,
) -> io::Result<()> {
    let sign = linear.determinant().signum();
    let mut assigned = vec![false; vertices.len()];
    let mut variants = HashMap::new();
    for (triangle, tangents) in triangles.iter_mut().zip(tangents) {
        for (index, tangent) in triangle.indices.iter_mut().zip(tangents) {
            let tangent = (linear * tangent.truncate())
                .try_normalize()
                .filter(|_| tangent.w.abs() == 1.0)
                .map_or(Vec4::ZERO, |t| t.extend(tangent.w * sign));
            let original = *index;
            if !assigned[original as usize] {
                vertices[original as usize].tangent = tangent;
                assigned[original as usize] = true;
            } else if vertices[original as usize].tangent != tangent {
                // A shared position can need different tangent frames across a UV seam.
                let key = (original, tangent.to_array().map(f32::to_bits));
                *index = if let Some(&index) = variants.get(&key) {
                    index
                } else {
                    let index = u32::try_from(vertices.len()).map_err(|_| {
                        io::Error::new(io::ErrorKind::InvalidData, "Too many tangent vertices")
                    })?;
                    let mut vertex = vertices[original as usize];
                    vertex.tangent = tangent;
                    vertices.push(vertex);
                    variants.insert(key, index);
                    index
                };
            }
        }
    }
    Ok(())
}
