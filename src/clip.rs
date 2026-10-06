use std::io;

use glam::Vec4;

use crate::{
    geometry::Triangle,
    vertex::{ClipPrimitive, ClipScene, ClipVertex},
};

// Left, right, bottom, top, near, far: w + sign * coordinate >= 0.
const PLANES: [(usize, f64); 6] = [
    (0, 1.0),
    (0, -1.0),
    (1, 1.0),
    (1, -1.0),
    (2, 1.0),
    (2, -1.0),
];

pub fn clip_scene(scene: &ClipScene) -> io::Result<ClipScene> {
    Ok(ClipScene {
        scene_index: scene.scene_index,
        primitives: scene
            .primitives
            .iter()
            .map(clip_primitive)
            .collect::<io::Result<_>>()?,
    })
}

fn distance(vertex: Vec4, axis: usize, sign: f64) -> f64 {
    f64::from(vertex.w) + sign * f64::from(vertex[axis])
}

fn outcode(vertex: Vec4) -> u8 {
    let mut code = 0;
    for (plane, (axis, sign)) in PLANES.into_iter().enumerate() {
        if distance(vertex, axis, sign) < 0.0 {
            code |= 1 << plane;
        }
    }
    code
}

fn intersection(outside: ClipVertex, inside: ClipVertex, axis: usize, sign: f64) -> ClipVertex {
    let a = distance(outside.position, axis, sign);
    let b = distance(inside.position, axis, sign);
    let t = a / (a - b);
    let mut position =
        (outside.position.as_dvec4() * (1.0 - t) + inside.position.as_dvec4() * t).as_vec4();
    position[axis] = -sign as f32 * position.w;
    let world_position = (outside.world_position.as_dvec3() * (1.0 - t)
        + inside.world_position.as_dvec3() * t)
        .as_vec3();
    let uv = (outside.uv.as_dvec2() * (1.0 - t) + inside.uv.as_dvec2() * t).as_vec2();
    let emissive_uv =
        (outside.emissive_uv.as_dvec2() * (1.0 - t) + inside.emissive_uv.as_dvec2() * t).as_vec2();
    let lightmap_uv =
        (outside.lightmap_uv.as_dvec2() * (1.0 - t) + inside.lightmap_uv.as_dvec2() * t).as_vec2();
    let normal = (outside.normal.as_dvec3() * (1.0 - t) + inside.normal.as_dvec3() * t).as_vec3();
    ClipVertex {
        position,
        world_position,
        uv,
        emissive_uv,
        lightmap_uv,
        normal,
    }
}

pub(crate) fn clip_primitive(primitive: &ClipPrimitive) -> io::Result<ClipPrimitive> {
    let mut output = ClipPrimitive {
        source: primitive.source,
        mirrored: primitive.mirrored,
        vertices: Vec::new(),
        triangles: Vec::new(),
    };
    let mut polygon = Vec::with_capacity(9);
    let mut scratch = Vec::with_capacity(9);

    for triangle in &primitive.triangles {
        let vertices = triangle
            .indices
            .map(|index| primitive.vertices[index as usize]);
        let codes = vertices.map(|vertex| outcode(vertex.position));
        if codes[0] & codes[1] & codes[2] != 0 {
            continue;
        }
        polygon.clear();
        polygon.extend(vertices);
        if codes[0] | codes[1] | codes[2] != 0 {
            for (axis, sign) in PLANES {
                if polygon.len() < 3 {
                    break;
                }
                scratch.clear();
                let mut previous = *polygon.last().unwrap();
                let mut previous_distance = distance(previous.position, axis, sign);
                for &current in &polygon {
                    let current_distance = distance(current.position, axis, sign);
                    if previous_distance < 0.0 && current_distance > 0.0 {
                        scratch.push(intersection(previous, current, axis, sign));
                    } else if current_distance < 0.0 && previous_distance > 0.0 {
                        scratch.push(intersection(current, previous, axis, sign));
                    }
                    if current_distance >= 0.0 {
                        scratch.push(current);
                    }
                    previous = current;
                    previous_distance = current_distance;
                }
                std::mem::swap(&mut polygon, &mut scratch);
            }
        }

        // The homogeneous origin cannot undergo perspective division.
        polygon.retain(|vertex| vertex.position.w > 0.0);
        if polygon.len() < 3 {
            continue;
        }
        let end = u32::try_from(output.vertices.len() + polygon.len())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Too many clipped vertices"))?;
        let base = end - polygon.len() as u32;
        output.vertices.extend_from_slice(&polygon);
        for index in 1..polygon.len() as u32 - 1 {
            output.triangles.push(Triangle {
                indices: [base, base + index, base + index + 1],
                source_index: triangle.source_index,
            });
        }
    }
    Ok(output)
}
