use std::io;

use glam::Vec4;

use crate::vertex::{ClipPrimitive, ClipScene};

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

fn intersection(outside: Vec4, inside: Vec4, axis: usize, sign: f64) -> Vec4 {
    let a = distance(outside, axis, sign);
    let b = distance(inside, axis, sign);
    let t = a / (a - b);
    let mut vertex = (outside.as_dvec4() * (1.0 - t) + inside.as_dvec4() * t).as_vec4();
    vertex[axis] = -sign as f32 * vertex.w;
    vertex
}

fn clip_primitive(primitive: &ClipPrimitive) -> io::Result<ClipPrimitive> {
    let mut output = ClipPrimitive {
        positions: Vec::new(),
        triangles: Vec::new(),
    };
    let mut polygon = Vec::with_capacity(9);
    let mut scratch = Vec::with_capacity(9);

    for triangle in &primitive.triangles {
        let vertices = triangle.map(|index| primitive.positions[index as usize]);
        let codes = vertices.map(outcode);
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
                let mut previous_distance = distance(previous, axis, sign);
                for &current in &polygon {
                    let current_distance = distance(current, axis, sign);
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
        polygon.retain(|vertex| vertex.w > 0.0);
        if polygon.len() < 3 {
            continue;
        }
        let end = u32::try_from(output.positions.len() + polygon.len())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Too many clipped vertices"))?;
        let base = end - polygon.len() as u32;
        output.positions.extend_from_slice(&polygon);
        for index in 1..polygon.len() as u32 - 1 {
            output
                .triangles
                .push([base, base + index, base + index + 1]);
        }
    }
    Ok(output)
}
