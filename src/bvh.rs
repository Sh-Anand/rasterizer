use glam::Vec3;

const LEAF_TRIANGLES: usize = 4;
// Median splits keep depth near log2(triangles / 4), far below this bound.
const STACK: usize = 64;

/// Leaves hold `count` entries of the triangle order from `first`; interior nodes have
/// `count == 0` and children `first` and `first + 1`.
#[derive(Clone, Copy)]
pub(crate) struct Node {
    min: Vec3,
    first: u32,
    max: Vec3,
    count: u32,
}

pub(crate) struct Hit {
    pub triangle: u32,
    pub t: f32,
    pub barycentric: [f32; 3],
}

/// Splits at the median centroid of the widest axis. Returns nodes and the triangle order
/// leaves index; an empty scene has no nodes.
pub(crate) fn build(triangles: &[[Vec3; 3]]) -> (Vec<Node>, Vec<u32>) {
    let mut order: Vec<u32> = (0..triangles.len() as u32).collect();
    if triangles.is_empty() {
        return (Vec::new(), order);
    }
    let centroids: Vec<Vec3> = triangles
        .iter()
        .map(|t| (t[0] + t[1] + t[2]) / 3.0)
        .collect();
    let empty = Node {
        min: Vec3::ZERO,
        first: 0,
        max: Vec3::ZERO,
        count: 0,
    };
    let mut nodes = vec![empty];
    let mut pending = vec![(0, 0, order.len())];
    while let Some((node, start, end)) = pending.pop() {
        let range = &mut order[start..end];
        let (mut min, mut max) = (Vec3::INFINITY, Vec3::NEG_INFINITY);
        let (mut centroid_min, mut centroid_max) = (Vec3::INFINITY, Vec3::NEG_INFINITY);
        for &i in range.iter() {
            for v in triangles[i as usize] {
                min = min.min(v);
                max = max.max(v);
            }
            centroid_min = centroid_min.min(centroids[i as usize]);
            centroid_max = centroid_max.max(centroids[i as usize]);
        }
        if range.len() <= LEAF_TRIANGLES {
            nodes[node] = Node {
                min,
                first: start as u32,
                max,
                count: range.len() as u32,
            };
            continue;
        }
        let extent = centroid_max - centroid_min;
        let axis = if extent.x >= extent.y && extent.x >= extent.z {
            0
        } else if extent.y >= extent.z {
            1
        } else {
            2
        };
        let half = range.len() / 2;
        range.select_nth_unstable_by(half, |&a, &b| {
            centroids[a as usize][axis].total_cmp(&centroids[b as usize][axis])
        });
        let left = nodes.len();
        nodes.extend([empty, empty]);
        nodes[node] = Node {
            min,
            first: left as u32,
            max,
            count: 0,
        };
        pending.extend([(left, start, start + half), (left + 1, start + half, end)]);
    }
    (nodes, order)
}

/// Closest hit beyond the origin; `triangle` returns a triangle's world-space corners.
#[inline(always)]
pub(crate) fn closest_hit(
    nodes: &[Node],
    order: &[u32],
    origin: Vec3,
    direction: Vec3,
    triangle: impl Fn(u32) -> [Vec3; 3],
) -> Option<Hit> {
    if nodes.is_empty() {
        return None;
    }
    let inverse = direction.recip();
    let mut stack = [0u32; STACK];
    let mut depth = 1;
    let mut closest = f32::INFINITY;
    let mut hit = None;
    while depth > 0 {
        depth -= 1;
        let node = nodes[stack[depth] as usize];
        let t0 = (node.min - origin) * inverse;
        let t1 = (node.max - origin) * inverse;
        if t0.min(t1).max_element().max(0.0) > t0.max(t1).min_element().min(closest) {
            continue;
        }
        if node.count == 0 {
            if depth + 2 <= STACK {
                stack[depth] = node.first;
                stack[depth + 1] = node.first + 1;
                depth += 2;
            }
            continue;
        }
        for i in node.first..node.first + node.count {
            let index = order[i as usize];
            let [a, b, c] = triangle(index);
            // Moller-Trumbore intersection.
            let (e1, e2) = (b - a, c - a);
            let p = direction.cross(e2);
            let inverse_det = 1.0 / e1.dot(p);
            let s = origin - a;
            let u = s.dot(p) * inverse_det;
            let q = s.cross(e1);
            let v = direction.dot(q) * inverse_det;
            let t = e2.dot(q) * inverse_det;
            if u >= 0.0 && v >= 0.0 && u + v <= 1.0 && t > 0.0 && t < closest {
                closest = t;
                hit = Some(Hit {
                    triangle: index,
                    t,
                    barycentric: [1.0 - u - v, u, v],
                });
            }
        }
    }
    hit
}
