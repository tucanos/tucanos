use crate::mesh::{Tetrahedron, Triangle, elements::Idx};

use super::{Hexahedron, Prism, Pyramid, Quadrangle};

// Subdivision of standard elements to triangles and tetrahedra maintaining a consistent mesh. The algorithms are taken from
// How to Subdivide Pyramids, Prisms and Hexahedra into Tetrahedra
// Julien Dompierre Paul Labbé Marie-Gabrielle Vallet Ricardo Camarero

const INDIRECTION_PRI: [[usize; 6]; 6] = [
    [0, 1, 2, 3, 4, 5],
    [1, 2, 0, 4, 5, 3],
    [2, 0, 1, 5, 3, 4],
    [3, 5, 4, 0, 2, 1],
    [4, 3, 5, 1, 0, 2],
    [5, 4, 3, 2, 1, 0],
];

const INDIRECTION_HEX: [[usize; 8]; 8] = [
    [0, 1, 2, 3, 4, 5, 6, 7],
    [1, 0, 4, 5, 2, 3, 7, 6],
    [2, 1, 5, 6, 3, 0, 4, 7],
    [3, 0, 1, 2, 7, 4, 5, 6],
    [4, 0, 3, 7, 5, 1, 2, 6],
    [5, 1, 0, 4, 6, 2, 3, 7],
    [6, 2, 1, 5, 7, 3, 0, 4],
    [7, 3, 2, 6, 4, 0, 1, 5],
];

const ROTATION_HEX: [usize; 8] = [0, 0, 240, 120, 120, 240, 0, 0];
const PERM_120: [usize; 8] = [0, 4, 5, 1, 3, 7, 6, 2];
const PERM_240: [usize; 8] = [0, 3, 7, 4, 1, 2, 6, 5];

fn compare_edges(i0: usize, i1: usize, i2: usize, i3: usize) -> bool {
    usize::min(i0, i1) < usize::min(i2, i3)
}

fn argmin<T: Idx>(arr: &[T]) -> usize {
    let mut imin = 0;
    let mut idx = arr[0];

    for (i, j) in arr.iter().copied().enumerate().skip(1) {
        if j < idx {
            idx = j;
            imin = i;
        }
    }

    imin
}

/// Convert a quadrangle into 2 triangles
#[must_use]
pub fn qua2tris<T: Idx>(quad: &Quadrangle<T>) -> [Triangle<T>; 2] {
    if compare_edges(quad.get(0), quad.get(2), quad.get(1), quad.get(3)) {
        [
            Triangle::new(quad.get(0), quad.get(1), quad.get(2)),
            Triangle::new(quad.get(0), quad.get(2), quad.get(3)),
        ]
    } else {
        [
            Triangle::new(quad.get(1), quad.get(2), quad.get(3)),
            Triangle::new(quad.get(1), quad.get(3), quad.get(0)),
        ]
    }
}

/// Convert a pyramid into 2 tetrahedra
#[must_use]
pub fn pyr2tets<T: Idx>(pyr: &Pyramid<T>) -> [Tetrahedron<T>; 2] {
    if compare_edges(pyr.get(0), pyr.get(2), pyr.get(1), pyr.get(3)) {
        [
            Tetrahedron::new(pyr.get(0), pyr.get(1), pyr.get(2), pyr.get(4)),
            Tetrahedron::new(pyr.get(0), pyr.get(2), pyr.get(3), pyr.get(4)),
        ]
    } else {
        [
            Tetrahedron::new(pyr.get(1), pyr.get(2), pyr.get(3), pyr.get(4)),
            Tetrahedron::new(pyr.get(1), pyr.get(3), pyr.get(0), pyr.get(4)),
        ]
    }
}

/// Convert a prism into 3 tetrahedra
#[must_use]
pub fn pri2tets<T: Idx>(pri: &Prism<T>) -> [Tetrahedron<T>; 3] {
    let imin = argmin(&pri.0);

    let mut idx = [0; 6];
    for (i, v) in idx.iter_mut().enumerate() {
        *v = pri.get(INDIRECTION_PRI[imin][i]);
    }

    let (tet1, tet2) = if compare_edges(idx[1], idx[5], idx[2], idx[4]) {
        (
            Tetrahedron::new(idx[0], idx[1], idx[2], idx[5]),
            Tetrahedron::new(idx[0], idx[1], idx[5], idx[4]),
        )
    } else {
        (
            Tetrahedron::new(idx[0], idx[1], idx[2], idx[4]),
            Tetrahedron::new(idx[0], idx[4], idx[2], idx[5]),
        )
    };
    let tet3 = Tetrahedron::new(idx[0], idx[4], idx[5], idx[3]);
    [tet1, tet2, tet3]
}

/// Convert a hex into 5 or 6 tetrahedra
#[must_use]
pub fn hex2tets<T: Idx>(hex: &Hexahedron<T>) -> ([Tetrahedron<T>; 5], Option<Tetrahedron<T>>) {
    let idx = INDIRECTION_HEX[argmin(&hex.0)].map(|i| hex.get(i));
    let i1 = usize::from(compare_edges(idx[1], idx[6], idx[2], idx[5]));
    let i2 = usize::from(compare_edges(idx[3], idx[6], idx[2], idx[7]));
    let i3 = usize::from(compare_edges(idx[4], idx[6], idx[5], idx[7]));

    let idx2 = match ROTATION_HEX[i1 + 2 * i2 + 4 * i3] {
        120 => PERM_120.map(|i| idx[i]),
        240 => PERM_240.map(|i| idx[i]),
        _ => idx,
    };

    let to_tet = |[a, b, c, d]: [usize; 4]| Tetrahedron::new(idx2[a], idx2[b], idx2[c], idx2[d]);
    match i1 + i2 + i3 {
        0 => (
            [
                [0, 1, 2, 5],
                [0, 2, 7, 5],
                [0, 2, 3, 7],
                [0, 5, 7, 4],
                [2, 7, 5, 6],
            ]
            .map(to_tet),
            None,
        ),
        1 => (
            [
                [0, 5, 7, 4],
                [0, 1, 7, 5],
                [1, 6, 7, 5],
                [0, 7, 2, 3],
                [0, 7, 1, 2],
            ]
            .map(to_tet),
            Some(to_tet([1, 7, 6, 2])),
        ),
        2 => (
            [
                [0, 4, 5, 6],
                [0, 3, 7, 6],
                [0, 7, 4, 6],
                [0, 1, 2, 5],
                [0, 3, 6, 2],
            ]
            .map(to_tet),
            Some(to_tet([0, 6, 5, 2])),
        ),
        3 => (
            [
                [0, 2, 3, 6],
                [0, 3, 7, 6],
                [0, 7, 4, 6],
                [0, 5, 6, 4],
                [1, 5, 6, 0],
            ]
            .map(to_tet),
            Some(to_tet([1, 6, 2, 0])),
        ),
        _ => unreachable!(),
    }
}
