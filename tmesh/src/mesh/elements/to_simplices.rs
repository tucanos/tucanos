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
#[allow(clippy::too_many_lines)]
pub fn hex2tets<T: Idx>(hex: &Hexahedron<T>) -> ([Tetrahedron<T>; 5], Option<Tetrahedron<T>>) {
    let imin = argmin(&hex.0);

    let mut idx = [0; 8];
    for (i, v) in idx.iter_mut().enumerate() {
        *v = hex.get(INDIRECTION_HEX[imin][i]);
    }

    let i1 = u32::from(compare_edges(idx[1], idx[6], idx[2], idx[5]));
    let i2 = u32::from(compare_edges(idx[3], idx[6], idx[2], idx[7]));
    let i3 = u32::from(compare_edges(idx[4], idx[6], idx[5], idx[7]));
    let flg = i1 + 2 * i2 + 4 * i3;
    let rot = ROTATION_HEX[flg as usize];

    let mut idx2 = idx;
    if rot == 120 {
        for i in 0..8 {
            idx2[i] = idx[PERM_120[i]];
        }
    } else if rot == 240 {
        for i in 0..8 {
            idx2[i] = idx[PERM_240[i]];
        }
    }
    let flg2 = i1 + i2 + i3;

    if flg2 == 0 {
        let tet1 = Tetrahedron::new(idx2[0], idx2[1], idx2[2], idx2[5]);
        let tet2 = Tetrahedron::new(idx2[0], idx2[2], idx2[7], idx2[5]);
        let tet3 = Tetrahedron::new(idx2[0], idx2[2], idx2[3], idx2[7]);
        let tet4 = Tetrahedron::new(idx2[0], idx2[5], idx2[7], idx2[4]);
        let tet5 = Tetrahedron::new(idx2[2], idx2[7], idx2[5], idx2[6]);
        return ([tet1, tet2, tet3, tet4, tet5], None);
    } else if flg2 == 1 {
        let tet1 = Tetrahedron::new(idx2[0], idx2[5], idx2[7], idx2[4]);
        let tet2 = Tetrahedron::new(idx2[0], idx2[1], idx2[7], idx2[5]);
        let tet3 = Tetrahedron::new(idx2[1], idx2[6], idx2[7], idx2[5]);
        let tet4 = Tetrahedron::new(idx2[0], idx2[7], idx2[2], idx2[3]);
        let tet5 = Tetrahedron::new(idx2[0], idx2[7], idx2[1], idx2[2]);
        let tet6 = Tetrahedron::new(idx2[1], idx2[7], idx2[6], idx2[2]);
        return ([tet1, tet2, tet3, tet4, tet5], Some(tet6));
    } else if flg2 == 2 {
        let tet1 = Tetrahedron::new(idx2[0], idx2[4], idx2[5], idx2[6]);
        let tet2 = Tetrahedron::new(idx2[0], idx2[3], idx2[7], idx2[6]);
        let tet3 = Tetrahedron::new(idx2[0], idx2[7], idx2[4], idx2[6]);
        let tet4 = Tetrahedron::new(idx2[0], idx2[1], idx2[2], idx2[5]);
        let tet5 = Tetrahedron::new(idx2[0], idx2[3], idx2[6], idx2[2]);
        let tet6 = Tetrahedron::new(idx2[0], idx2[6], idx2[5], idx2[2]);
        return ([tet1, tet2, tet3, tet4, tet5], Some(tet6));
    } else if flg2 == 3 {
        let tet1 = Tetrahedron::new(idx2[0], idx2[2], idx2[3], idx2[6]);
        let tet2 = Tetrahedron::new(idx2[0], idx2[3], idx2[7], idx2[6]);
        let tet3 = Tetrahedron::new(idx2[0], idx2[7], idx2[4], idx2[6]);
        let tet4 = Tetrahedron::new(idx2[0], idx2[5], idx2[6], idx2[4]);
        let tet5 = Tetrahedron::new(idx2[1], idx2[5], idx2[6], idx2[0]);
        let tet6 = Tetrahedron::new(idx2[1], idx2[6], idx2[2], idx2[0]);
        return ([tet1, tet2, tet3, tet4, tet5], Some(tet6));
    }
    unreachable!()
}
