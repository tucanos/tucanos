use crate::{
    topo_elems::{Tetrahedron, Triangle},
    Idx,
};

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

fn compare_edges(i0: Idx, i1: Idx, i2: Idx, i3: Idx) -> bool {
    Idx::min(i0, i1) < Idx::min(i2, i3)
}

fn argmin(arr: &[Idx]) -> usize {
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
pub fn qua2tris(quad: &[Idx]) -> [Triangle; 2] {
    let mut tri1 = Triangle::default();
    let mut tri2 = Triangle::default();

    if compare_edges(quad[0], quad[2], quad[1], quad[3]) {
        tri1[0] = quad[0];
        tri1[1] = quad[1];
        tri1[2] = quad[2];
        tri2[0] = quad[0];
        tri2[1] = quad[2];
        tri2[2] = quad[3];
    } else {
        tri1[0] = quad[1];
        tri1[1] = quad[2];
        tri1[2] = quad[3];
        tri2[0] = quad[1];
        tri2[1] = quad[3];
        tri2[2] = quad[0];
    }

    [tri1, tri2]
}

/// Convert a pyramid into 2 tetrahedra
#[must_use]
pub fn pyr2tets(pyr: &[Idx]) -> [Tetrahedron; 2] {
    let mut tet1 = Tetrahedron::default();
    let mut tet2 = Tetrahedron::default();

    if compare_edges(pyr[0], pyr[2], pyr[1], pyr[3]) {
        tet1[0] = pyr[0];
        tet1[1] = pyr[1];
        tet1[2] = pyr[2];
        tet1[3] = pyr[4];
        tet2[0] = pyr[0];
        tet2[1] = pyr[2];
        tet2[2] = pyr[3];
    } else {
        tet1[0] = pyr[1];
        tet1[1] = pyr[2];
        tet1[2] = pyr[3];
        tet1[3] = pyr[4];
        tet2[0] = pyr[1];
        tet2[1] = pyr[3];
        tet2[2] = pyr[0];
    }
    tet2[3] = pyr[4];
    [tet1, tet2]
}

/// Convert a prism into 3 tetrahedra
#[must_use]
pub fn pri2tets(pri: &[Idx]) -> [Tetrahedron; 3] {
    let imin = argmin(pri);

    let mut idx = [0; 6];
    for i in 0..6 {
        idx[i] = pri[INDIRECTION_PRI[imin][i]];
    }

    let mut tet1 = Tetrahedron::default();
    let mut tet2 = Tetrahedron::default();
    let mut tet3 = Tetrahedron::default();
    tet1[0] = idx[0];
    tet1[1] = idx[1];
    tet1[2] = idx[2];
    if compare_edges(idx[1], idx[5], idx[2], idx[4]) {
        tet1[3] = idx[5];
        tet2[0] = idx[0];
        tet2[1] = idx[1];
        tet2[2] = idx[5];
        tet2[3] = idx[4];
    } else {
        tet1[3] = idx[4];
        tet2[0] = idx[0];
        tet2[1] = idx[4];
        tet2[2] = idx[2];
        tet2[3] = idx[5];
    }
    tet3[0] = idx[0];
    tet3[1] = idx[4];
    tet3[2] = idx[5];
    tet3[3] = idx[3];
    [tet1, tet2, tet3]
}

/// Convert a hex into 5 or 6 tetrahedra
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn hex2tets(hex: &[Idx]) -> ([Tetrahedron; 5], Option<Tetrahedron>) {
    let imin = argmin(hex);

    let mut idx = [0; 8];
    for i in 0..8 {
        idx[i] = hex[INDIRECTION_HEX[imin][i]];
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

    let mut tet1 = Tetrahedron::default();
    let mut tet2 = Tetrahedron::default();
    let mut tet3 = Tetrahedron::default();
    let mut tet4 = Tetrahedron::default();
    let mut tet5 = Tetrahedron::default();
    let mut tet6 = Tetrahedron::default();

    if flg2 == 0 {
        tet1[0] = idx2[0];
        tet1[1] = idx2[1];
        tet1[2] = idx2[2];
        tet1[3] = idx2[5];
        tet2[0] = idx2[0];
        tet2[1] = idx2[2];
        tet2[2] = idx2[7];
        tet2[3] = idx2[5];
        tet3[0] = idx2[0];
        tet3[1] = idx2[2];
        tet3[2] = idx2[3];
        tet3[3] = idx2[7];
        tet4[0] = idx2[0];
        tet4[1] = idx2[5];
        tet4[2] = idx2[7];
        tet4[3] = idx2[4];
        tet5[0] = idx2[2];
        tet5[1] = idx2[7];
        tet5[2] = idx2[5];
        tet5[3] = idx2[6];
        return ([tet1, tet2, tet3, tet4, tet5], None);
    } else if flg2 == 1 {
        tet1[0] = idx2[0];
        tet1[1] = idx2[5];
        tet1[2] = idx2[7];
        tet1[3] = idx2[4];
        tet2[0] = idx2[0];
        tet2[1] = idx2[1];
        tet2[2] = idx2[7];
        tet2[3] = idx2[5];
        tet3[0] = idx2[1];
        tet3[1] = idx2[6];
        tet3[2] = idx2[7];
        tet3[3] = idx2[5];
        tet4[0] = idx2[0];
        tet4[1] = idx2[7];
        tet4[2] = idx2[2];
        tet4[3] = idx2[3];
        tet5[0] = idx2[0];
        tet5[1] = idx2[7];
        tet5[2] = idx2[1];
        tet5[3] = idx2[2];
        tet6[0] = idx2[1];
        tet6[1] = idx2[7];
        tet6[2] = idx2[6];
        tet6[3] = idx2[2];
    } else if flg2 == 2 {
        tet1[0] = idx2[0];
        tet1[1] = idx2[4];
        tet1[2] = idx2[5];
        tet1[3] = idx2[6];
        tet2[0] = idx2[0];
        tet2[1] = idx2[3];
        tet2[2] = idx2[7];
        tet2[3] = idx2[6];
        tet3[0] = idx2[0];
        tet3[1] = idx2[7];
        tet3[2] = idx2[4];
        tet3[3] = idx2[6];
        tet4[0] = idx2[0];
        tet4[1] = idx2[1];
        tet4[2] = idx2[2];
        tet4[3] = idx2[5];
        tet5[0] = idx2[0];
        tet5[1] = idx2[3];
        tet5[2] = idx2[6];
        tet5[3] = idx2[2];
        tet6[0] = idx2[0];
        tet6[1] = idx2[6];
        tet6[2] = idx2[5];
        tet6[3] = idx2[2];
    } else if flg2 == 3 {
        tet1[0] = idx2[0];
        tet1[1] = idx2[2];
        tet1[2] = idx2[3];
        tet1[3] = idx2[6];
        tet2[0] = idx2[0];
        tet2[1] = idx2[3];
        tet2[2] = idx2[7];
        tet2[3] = idx2[6];
        tet3[0] = idx2[0];
        tet3[1] = idx2[7];
        tet3[2] = idx2[4];
        tet3[3] = idx2[6];
        tet4[0] = idx2[0];
        tet4[1] = idx2[5];
        tet4[2] = idx2[6];
        tet4[3] = idx2[4];
        tet5[0] = idx2[1];
        tet5[1] = idx2[5];
        tet5[2] = idx2[6];
        tet5[3] = idx2[0];
        tet6[0] = idx2[1];
        tet6[1] = idx2[6];
        tet6[2] = idx2[2];
        tet6[3] = idx2[0];
    }
    ([tet1, tet2, tet3, tet4, tet5], Some(tet6))
}
