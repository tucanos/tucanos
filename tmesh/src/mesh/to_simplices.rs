use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};

use crate::{Tag, graph::CSRGraph};

use super::{Hexahedron, Prism, Pyramid, Quadrangle, Tetrahedron, Triangle};

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

fn argmin(arr: &[usize]) -> usize {
    let mut imin = 0;
    let mut usize = arr[0];

    for (i, j) in arr.iter().copied().enumerate().skip(1) {
        if j < usize {
            usize = j;
            imin = i;
        }
    }

    imin
}

/// Convert a quadrangle into 2 triangles
#[must_use]
pub fn qua2tris(quad: &Quadrangle) -> [Triangle; 2] {
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
pub fn pyr2tets(pyr: &Pyramid) -> [Tetrahedron; 2] {
    let mut tet1 = Tetrahedron::default();
    let mut tet2 = Tetrahedron::default();

    if compare_edges(pyr[0], pyr[2], pyr[1], pyr[3]) {
        tet1[0] = pyr[0];
        tet1[1] = pyr[1];
        tet1[2] = pyr[2];
        tet2[0] = pyr[0];
        tet2[1] = pyr[2];
        tet2[2] = pyr[3];
    } else {
        tet1[0] = pyr[1];
        tet1[1] = pyr[2];
        tet1[2] = pyr[3];
        tet2[0] = pyr[1];
        tet2[1] = pyr[3];
        tet2[2] = pyr[0];
    }
    tet1[3] = pyr[4];
    tet2[3] = pyr[4];
    [tet1, tet2]
}

/// Convert a prism into 3 tetrahedra
#[must_use]
pub fn pri2tets(pri: &Prism) -> [Tetrahedron; 3] {
    let imin = argmin(pri);

    let mut usize = [0; 6];
    for i in 0..6 {
        usize[i] = pri[INDIRECTION_PRI[imin][i]];
    }

    let mut tet1 = Tetrahedron::default();
    let mut tet2 = Tetrahedron::default();
    let mut tet3 = Tetrahedron::default();
    tet1[0] = usize[0];
    tet1[1] = usize[1];
    tet1[2] = usize[2];
    if compare_edges(usize[1], usize[5], usize[2], usize[4]) {
        tet1[3] = usize[5];
        tet2[0] = usize[0];
        tet2[1] = usize[1];
        tet2[2] = usize[5];
        tet2[3] = usize[4];
    } else {
        tet1[3] = usize[4];
        tet2[0] = usize[0];
        tet2[1] = usize[4];
        tet2[2] = usize[2];
        tet2[3] = usize[5];
    }
    tet3[0] = usize[0];
    tet3[1] = usize[4];
    tet3[2] = usize[5];
    tet3[3] = usize[3];
    [tet1, tet2, tet3]
}

/// Convert a hex into 5 or 6 tetrahedra
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn hex2tets(hex: &Hexahedron) -> ([Tetrahedron; 5], Option<Tetrahedron>) {
    let imin = argmin(hex);

    let mut usize = [0; 8];
    for i in 0..8 {
        usize[i] = hex[INDIRECTION_HEX[imin][i]];
    }

    let i1 = u32::from(compare_edges(usize[1], usize[6], usize[2], usize[5]));
    let i2 = u32::from(compare_edges(usize[3], usize[6], usize[2], usize[7]));
    let i3 = u32::from(compare_edges(usize[4], usize[6], usize[5], usize[7]));
    let flg = i1 + 2 * i2 + 4 * i3;
    let rot = ROTATION_HEX[flg as usize];

    let mut usize2 = usize;
    if rot == 120 {
        for i in 0..8 {
            usize2[i] = usize[PERM_120[i]];
        }
    } else if rot == 240 {
        for i in 0..8 {
            usize2[i] = usize[PERM_240[i]];
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
        tet1[0] = usize2[0];
        tet1[1] = usize2[1];
        tet1[2] = usize2[2];
        tet1[3] = usize2[5];
        tet2[0] = usize2[0];
        tet2[1] = usize2[2];
        tet2[2] = usize2[7];
        tet2[3] = usize2[5];
        tet3[0] = usize2[0];
        tet3[1] = usize2[2];
        tet3[2] = usize2[3];
        tet3[3] = usize2[7];
        tet4[0] = usize2[0];
        tet4[1] = usize2[5];
        tet4[2] = usize2[7];
        tet4[3] = usize2[4];
        tet5[0] = usize2[2];
        tet5[1] = usize2[7];
        tet5[2] = usize2[5];
        tet5[3] = usize2[6];
        return ([tet1, tet2, tet3, tet4, tet5], None);
    } else if flg2 == 1 {
        tet1[0] = usize2[0];
        tet1[1] = usize2[5];
        tet1[2] = usize2[7];
        tet1[3] = usize2[4];
        tet2[0] = usize2[0];
        tet2[1] = usize2[1];
        tet2[2] = usize2[7];
        tet2[3] = usize2[5];
        tet3[0] = usize2[1];
        tet3[1] = usize2[6];
        tet3[2] = usize2[7];
        tet3[3] = usize2[5];
        tet4[0] = usize2[0];
        tet4[1] = usize2[7];
        tet4[2] = usize2[2];
        tet4[3] = usize2[3];
        tet5[0] = usize2[0];
        tet5[1] = usize2[7];
        tet5[2] = usize2[1];
        tet5[3] = usize2[2];
        tet6[0] = usize2[1];
        tet6[1] = usize2[7];
        tet6[2] = usize2[6];
        tet6[3] = usize2[2];
    } else if flg2 == 2 {
        tet1[0] = usize2[0];
        tet1[1] = usize2[4];
        tet1[2] = usize2[5];
        tet1[3] = usize2[6];
        tet2[0] = usize2[0];
        tet2[1] = usize2[3];
        tet2[2] = usize2[7];
        tet2[3] = usize2[6];
        tet3[0] = usize2[0];
        tet3[1] = usize2[7];
        tet3[2] = usize2[4];
        tet3[3] = usize2[6];
        tet4[0] = usize2[0];
        tet4[1] = usize2[1];
        tet4[2] = usize2[2];
        tet4[3] = usize2[5];
        tet5[0] = usize2[0];
        tet5[1] = usize2[3];
        tet5[2] = usize2[6];
        tet5[3] = usize2[2];
        tet6[0] = usize2[0];
        tet6[1] = usize2[6];
        tet6[2] = usize2[5];
        tet6[3] = usize2[2];
    } else if flg2 == 3 {
        tet1[0] = usize2[0];
        tet1[1] = usize2[2];
        tet1[2] = usize2[3];
        tet1[3] = usize2[6];
        tet2[0] = usize2[0];
        tet2[1] = usize2[3];
        tet2[2] = usize2[7];
        tet2[3] = usize2[6];
        tet3[0] = usize2[0];
        tet3[1] = usize2[7];
        tet3[2] = usize2[4];
        tet3[3] = usize2[6];
        tet4[0] = usize2[0];
        tet4[1] = usize2[5];
        tet4[2] = usize2[6];
        tet4[3] = usize2[4];
        tet5[0] = usize2[1];
        tet5[1] = usize2[5];
        tet5[2] = usize2[6];
        tet5[3] = usize2[0];
        tet6[0] = usize2[1];
        tet6[1] = usize2[6];
        tet6[2] = usize2[2];
        tet6[3] = usize2[0];
    }
    ([tet1, tet2, tet3, tet4, tet5], Some(tet6))
}

#[allow(dead_code)]
#[must_use]
#[allow(clippy::too_many_lines)]
#[allow(clippy::type_complexity)]
pub fn simplify_hexas(
    hexs: &[Hexahedron],
    hex_tags: &[Tag],
) -> (
    Vec<Hexahedron>,
    Vec<Tag>,
    Vec<usize>,
    Vec<Prism>,
    Vec<Tag>,
    Vec<usize>,
    Vec<usize>,
) {
    let hex2quads = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ];
    let mut quads = FxHashMap::with_hasher(FxBuildHasher);

    for (i_hex, &hex) in hexs.iter().enumerate() {
        for &f in &hex2quads {
            let mut f = f;
            for i in &mut f {
                *i = hex[*i];
            }
            f.sort_unstable();
            match quads.entry(f) {
                std::collections::hash_map::Entry::Occupied(occupied_entry) => {
                    let [i0, i1] = occupied_entry.into_mut();
                    assert_eq!(*i1, usize::MAX);
                    assert_ne!(*i0, i_hex, "{hex:?} {f:?}");
                    *i1 = i_hex;
                }
                std::collections::hash_map::Entry::Vacant(vacant_entry) => {
                    vacant_entry.insert([i_hex, usize::MAX]);
                }
            }
        }
        let tmp = hex.iter().copied().collect::<FxHashSet<_>>();
        assert_eq!(tmp.len(), 8);
    }

    let edgs = quads
        .values()
        .copied()
        .filter(|&[_, i1]| i1 != usize::MAX)
        .collect::<Vec<_>>();
    let hex2hex = CSRGraph::from_edges(edgs.iter().copied(), Some(hexs.len()));

    let mut out_hexs = Vec::with_capacity(hexs.len());
    let mut out_hex_tags = Vec::with_capacity(hexs.len());
    let mut out_hex_ids = Vec::with_capacity(hexs.len());

    let mut out_pris = Vec::new();
    let mut out_pri_tags = Vec::new();
    let mut out_pri_ids = Vec::new();

    let mut unused_verts = Vec::new();
    for (i_hex, (hex, tag)) in hexs.iter().zip(hex_tags.iter()).enumerate() {
        let neighbors = hex2hex.row(i_hex);
        let n = neighbors.len();
        let mut count = 0;
        for i in 0..n - 1 {
            if neighbors[i] == neighbors[i + 1] {
                let other = hexs[neighbors[i]];
                let mut tmp = [0; 8];
                for &f in &hex2quads {
                    let mut f2 = f;
                    for i in &mut f2 {
                        *i = hex[*i];
                    }
                    for (i, j) in f.iter().zip(f2.iter()) {
                        if other.contains(j) {
                            tmp[*i] += 1;
                        }
                    }
                }
                let mut i0 = usize::MAX;
                let mut i1 = usize::MAX;
                for (i, &c) in tmp.iter().enumerate() {
                    if c == 0 {
                        if i0 == usize::MAX {
                            i0 = i;
                        } else {
                            assert_eq!(i1, usize::MAX);
                            i1 = i;
                        }
                    } else {
                        assert_eq!(c, 3);
                    }
                }
                if i0 == 0 && i1 == 4 {
                    out_pris.push([hex[0], hex[1], hex[3], hex[4], hex[5], hex[7]]);
                    unused_verts.push(hex[2]);
                    unused_verts.push(hex[6]);
                } else if i0 == 1 && i1 == 5 {
                    out_pris.push([hex[0], hex[1], hex[2], hex[4], hex[5], hex[6]]);
                    unused_verts.push(hex[3]);
                    unused_verts.push(hex[7]);
                }
                out_pri_tags.push(*tag);
                out_pri_ids.push(i_hex);
                count += 1;
            }
        }
        if count == 0 {
            out_hexs.push(*hex);
            out_hex_tags.push(*tag);
            out_hex_ids.push(i_hex);
        }

        assert!(count < 2);
    }
    (
        out_hexs,
        out_hex_tags,
        out_hex_ids,
        out_pris,
        out_pri_tags,
        out_pri_ids,
        unused_verts,
    )
}
