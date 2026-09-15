use rustc_hash::{FxBuildHasher, FxHashMap};

use super::{Edge, Mesh, Prism, Quadrangle, Simplex, pri2tets, qua2tris};
use crate::{Tag, Vertex};

pub type SplitEdgeData<T> = FxHashMap<Edge<T>, T>;
type SplitElemData<C> = (Vec<C>, Vec<Tag>, Vec<<C as Simplex>::FACE>, Vec<Tag>);

/// Rotates a triangle so that the vertex on the single side of the isosurface comes first.
#[must_use]
fn rotate_triangle_single_side(mut tri: [usize; 3], f: &[f64]) -> ([usize; 3], bool) {
    let n_pos = tri.iter().filter(|&&i| f[i] > 0.0).count();
    let n_neg = tri.iter().filter(|&&i| f[i] < 0.0).count();
    assert!(n_pos == 1 || n_neg == 1);

    let single_is_pos = n_pos == 1;
    let i = tri
        .iter()
        .position(|&v| (f[v] > 0.0) == single_is_pos)
        .unwrap();
    tri.rotate_left(i);
    (tri, single_is_pos)
}

/// Splits a triangle along the isosurface, returning the rotated triangle, the indices of the new vertices on the split edges, and a flag indicating if the single vertex is positive.
#[must_use]
fn split_triangle_data<C: Simplex>(
    tri: [usize; 3],
    f: &[f64],
    split_edgs: &FxHashMap<Edge<C::T>, C::T>,
) -> ([usize; 3], usize, usize, bool) {
    let (tri, single_is_pos) = rotate_triangle_single_side(tri, f);
    let i0 = *split_edgs.get(&Edge::new(tri[0], tri[1]).sorted()).unwrap();
    let i1 = *split_edgs.get(&Edge::new(tri[0], tri[2]).sorted()).unwrap();
    (
        tri,
        i0.try_into().unwrap(),
        i1.try_into().unwrap(),
        single_is_pos,
    )
}

#[must_use]
const fn split_signed_pair_tags(tag: Tag, first_is_pos: bool) -> (Tag, Tag) {
    if first_is_pos {
        (tag, -tag)
    } else {
        (-tag, tag)
    }
}

/// Splits the edges of the mesh along the isosurface, returning the new vertices and a map from edges to the indices of the new vertices.
#[must_use]
fn split_isosurface_edges<const D: usize, M: Mesh<D>>(
    mesh: &M,
    f: &[f64],
) -> (Vec<Vertex<D>>, SplitEdgeData<<M::C as Simplex>::T>) {
    let mut split_edgs = FxHashMap::with_hasher(FxBuildHasher);
    let mut idx = mesh.n_verts();
    let mut new_verts = mesh.verts().collect::<Vec<_>>();

    for &edg in mesh.edges().keys() {
        let [i0, i1] = edg.into();
        if f[i0] == 0.0 && f[i1] == 0.0 {
            todo!();
        }
        if f[i0] * f[i1] <= 0.0 {
            let t = -f[i0] / (f[i1] - f[i0]);
            assert!((0.0..=1.0).contains(&t));
            let p = (1.0 - t) * mesh.vert(i0) + t * mesh.vert(i1);
            new_verts.push(p);
            split_edgs.insert(edg, idx.try_into().unwrap());
            idx += 1;
        }
    }

    (new_verts, split_edgs)
}

/// Splits a tetrahedron along the isosurface, adding the resulting new tetrahedra and their tags to the provided vectors.
fn split_isosurface_tet<C: Simplex>(
    elem: &C,
    f: &[f64],
    split_edgs: &SplitEdgeData<C::T>,
    new_elems: &mut Vec<C>,
    new_etags: &mut Vec<Tag>,
    new_faces: &mut Vec<C::FACE>,
    new_ftags: &mut Vec<Tag>,
) {
    let n_pos = elem.into_iter().filter(|&i| f[i] > 0.0).count();

    let split = |a: usize, b: usize| -> usize {
        let idx = *split_edgs.get(&Edge::new(a, b).sorted()).unwrap();
        idx.try_into().unwrap()
    };

    if n_pos == 1 || n_pos == 3 {
        let sgn = if n_pos == 1 { 1.0 } else { -1.0 };
        // permute the element indices so the single-side vertex comes first
        let i = elem.into_iter().position(|v| sgn * f[v] > 0.0).unwrap();
        let v0 = elem.get(i);
        let face = elem.face(i);
        let v1 = face.get(0);
        let v2 = face.get(1);
        let v3 = face.get(2);
        debug_assert!(elem.is_same(&C::from_iter([v0, v1, v2, v3])));

        debug_assert!(sgn * f[v0] > 0.0);
        debug_assert!(sgn * f[v1] < 0.0);
        debug_assert!(sgn * f[v2] < 0.0);
        debug_assert!(sgn * f[v3] < 0.0);

        let e01 = split(v0, v1);
        let e02 = split(v0, v2);
        let e03 = split(v0, v3);

        // The 1/3 split yields one tetrahedron around the single-side vertex
        // and one triangular prism on the opposite side.
        let etag = if sgn > 0.0 { 1 } else { -1 };
        new_elems.push(C::from_iter([v0, e01, e02, e03]));
        new_etags.push(etag);

        let etag = if sgn > 0.0 { -1 } else { 1 };
        let prism = Prism::<C::T>::new(v1, v3, v2, e01, e03, e02);
        for tet in pri2tets(&prism) {
            new_elems.push(C::from_iter(tet));
            new_etags.push(etag);
        }

        let iface = if sgn > 0.0 {
            C::FACE::from_iter([e01, e02, e03])
        } else {
            C::FACE::from_iter([e01, e03, e02])
        };
        new_faces.push(iface);
        new_ftags.push(Tag::MAX);
        return;
    }

    if n_pos == 2 {
        // permute the element indices so the two positive vertices come first
        let mut i0 = usize::MAX;
        let mut i1 = usize::MAX;
        for (i, j) in elem.into_iter().enumerate() {
            if f[j] > 0.0 {
                if i0 == usize::MAX {
                    i0 = i; // local index
                } else {
                    i1 = j; // global index
                }
            }
        }
        debug_assert!(i0 != usize::MAX && i1 != usize::MAX);

        let v0 = elem.get(i0);
        let face = elem.face(i0);
        let i = face.into_iter().position(|v| v == i1).unwrap();
        let v1 = face.get(i);
        let (v2, v3) = match i {
            0 => (face.get(1), face.get(2)),
            1 => (face.get(2), face.get(0)),
            2 => (face.get(0), face.get(1)),
            _ => unreachable!(),
        };
        debug_assert!(elem.is_same(&C::from_iter([v0, v1, v2, v3])));

        debug_assert!(f[v0] > 0.0);
        debug_assert!(f[v1] > 0.0);
        debug_assert!(f[v2] < 0.0);
        debug_assert!(f[v3] < 0.0);

        let e02 = split(v0, v2);
        let e03 = split(v0, v3);
        let e12 = split(v1, v2);
        let e13 = split(v1, v3);

        // 2+2 split produces one prism per side of the interface. Both prisms
        // are split with pri2tets so diagonal choices follow global indices.
        let pos_prism = Prism::<C::T>::new(v0, e02, e03, v1, e12, e13);
        for tet in pri2tets(&pos_prism) {
            new_elems.push(C::from_iter(tet));
            new_etags.push(1);
        }

        let neg_prism = Prism::<C::T>::new(v2, e02, e12, v3, e03, e13);
        for tet in pri2tets(&neg_prism) {
            new_elems.push(C::from_iter(tet));
            new_etags.push(-1);
        }

        let quad = Quadrangle::<usize>::new(e02, e03, e13, e12);
        for iface in qua2tris(&quad) {
            new_faces.push(C::FACE::from_iter(iface));
            new_ftags.push(Tag::MAX);
        }
        return;
    }

    unreachable!(
        "Exact zeros encountered. Ensure f is perturbed slightly (e.g., v = 1e-12 if v == 0.0) before the element loop."
    );
}

/// Splits all elements of the mesh along the isosurface, returning the new elements, their tags, and the new faces and their tags.
#[must_use]
fn split_isosurface_elems<const D: usize, M: Mesh<D>>(
    mesh: &M,
    f: &[f64],
    split_edgs: &SplitEdgeData<<M::C as Simplex>::T>,
) -> SplitElemData<M::C> {
    let mut new_elems = Vec::with_capacity(mesh.n_elems());
    let mut new_etags = Vec::with_capacity(mesh.n_elems());
    let mut new_faces = Vec::new();
    let mut new_ftags = Vec::new();

    for elem in mesh.elems() {
        if elem.into_iter().all(|i| f[i] > 0.0) {
            new_elems.push(elem);
            new_etags.push(1);
            continue;
        }
        if elem.into_iter().all(|i| f[i] < 0.0) {
            new_elems.push(elem);
            new_etags.push(-1);
            continue;
        }

        match M::C::N_VERTS {
            2 => {
                let i0 = elem.get(0);
                let i1 = elem.get(1);
                let i = *split_edgs.get(&Edge::new(i0, i1).sorted()).unwrap();
                let i = i.try_into().unwrap();
                new_elems.push(M::C::from_iter([i0, i]));
                new_elems.push(M::C::from_iter([i, i1]));
                let (t0, t1) = split_signed_pair_tags(1, f[i0] > 0.0);
                new_etags.push(t0);
                new_etags.push(t1);
            }
            3 => {
                let tri = [elem.get(0), elem.get(1), elem.get(2)];
                let (tri, i0, i1, single_is_pos) = split_triangle_data::<M::C>(tri, f, split_edgs);

                // Decompose into one triangle around the single-side vertex
                // and one quad on the opposite side, then split the quad using
                // the global-index rule from qua2tris.
                new_elems.push(M::C::from_iter([tri[0], i0, i1]));
                let quad = Quadrangle::<usize>::new(tri[1], tri[2], i1, i0);
                for qtri in qua2tris(&quad) {
                    new_elems.push(M::C::from_iter(qtri));
                }
                if single_is_pos {
                    new_etags.push(1);
                    new_etags.push(-1);
                    new_etags.push(-1);
                    new_faces.push(<M::C as Simplex>::FACE::from_iter([i0, i1]));
                } else {
                    new_etags.push(-1);
                    new_etags.push(1);
                    new_etags.push(1);
                    new_faces.push(<M::C as Simplex>::FACE::from_iter([i1, i0]));
                }
                new_ftags.push(Tag::MAX);
            }
            4 => {
                split_isosurface_tet::<M::C>(
                    &elem,
                    f,
                    split_edgs,
                    &mut new_elems,
                    &mut new_etags,
                    &mut new_faces,
                    &mut new_ftags,
                );
            }
            _ => unimplemented!(),
        }
    }

    (new_elems, new_etags, new_faces, new_ftags)
}

/// Splits all faces of the mesh along the isosurface, returning the new faces and their tags.
#[must_use]
fn split_isosurface_faces<const D: usize, M: Mesh<D>>(
    mesh: &M,
    f: &[f64],
    split_edgs: &SplitEdgeData<<M::C as Simplex>::T>,
) -> (Vec<<M::C as Simplex>::FACE>, Vec<Tag>) {
    let mut new_faces = Vec::with_capacity(mesh.n_faces());
    let mut new_ftags = Vec::with_capacity(mesh.n_faces());

    for (face, tag) in mesh.faces().zip(mesh.ftags()) {
        if face.into_iter().all(|i| f[i] > 0.0) {
            new_faces.push(face);
            new_ftags.push(tag);
            continue;
        }
        if face.into_iter().all(|i| f[i] < 0.0) {
            new_faces.push(face);
            new_ftags.push(-tag);
            continue;
        }

        match <M::C as Simplex>::FACE::N_VERTS {
            2 => {
                let i0 = face.get(0);
                let i1 = face.get(1);
                let i = *split_edgs.get(&Edge::new(i0, i1).sorted()).unwrap();
                let i = i.try_into().unwrap();
                new_faces.push(<M::C as Simplex>::FACE::from_iter([i0, i]));
                new_faces.push(<M::C as Simplex>::FACE::from_iter([i, i1]));
                let (t0, t1) = split_signed_pair_tags(tag, f[i0] > 0.0);
                new_ftags.push(t0);
                new_ftags.push(t1);
            }
            3 => {
                let tri = [face.get(0), face.get(1), face.get(2)];
                let (tri, i0, i1, single_is_pos) = split_triangle_data::<M::C>(tri, f, split_edgs);

                let f0 = <M::C as Simplex>::FACE::from_iter([tri[0], i0, i1]);
                let quad = Quadrangle::<usize>::new(tri[1], tri[2], i1, i0);
                let [q0, q1] = qua2tris(&quad);
                let fq0 = <M::C as Simplex>::FACE::from_iter(q0);
                let fq1 = <M::C as Simplex>::FACE::from_iter(q1);

                new_faces.push(f0);
                new_faces.push(fq0);
                new_faces.push(fq1);
                if single_is_pos {
                    new_ftags.push(tag);
                    new_ftags.push(-tag);
                    new_ftags.push(-tag);
                } else {
                    new_ftags.push(-tag);
                    new_ftags.push(tag);
                    new_ftags.push(tag);
                }
            }
            _ => unimplemented!(),
        }
    }

    (new_faces, new_ftags)
}

/// Splits a mesh along the 0.0 isosurface, returning a new mesh with the split elements and faces.
pub(super) fn split_isosurface<const D: usize, M: Mesh<D>, M2: Mesh<D, C = M::C>>(
    mesh: &M,
    f: &[f64],
) -> (M2, SplitEdgeData<<M::C as Simplex>::T>) {
    assert!(mesh.etags().all(|t| t == 1));
    assert!(mesh.ftags().all(|t| t > 0));
    assert_eq!(M::C::order(), 1);

    // Perturb exact zeros to avoid complex topological singularities.
    let f: Vec<f64> = f
        .iter()
        .map(|&v| if v == 0.0 { 1e-12 } else { v })
        .collect();

    let (new_verts, split_edgs) = split_isosurface_edges(mesh, &f);
    let (new_elems, new_etags, mut new_faces, mut new_ftags) =
        split_isosurface_elems(mesh, &f, &split_edgs);

    let (extra_faces, extra_ftags) = split_isosurface_faces(mesh, &f, &split_edgs);
    new_faces.extend(extra_faces);
    new_ftags.extend(extra_ftags);

    // Explicit split routines now generate required boundary/interface faces.

    (
        M2::new(&new_verts, &new_elems, &new_etags, &new_faces, &new_ftags),
        split_edgs,
    )
}
