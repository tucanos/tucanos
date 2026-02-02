use core::iter::once;

use crate::mesh::{
    Edge, Idx, Mesh, QuadraticEdge, QuadraticTetrahedron, QuadraticTriangle, Simplex, Tetrahedron,
    Triangle, Vertex,
};
use rustc_hash::FxHashMap;

fn get_midpoint(
    edges: &FxHashMap<Edge<impl Idx>, usize>,
    offset: usize,
    t: impl Simplex,
    i0: usize,
    i1: usize,
) -> usize {
    *edges
        .get(&Edge::new(t.get(i0), t.get(i1)).sorted())
        .unwrap()
        + offset
}

#[must_use]
pub fn to_quadratic_edge_mesh<const D: usize, T: Idx, M: Mesh<D, C = QuadraticEdge<T>>, T2: Idx>(
    msh: &impl Mesh<D, C = Edge<T2>>,
) -> M {
    let edges = msh.edges();

    let mut res = M::empty();
    res.add_verts(msh.verts());
    let mut new_verts = vec![Vertex::zeros(); edges.len()];
    for (&e, &i) in &edges {
        new_verts[i] = 0.5 * (msh.vert(e.get(0)) + msh.vert(e.get(1)));
    }
    res.add_verts(new_verts.iter().copied());

    for (e, t) in msh.elems().zip(msh.etags()) {
        res.add_elems(
            once(QuadraticEdge::new(
                e.get(0),
                e.get(1),
                get_midpoint(&edges, msh.n_verts(), e, 0, 1),
            )),
            once(t),
        );
    }
    res
}

#[must_use]
pub fn to_quadratic_triangle_mesh<
    const D: usize,
    T: Idx,
    M: Mesh<D, C = QuadraticTriangle<T>>,
    T2: Idx,
>(
    msh: &impl Mesh<D, C = Triangle<T2>>,
) -> M {
    let edges = msh.edges();

    let mut res = M::empty();
    res.add_verts(msh.verts());
    let mut new_verts = vec![Vertex::zeros(); edges.len()];
    for (&e, &i) in &edges {
        new_verts[i] = 0.5 * (msh.vert(e.get(0)) + msh.vert(e.get(1)));
    }
    res.add_verts(new_verts.iter().copied());

    for (t, tag) in msh.elems().zip(msh.etags()) {
        res.add_elems(
            once(QuadraticTriangle::new(
                t.get(0),
                t.get(1),
                t.get(2),
                edges.get(&t.face(2).sorted()).unwrap() + msh.n_verts(),
                edges.get(&t.face(0).sorted()).unwrap() + msh.n_verts(),
                edges.get(&t.face(1).sorted()).unwrap() + msh.n_verts(),
            )),
            once(tag),
        );
    }

    for (e, tag) in msh.faces().zip(msh.ftags()) {
        res.add_faces(
            once(QuadraticEdge::new(
                e.get(0),
                e.get(1),
                edges.get(&e.sorted()).unwrap() + msh.n_verts(),
            )),
            once(tag),
        );
    }

    res
}

#[must_use]
pub fn to_quadratic_tetrahedron_mesh<T: Idx, M: Mesh<3, C = QuadraticTetrahedron<T>>, T2: Idx>(
    msh: &impl Mesh<3, C = Tetrahedron<T2>>,
) -> M {
    let edges = msh.edges();

    let mut res = M::empty();
    res.add_verts(msh.verts());

    let mut new_verts = vec![Vertex::zeros(); edges.len()];
    for (&e, &i) in &edges {
        new_verts[i] = 0.5 * (msh.vert(e.get(0)) + msh.vert(e.get(1)));
    }
    res.add_verts(new_verts.iter().copied());

    for (t, tag) in msh.elems().zip(msh.etags()) {
        res.add_elems(
            once(QuadraticTetrahedron::new(
                t.get(0),
                t.get(1),
                t.get(2),
                t.get(3),
                get_midpoint(&edges, msh.n_verts(), t, 0, 1),
                get_midpoint(&edges, msh.n_verts(), t, 1, 2),
                get_midpoint(&edges, msh.n_verts(), t, 2, 0),
                get_midpoint(&edges, msh.n_verts(), t, 0, 3),
                get_midpoint(&edges, msh.n_verts(), t, 1, 3),
                get_midpoint(&edges, msh.n_verts(), t, 2, 3),
            )),
            once(tag),
        );
    }

    for (t, tag) in msh.faces().zip(msh.ftags()) {
        res.add_faces(
            once(QuadraticTriangle::new(
                t.get(0),
                t.get(1),
                t.get(2),
                get_midpoint(&edges, msh.n_verts(), t, 0, 1),
                get_midpoint(&edges, msh.n_verts(), t, 1, 2),
                get_midpoint(&edges, msh.n_verts(), t, 2, 0),
            )),
            once(tag),
        );
    }

    res
}
