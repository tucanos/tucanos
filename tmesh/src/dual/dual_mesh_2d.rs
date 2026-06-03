//! Computation of the 2D dual mesh for triangular primal meshes (`Mesh<2>` with
//! `C = Triangle<T>`).
//!
//! The constructor [`DualMesh2d::new`] builds a polygonal dual control-volume
//! around each primal vertex with the following algorithm:
//!
//! 1. Enumerate all primal edges and boundary vertices.
//! 2. Create dual vertices from:
//!    - primal boundary vertices,
//!    - primal edge midpoints,
//!    - one center per primal triangle (depends on [`DualType`]):
//!      - `Median`: triangle centroid,
//!      - `Barth` / `ThresholdBarth`: circumcenter when admissible, otherwise
//!        fallback to an edge-centered location to keep robust cells.
//! 3. Build dual internal faces by connecting each primal edge midpoint to the
//!    center of each adjacent primal element.
//! 4. Build dual boundary faces by connecting a boundary primal vertex to its
//!    boundary-edge midpoint(s).
//! 5. Accumulate oriented dual face incidences per primal vertex to define
//!    polygonal dual elements (`elem_to_face`).
//! 6. Compute edge-based flux normals by summing oriented contributions from the
//!    dual segments attached to each primal edge.
//! 7. Remove degenerate/empty dual faces and filter zero-length edge-normal
//!    contributions.
//!
//! TODO (possible improvements):
//! - Add focused tests for each `DualType` branch (`Median`, `Barth`,
//!   `ThresholdBarth`) on pathological obtuse meshes.
//! - Add a geometric validation pass for dual boundary orientation and normal
//!   consistency.
//! - Reduce temporary allocations in `new` (especially face incidence buffers)
//!   to improve construction performance on large meshes.
//! - Expose optional diagnostics (number of degenerate faces removed, fallback
//!   center usage count) for debugging and quality monitoring.
use super::{DualCellCenter, DualMesh, DualType, PolyMesh, PolyMeshType};
use crate::{
    Tag, Vert2d,
    mesh::{Edge, GEdge, GSimplex, GTriangle, Idx, Mesh, Simplex, Triangle, sort_elem_min_ids},
};
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use rustc_hash::FxHashMap;

/// Dual of a 2D triangular primal mesh.
///
/// The dual mesh stores polygonal control volumes centered on primal vertices,
/// plus edge-based normals commonly used by finite-volume discretizations.
pub struct DualMesh2d<T: Idx> {
    verts: Vec<Vert2d>,
    faces: Vec<Edge<T>>,
    ftags: Vec<Tag>,
    elem_to_face_ptr: Vec<usize>,
    elem_to_face: Vec<(usize, bool)>,
    etags: Vec<Tag>,
    edges: Vec<Edge<T>>,
    edge_normals: Vec<Vert2d>,
    bdy_faces: Vec<(usize, Tag, Vert2d)>,
}

struct FaceBuildState2d<T: Idx> {
    faces: Vec<Edge<T>>,
    ftags: Vec<Tag>,
    poly_to_face: Vec<(usize, bool)>,
    edge_normals: Vec<Vert2d>,
    bdy_faces: Vec<(usize, Tag, Vert2d)>,
    n_empty_faces: usize,
}

struct VertexBuild2d {
    verts: Vec<Vert2d>,
    bdy_verts: FxHashMap<usize, usize>,
    vert_idx_elem: Vec<usize>,
    n_bdy_verts: usize,
}

impl<T: Idx> DualMesh2d<T> {
    fn get_tri_center(v: &GTriangle<2>, t: DualType) -> DualCellCenter<2, Triangle<T>> {
        match t {
            DualType::Median => DualCellCenter::Vertex(v.center()),
            DualType::Barth | DualType::ThresholdBarth(_) => {
                let f = match t {
                    DualType::Barth => 0.0,
                    DualType::ThresholdBarth(l) => l,
                    DualType::Median => unreachable!(),
                }
                .max(1e-6);
                let bcoords = v.circumcenter_bcoords();
                if bcoords.iter().all(|&x| x > f) {
                    DualCellCenter::Vertex(v.vert(&bcoords))
                } else if bcoords[0] <= f {
                    DualCellCenter::Face(Edge::new(1, 2))
                } else if bcoords[1] <= f {
                    DualCellCenter::Face(Edge::new(2, 0))
                } else {
                    DualCellCenter::Face(Edge::new(1, 0))
                }
            }
        }
    }

    fn init_vertices(
        msh: &impl Mesh<2, C = Triangle<T>>,
        all_edges: &FxHashMap<Edge<T>, usize>,
        t: DualType,
    ) -> VertexBuild2d {
        let n_edges = all_edges.len();
        let n_elems = msh.n_elems();
        let mut bdy_verts: FxHashMap<usize, usize> =
            msh.faces().flatten().map(|i| (i, 0)).collect();
        let n_bdy_verts = bdy_verts.len();

        let mut verts = Vec::with_capacity(n_bdy_verts + n_edges + n_elems);
        for (&i_old, i_new) in &mut bdy_verts {
            *i_new = verts.len();
            verts.push(msh.vert(i_old));
        }

        verts.resize(verts.len() + n_edges, Vert2d::zeros());
        for (&edge, &i_edge) in all_edges {
            let ge = GEdge::new(&msh.vert(edge.get(0)), &msh.vert(edge.get(1)));
            verts[n_bdy_verts + i_edge] = ge.center();
        }

        let mut vert_idx_elem = vec![usize::MAX; n_elems];
        for (i_elem, e) in msh.elems().enumerate() {
            let ge = msh.gelem(&e);
            match Self::get_tri_center(&ge, t) {
                DualCellCenter::Vertex(center) => {
                    vert_idx_elem[i_elem] = verts.len();
                    verts.push(center);
                }
                DualCellCenter::Face(f) => {
                    let edge = Edge::new(e.get(f.get(0)), e.get(f.get(1))).sorted();
                    vert_idx_elem[i_elem] = n_bdy_verts + all_edges[&edge];
                }
            }
        }

        VertexBuild2d {
            verts,
            bdy_verts,
            vert_idx_elem,
            n_bdy_verts,
        }
    }

    fn init_poly_to_face_ptr(msh: &impl Mesh<2, C = Triangle<T>>) -> Vec<usize> {
        let mut ptr = vec![0; msh.n_verts() + 1];
        for e in msh.elems() {
            for edg in e.edges() {
                ptr[edg.get(0) + 1] += 1;
                ptr[edg.get(1) + 1] += 1;
            }
        }
        for f in msh.faces() {
            for v in f {
                ptr[v + 1] += 1;
            }
        }
        for i in 0..msh.n_verts() {
            ptr[i + 1] += ptr[i];
        }
        ptr
    }

    fn insert_face(
        poly_to_face: &mut [(usize, bool)],
        poly_to_face_ptr: &[usize],
        i_vert: usize,
        face_id: usize,
        orient: bool,
    ) {
        let slice = &mut poly_to_face[poly_to_face_ptr[i_vert]..poly_to_face_ptr[i_vert + 1]];
        for slot in slice {
            if slot.0 == usize::MAX {
                *slot = (face_id, orient);
                return;
            }
        }
        panic!("No free slot for dual face insertion");
    }

    fn build_internal_faces(
        msh: &impl Mesh<2, C = Triangle<T>>,
        all_edges: &FxHashMap<Edge<T>, usize>,
        vb: &VertexBuild2d,
        poly_to_face_ptr: &[usize],
        state: &mut FaceBuildState2d<T>,
    ) {
        for (i_elem, e) in msh.elems().enumerate() {
            for edg in e.edges() {
                let (i_edge, sgn) = if edg.get(0) < edg.get(1) {
                    let tmp = Edge::new(edg.get(0), edg.get(1));
                    (*all_edges.get(&tmp).unwrap(), 1.0)
                } else {
                    let tmp = Edge::new(edg.get(1), edg.get(0));
                    (*all_edges.get(&tmp).unwrap(), -1.0)
                };

                let face = Edge::new(vb.n_bdy_verts + i_edge, vb.vert_idx_elem[i_elem]);
                if face.get(0) == face.get(1) {
                    state.n_empty_faces += 1;
                    continue;
                }

                let gf = GEdge::new(&vb.verts[face.get(0)], &vb.verts[face.get(1)]);
                state.edge_normals[i_edge] += sgn * gf.normal(None);

                let i_new_face = state.faces.len();
                state.faces.push(face);
                state.ftags.push(0);
                Self::insert_face(
                    &mut state.poly_to_face,
                    poly_to_face_ptr,
                    edg.get(0),
                    i_new_face,
                    true,
                );
                Self::insert_face(
                    &mut state.poly_to_face,
                    poly_to_face_ptr,
                    edg.get(1),
                    i_new_face,
                    false,
                );
            }
        }
    }

    fn build_boundary_faces(
        msh: &impl Mesh<2, C = Triangle<T>>,
        all_edges: &FxHashMap<Edge<T>, usize>,
        vb: &VertexBuild2d,
        poly_to_face_ptr: &[usize],
        state: &mut FaceBuildState2d<T>,
    ) {
        for (f, tag) in msh.faces().zip(msh.ftags()) {
            let i_edge = all_edges[&f.sorted()];
            let v0 = vb.bdy_verts[&f.get(0)];
            let v1 = vb.bdy_verts[&f.get(1)];
            let em = vb.n_bdy_verts + i_edge;

            let f0 = Edge::new(v0, em);
            if f0.get(0) == f0.get(1) {
                state.n_empty_faces += 1;
            } else {
                let gf = GEdge::new(&vb.verts[f0.get(0)], &vb.verts[f0.get(1)]);
                state.bdy_faces.push((f.get(0), tag, gf.normal(None)));
                let id = state.faces.len();
                state.faces.push(f0);
                state.ftags.push(tag);
                Self::insert_face(
                    &mut state.poly_to_face,
                    poly_to_face_ptr,
                    f.get(0),
                    id,
                    true,
                );
            }

            let f1 = Edge::new(em, v1);
            if f1.get(0) == f1.get(1) {
                state.n_empty_faces += 1;
            } else {
                let gf = GEdge::new(&vb.verts[f1.get(0)], &vb.verts[f1.get(1)]);
                state.bdy_faces.push((f.get(0), tag, gf.normal(None)));
                let id = state.faces.len();
                state.faces.push(f1);
                state.ftags.push(tag);
                Self::insert_face(
                    &mut state.poly_to_face,
                    poly_to_face_ptr,
                    f.get(1),
                    id,
                    true,
                );
            }
        }
    }

    fn compact_poly_to_face(
        poly_to_face_ptr: &[usize],
        poly_to_face: &[(usize, bool)],
        n_verts: usize,
    ) -> (Vec<usize>, Vec<(usize, bool)>) {
        let mut new_ptr = Vec::with_capacity(poly_to_face_ptr.len());
        let mut new_map = Vec::new();
        new_ptr.push(0);
        for i in 0..n_verts {
            for v in poly_to_face
                .iter()
                .take(poly_to_face_ptr[i + 1])
                .skip(poly_to_face_ptr[i])
            {
                if v.0 != usize::MAX {
                    new_map.push(*v);
                }
            }
            new_ptr.push(new_map.len());
        }
        (new_ptr, new_map)
    }

    fn collect_edges_and_normals(
        all_edges: &FxHashMap<Edge<T>, usize>,
        edge_normals: &[Vert2d],
    ) -> (Vec<Edge<T>>, Vec<Vert2d>) {
        let mut edges = vec![Edge::default(); all_edges.len()];
        for (&edg, &i_edg) in all_edges {
            edges[i_edg] = edg;
        }

        let ids = sort_elem_min_ids(edges.iter().copied());
        let edges = ids
            .iter()
            .filter(|&&i| edge_normals[i].norm() > 1e-12)
            .map(|&i| edges[i])
            .collect::<Vec<_>>();
        let edge_normals = ids
            .iter()
            .filter(|&&i| edge_normals[i].norm() > 1e-12)
            .map(|&i| edge_normals[i])
            .collect::<Vec<_>>();

        (edges, edge_normals)
    }
}

impl<T: Idx> PolyMesh<2> for DualMesh2d<T> {
    fn poly_type(&self) -> PolyMeshType {
        PolyMeshType::Polygons
    }

    fn n_verts(&self) -> usize {
        self.verts.len()
    }

    fn n_elems(&self) -> usize {
        self.elem_to_face_ptr.len() - 1
    }

    fn n_faces(&self) -> usize {
        self.faces.len()
    }

    fn vert(&self, i: usize) -> Vert2d {
        self.verts[i]
    }

    fn etag(&self, i: usize) -> Tag {
        self.etags[i]
    }

    fn ftag(&self, i: usize) -> Tag {
        self.ftags[i]
    }

    fn elem(&self, i: usize) -> impl ExactSizeIterator<Item = (usize, bool)> + Clone {
        self.elem_to_face[self.elem_to_face_ptr[i]..self.elem_to_face_ptr[i + 1]]
            .iter()
            .copied()
    }

    fn face(&self, i: usize) -> impl ExactSizeIterator<Item = usize> + Clone + Send {
        self.faces[i].into_iter()
    }
}

impl<T: Idx> DualMesh<2> for DualMesh2d<T> {
    type C = Triangle<T>;

    fn new(msh: &impl Mesh<2, C = Self::C>, t: DualType) -> Self {
        let all_edges = msh.edges();
        let n_poly_faces = 3 * msh.n_elems() + 2 * msh.n_faces();
        let vb = Self::init_vertices(msh, &all_edges, t);
        let poly_to_face_ptr = Self::init_poly_to_face_ptr(msh);

        let mut state = FaceBuildState2d {
            faces: Vec::with_capacity(n_poly_faces),
            ftags: Vec::with_capacity(n_poly_faces),
            poly_to_face: vec![(usize::MAX, true); poly_to_face_ptr[msh.n_verts()]],
            edge_normals: vec![Vert2d::zeros(); all_edges.len()],
            bdy_faces: Vec::with_capacity(msh.n_faces() * 3),
            n_empty_faces: 0,
        };

        Self::build_internal_faces(msh, &all_edges, &vb, &poly_to_face_ptr, &mut state);
        Self::build_boundary_faces(msh, &all_edges, &vb, &poly_to_face_ptr, &mut state);

        assert_eq!(state.faces.len(), n_poly_faces - state.n_empty_faces);
        assert_eq!(state.ftags.len(), n_poly_faces - state.n_empty_faces);

        let (elem_to_face_ptr, elem_to_face) =
            Self::compact_poly_to_face(&poly_to_face_ptr, &state.poly_to_face, msh.n_verts());
        assert!(!elem_to_face.iter().any(|x| x.0 == usize::MAX));
        let (edges, edge_normals) =
            Self::collect_edges_and_normals(&all_edges, &state.edge_normals);

        Self {
            verts: vb.verts,
            faces: state.faces,
            ftags: state.ftags,
            elem_to_face_ptr,
            elem_to_face,
            etags: vec![1; msh.n_verts()],
            edges,
            edge_normals,
            bdy_faces: state.bdy_faces,
        }
    }

    fn n_edges(&self) -> usize {
        self.edges.len()
    }

    fn edge(&self, i: usize) -> Edge<T> {
        self.edges[i]
    }

    fn edge_normal(&self, i: usize) -> Vert2d {
        self.edge_normals[i]
    }

    fn n_boundary_faces(&self) -> usize {
        self.bdy_faces.len()
    }

    fn par_boundary_faces(&self) -> impl IndexedParallelIterator<Item = (usize, Tag, Vert2d)> + '_ {
        self.bdy_faces.par_iter().copied()
    }

    fn boundary_faces(&self) -> impl ExactSizeIterator<Item = (usize, Tag, Vert2d)> + '_ {
        self.bdy_faces.iter().copied()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::{
        Vert2d,
        dual::{DualMesh, DualMesh2d, DualType, PolyMesh},
        mesh::{Edge, GSimplex, Mesh, Mesh2d, rectangle_mesh},
    };
    use rayon::iter::ParallelIterator;

    #[test]
    fn test_dual_mesh_2d_simple() {
        let msh = rectangle_mesh::<Mesh2d>(1.0, 2, 1.0, 2);
        let dual = DualMesh2d::new(&msh, DualType::Median);
        dual.check().unwrap();

        assert_eq!(dual.n_verts(), 11);
        assert_eq!(dual.n_elems(), 4);
        assert_eq!(dual.n_faces(), 14);
        assert_eq!(dual.n_edges(), 5);

        assert!((dual.par_vols().sum::<f64>() - 1.0) < 1e-10);

        let n_empty_faces = dual
            .par_gfaces()
            .filter(|gf| gf.normal(None).norm() < 1e-12)
            .count();
        assert_eq!(n_empty_faces, 0);
    }

    #[test]
    fn test_dual_mesh_2d_simple_median() {
        let msh = rectangle_mesh::<Mesh2d>(2.0, 3, 1.0, 2);
        let dual = DualMesh2d::new(&msh, DualType::Median);
        dual.check().unwrap();

        assert_eq!(dual.n_verts(), 6 + 9 + 4);
        assert_eq!(dual.n_elems(), 6);
        assert_eq!(dual.n_faces(), 3 * 4 + 2 * 6);

        assert!((dual.par_vols().sum::<f64>() - 2.0) < 1e-10);

        let n_empty_faces = dual
            .par_gfaces()
            .filter(|gf| gf.normal(None).norm() < 1e-10)
            .count();
        assert_eq!(n_empty_faces, 0);

        let mut res = HashMap::new();
        res.insert(Edge::new(0, 1), Vert2d::new(1. / 3., -1. / 6.));
        res.insert(Edge::new(0, 4), Vert2d::new(1. / 3., 1. / 3.));
        res.insert(Edge::new(0, 3), Vert2d::new(-1. / 6., 1. / 3.));
        res.insert(Edge::new(3, 4), Vert2d::new(1. / 3., -1. / 6.));
        res.insert(Edge::new(1, 4), Vert2d::new(-1. / 3., 2. / 3.));
        res.insert(Edge::new(1, 2), Vert2d::new(1. / 3., -1. / 6.));
        res.insert(Edge::new(1, 5), Vert2d::new(1. / 3., 1. / 3.));
        res.insert(Edge::new(4, 5), Vert2d::new(1. / 3., -1. / 6.));
        res.insert(Edge::new(2, 5), Vert2d::new(-1. / 6., 1. / 3.));

        dual.par_edges_and_normals().for_each(|(e, n)| {
            let n_res = *res.get(&e).unwrap();
            assert!((n - n_res).norm() < 1e-10);
        });
    }

    #[test]
    fn test_dual_mesh_2d_simple_barth() {
        let msh = rectangle_mesh::<Mesh2d>(2.0, 3, 1.0, 2);
        let dual = DualMesh2d::new(&msh, DualType::Barth);
        dual.check().unwrap();

        let n_empty_faces = dual
            .par_gfaces()
            .filter(|gf| gf.normal(None).norm() < 1e-10)
            .count();
        assert_eq!(n_empty_faces, 0);

        let n_faces_removed = 4;
        assert_eq!(dual.n_verts(), 6 + 9 + 4 - n_faces_removed);
        assert_eq!(dual.n_elems(), 6);
        assert_eq!(dual.n_faces(), 3 * 4 + 2 * 6 - n_faces_removed);

        assert!((dual.par_vols().sum::<f64>() - 2.0) < 1e-10);

        let mut res = HashMap::new();
        res.insert(Edge::new(0, 1), Vert2d::new(0.5, 0.0));
        res.insert(Edge::new(0, 4), Vert2d::new(0.0, 0.0));
        res.insert(Edge::new(0, 3), Vert2d::new(0.0, 0.5));
        res.insert(Edge::new(3, 4), Vert2d::new(0.5, 0.0));
        res.insert(Edge::new(1, 4), Vert2d::new(0.0, 1.0));
        res.insert(Edge::new(1, 2), Vert2d::new(0.5, 0.0));
        res.insert(Edge::new(1, 5), Vert2d::new(0.0, 0.0));
        res.insert(Edge::new(4, 5), Vert2d::new(0.5, 0.0));
        res.insert(Edge::new(2, 5), Vert2d::new(0.0, 0.5));

        dual.par_edges_and_normals().for_each(|(e, n)| {
            let n_res = *res.get(&e).unwrap();
            assert!((n - n_res).norm() < 1e-10);
        });
    }

    #[test]
    fn test_dual_mesh_2d_simple_barth_2() {
        let msh = rectangle_mesh::<Mesh2d>(2.0, 30, 1.0, 20).random_shuffle();
        let dual = DualMesh2d::new(&msh, DualType::Barth);
        dual.check().unwrap();

        assert!((dual.par_vols().sum::<f64>() - 2.0) < 1e-10);
    }
}
