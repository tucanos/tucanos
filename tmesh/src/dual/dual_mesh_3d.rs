//! Computation of the 3D dual mesh for tetrahedral primal meshes (`Mesh<3>` with
//! `C = Tetrahedron<T>`).
//!
//! The constructor [`DualMesh3d::new`] builds a polyhedral dual control-volume
//! around each primal vertex with the following algorithm:
//!
//! 1. Enumerate primal edges and faces.
//! 2. Create dual vertices from:
//!    - primal boundary vertices,
//!    - primal edge midpoints,
//!    - one center per primal face,
//!    - one center per primal tetrahedron.
//! 3. For face/tetra centers, choose location according to [`DualType`]:
//!    - `Median`: geometric center,
//!    - `Barth` / `ThresholdBarth`: circumcenter when admissible, otherwise
//!      fallback to a lower-dimensional center (edge or face) to keep robust
//!      dual cells on obtuse/degenerate configurations.
//! 4. Build internal dual triangular faces from edge-center / face-center /
//!    cell-center triplets.
//! 5. Build boundary dual triangular faces from boundary-vertex / edge-center /
//!    face-center triplets.
//! 6. Deduplicate dual faces (needed for Barth-like constructions), then build
//!    per-dual-cell face incidence with orientation.
//! 7. Accumulate oriented edge-based normals from internal dual faces and filter
//!    degenerate/zero-norm contributions.
//!
//! TODO (possible improvements):
//! - Add dedicated tests for `Barth` and `ThresholdBarth` on pathological
//!   tetrahedra (high obtuseness, near-coplanar configurations).
//! - Add explicit diagnostics for deduplication and fallback-center usage
//!   frequencies to support quality analysis.
//! - Strengthen geometric validation for boundary orientation and face normal
//!   consistency in debug builds.
//! - Reduce temporary memory pressure in face construction/deduplication for
//!   large meshes.
use super::{DualCellCenter, DualMesh, DualType, PolyMesh, PolyMeshType};
use crate::{
    Tag, Vert3d,
    dual::poly_mesh::PolyFaceType,
    mesh::{
        Edge, FaceConnectivity, GEdge, GSimplex, GTetrahedron, GTriangle, Idx, Mesh, Simplex,
        Tetrahedron, Triangle, sort_elem_min_ids,
    },
};
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use rustc_hash::{FxBuildHasher, FxHashMap};

/// Dual of a 3D tetrahedral primal mesh.
///
/// The dual mesh stores polyhedral control volumes centered on primal
/// vertices, together with edge-based normals typically used by finite-volume
/// discretizations.
pub struct DualMesh3d<T: Idx> {
    verts: Vec<Vert3d>,
    faces: Vec<Triangle<T>>,
    ftags: Vec<Tag>,
    elem_to_face_ptr: Vec<usize>,
    elem_to_face: Vec<(usize, bool)>,
    etags: Vec<Tag>,
    edges: Vec<Edge<T>>,
    edge_normals: Vec<Vert3d>,
    bdy_faces: Vec<(usize, Tag, Vert3d)>,
}

struct VertexBuild3d {
    verts: Vec<Vert3d>,
    bdy_verts: FxHashMap<usize, usize>,
    n_bdy_verts: usize,
    vert_idx_face: Vec<usize>,
    vert_idx_elem: Vec<usize>,
}

struct FaceBuildState3d<T: Idx> {
    tmp_faces: FxHashMap<Triangle<T>, (usize, Tag)>,
    poly_to_face: Vec<(usize, bool)>,
    edge_normals: Vec<Vert3d>,
    bdy_faces: Vec<(usize, Tag, Vert3d)>,
    n_empty_faces: usize,
}

struct FinalizedFaces3d<T: Idx> {
    faces: Vec<Triangle<T>>,
    ftags: Vec<Tag>,
    poly_to_face: Vec<(usize, bool)>,
    new_face_idx: Vec<usize>,
}

impl<T: Idx> DualMesh3d<T> {
    fn get_tet_center(v: &GTetrahedron<3>, t: DualType) -> DualCellCenter<3, Tetrahedron<T>> {
        match t {
            DualType::Median => DualCellCenter::Vertex(v.center()),
            DualType::Barth | DualType::ThresholdBarth(_) => {
                let f = match t {
                    DualType::Barth => 0.0,
                    DualType::ThresholdBarth(l) => l,
                    DualType::Median => unreachable!(),
                };
                let f = f.max(1e-6);
                let bcoords = v.circumcenter_bcoords();
                if bcoords.iter().all(|&x| x > f) {
                    DualCellCenter::Vertex(v.vert(&bcoords))
                } else if bcoords[0] <= f {
                    DualCellCenter::Face(Triangle::new(1, 2, 3))
                } else if bcoords[1] <= f {
                    DualCellCenter::Face(Triangle::new(2, 0, 3))
                } else if bcoords[2] <= f {
                    DualCellCenter::Face(Triangle::new(0, 1, 3))
                } else {
                    DualCellCenter::Face(Triangle::new(0, 2, 1))
                }
            }
        }
    }
    fn get_tri_center(v: &GTriangle<3>, t: DualType) -> DualCellCenter<3, Triangle<T>> {
        match t {
            DualType::Median => DualCellCenter::Vertex(v.center()),
            DualType::Barth | DualType::ThresholdBarth(_) => {
                let f = match t {
                    DualType::Barth => 0.0,
                    DualType::ThresholdBarth(l) => l,
                    DualType::Median => unreachable!(),
                };
                let bcoords = v.circumcenter_bcoords();
                if bcoords.iter().all(|&x| x >= f) {
                    DualCellCenter::Vertex(v.vert(&bcoords))
                } else if bcoords[0] < f {
                    DualCellCenter::Face(Edge::new(1, 2))
                } else if bcoords[1] < f {
                    DualCellCenter::Face(Edge::new(2, 0))
                } else {
                    DualCellCenter::Face(Edge::new(1, 0))
                }
            }
        }
    }

    fn init_vertices(
        msh: &impl Mesh<3, C = Tetrahedron<T>>,
        all_edges: &FxHashMap<Edge<T>, usize>,
        all_faces: &FaceConnectivity<Triangle<T>>,
        t: DualType,
    ) -> VertexBuild3d {
        let n_edges = all_edges.len();
        let n_faces = all_faces.len();
        let n_elems = msh.n_elems();

        let mut bdy_verts: FxHashMap<usize, usize> =
            msh.faces().flatten().map(|i| (i, 0)).collect();
        for (i, (_, i_new)) in bdy_verts.iter_mut().enumerate() {
            *i_new = i;
        }
        let n_bdy_verts = bdy_verts.len();

        let mut verts = Vec::with_capacity(n_bdy_verts + n_edges + n_faces + n_elems);
        for (&i_old, i_new) in &mut bdy_verts {
            *i_new = verts.len();
            verts.push(msh.vert(i_old));
        }

        let vert_idx_edge = |i: usize| i + n_bdy_verts;
        verts.resize(verts.len() + n_edges, Vert3d::zeros());
        for (&edge, &i_edge) in all_edges {
            let ge = GEdge::new(&msh.vert(edge.get(0)), &msh.vert(edge.get(1)));
            verts[vert_idx_edge(i_edge)] = ge.center();
        }

        let mut vert_idx_face = vec![usize::MAX; n_faces];
        for (f, &(i_face, _, _)) in all_faces {
            match Self::get_tri_center(&msh.gface(f), t) {
                DualCellCenter::Vertex(center) => {
                    vert_idx_face[i_face] = verts.len();
                    verts.push(center);
                }
                DualCellCenter::Face(e) => {
                    let edge = Edge::new(f.get(e.get(0)), f.get(e.get(1))).sorted();
                    vert_idx_face[i_face] = vert_idx_edge(all_edges[&edge]);
                }
            }
        }

        let mut vert_idx_elem = vec![usize::MAX; n_elems];
        for (i_elem, e) in msh.elems().enumerate() {
            match Self::get_tet_center(&msh.gelem(&e), t) {
                DualCellCenter::Vertex(center) => {
                    vert_idx_elem[i_elem] = verts.len();
                    verts.push(center);
                }
                DualCellCenter::Face(f) => {
                    let face =
                        Triangle::new(e.get(f.get(0)), e.get(f.get(1)), e.get(f.get(2))).sorted();
                    let i_face = all_faces[&face].0;
                    vert_idx_elem[i_elem] = vert_idx_face[i_face];
                }
            }
        }

        VertexBuild3d {
            verts,
            bdy_verts,
            n_bdy_verts,
            vert_idx_face,
            vert_idx_elem,
        }
    }

    fn init_poly_to_face_ptr(msh: &impl Mesh<3, C = Tetrahedron<T>>) -> Vec<usize> {
        let mut ptr = vec![0; msh.n_verts() + 1];
        for e in msh.elems() {
            for edg in e.edges() {
                ptr[edg.get(0) + 1] += 2;
                ptr[edg.get(1) + 1] += 2;
            }
        }
        for f in msh.faces() {
            for edg in f.edges() {
                ptr[edg.get(0) + 1] += 1;
                ptr[edg.get(1) + 1] += 1;
            }
        }
        for i in 0..msh.n_verts() {
            ptr[i + 1] += ptr[i];
        }
        ptr
    }

    fn insert_internal_face(slice: &mut [(usize, bool)], face_id: usize, orient: bool) {
        let n = slice
            .iter_mut()
            .filter(|(i, _)| *i == face_id)
            .map(|(i, _)| *i = usize::MAX)
            .count();
        if n == 0 {
            for j in slice {
                if j.0 == usize::MAX {
                    *j = (face_id, orient);
                    return;
                }
            }
            panic!("No free slot for internal dual face insertion");
        }
        assert_eq!(n, 1);
    }

    fn insert_boundary_face(slice: &mut [(usize, bool)], face_id: usize, orient: bool) {
        assert_eq!(slice.iter().filter(|(i, _)| *i == face_id).count(), 0);
        for j in slice {
            if j.0 == usize::MAX {
                *j = (face_id, orient);
                return;
            }
        }
        panic!("No free slot for boundary dual face insertion");
    }

    fn build_internal_faces(
        msh: &impl Mesh<3, C = Tetrahedron<T>>,
        all_edges: &FxHashMap<Edge<T>, usize>,
        all_faces: &FaceConnectivity<Triangle<T>>,
        vb: &VertexBuild3d,
        poly_to_face_ptr: &[usize],
        state: &mut FaceBuildState3d<T>,
    ) {
        let vert_idx_edge = |i: usize| i + vb.n_bdy_verts;
        for (i_elem, e) in msh.elems().enumerate() {
            for f in e.faces() {
                let i_face = all_faces[&f.sorted()].0;
                for edg in f.edges() {
                    let (i_edge, sgn) = if edg.get(0) < edg.get(1) {
                        let tmp = Edge::new(edg.get(0), edg.get(1));
                        (*all_edges.get(&tmp).unwrap(), 1.0)
                    } else {
                        let tmp = Edge::new(edg.get(1), edg.get(0));
                        (*all_edges.get(&tmp).unwrap(), -1.0)
                    };

                    let face = Triangle::new(
                        vert_idx_edge(i_edge),
                        vb.vert_idx_elem[i_elem],
                        vb.vert_idx_face[i_face],
                    );
                    let skip = face.get(0) == face.get(1)
                        || face.get(0) == face.get(2)
                        || face.get(1) == face.get(2);
                    if skip {
                        state.n_empty_faces += 1;
                        continue;
                    }

                    let gf = GTriangle::new(
                        &vb.verts[face.get(0)],
                        &vb.verts[face.get(1)],
                        &vb.verts[face.get(2)],
                    );
                    state.edge_normals[i_edge] += sgn * gf.normal(None);

                    let sorted_face = face.sorted();
                    let is_sorted = face.is_same(&sorted_face);
                    let i_new = if let Some((i_face, _)) = state.tmp_faces.get(&sorted_face) {
                        *i_face
                    } else {
                        let i_face = state.tmp_faces.len();
                        state.tmp_faces.insert(sorted_face, (i_face, 0));
                        i_face
                    };

                    let s0 = &mut state.poly_to_face
                        [poly_to_face_ptr[edg.get(0)]..poly_to_face_ptr[edg.get(0) + 1]];
                    Self::insert_internal_face(s0, i_new, is_sorted);

                    let s1 = &mut state.poly_to_face
                        [poly_to_face_ptr[edg.get(1)]..poly_to_face_ptr[edg.get(1) + 1]];
                    Self::insert_internal_face(s1, i_new, !is_sorted);
                }
            }
        }
    }

    fn build_boundary_faces(
        msh: &impl Mesh<3, C = Tetrahedron<T>>,
        all_edges: &FxHashMap<Edge<T>, usize>,
        all_faces: &FaceConnectivity<Triangle<T>>,
        vb: &VertexBuild3d,
        poly_to_face_ptr: &[usize],
        state: &mut FaceBuildState3d<T>,
    ) {
        let vert_idx_edge = |i: usize| i + vb.n_bdy_verts;
        let vert_ids_bdy = |i: usize| vb.bdy_verts[&i];
        for (f, tag) in msh.faces().zip(msh.ftags()) {
            let i_face = all_faces[&f.sorted()].0;
            for edg in f.edges() {
                let i_edge = all_edges[&Edge::from_iter(edg.sorted())];
                for (i_v, face) in [
                    (
                        edg.get(0),
                        Triangle::new(
                            vert_ids_bdy(edg.get(0)),
                            vert_idx_edge(i_edge),
                            vb.vert_idx_face[i_face],
                        ),
                    ),
                    (
                        edg.get(1),
                        Triangle::new(
                            vert_ids_bdy(edg.get(1)),
                            vb.vert_idx_face[i_face],
                            vert_idx_edge(i_edge),
                        ),
                    ),
                ] {
                    let skip = face.get(0) == face.get(1)
                        || face.get(0) == face.get(2)
                        || face.get(1) == face.get(2);
                    if skip {
                        state.n_empty_faces += 1;
                        continue;
                    }

                    let gf = GTriangle::new(
                        &vb.verts[face.get(0)],
                        &vb.verts[face.get(1)],
                        &vb.verts[face.get(2)],
                    );
                    state.bdy_faces.push((edg.get(0), tag, gf.normal(None)));

                    let sorted_face = face.sorted();
                    let is_sorted = face.is_same(&sorted_face);
                    let i_new = if let Some((i_face, _)) = state.tmp_faces.get(&sorted_face) {
                        *i_face
                    } else {
                        let i_face = state.tmp_faces.len();
                        state.tmp_faces.insert(sorted_face, (i_face, tag));
                        i_face
                    };

                    let slice =
                        &mut state.poly_to_face[poly_to_face_ptr[i_v]..poly_to_face_ptr[i_v + 1]];
                    Self::insert_boundary_face(slice, i_new, is_sorted);
                }
            }
        }
    }

    fn finalize_faces(
        mut state: FaceBuildState3d<T>,
        n_poly_faces: usize,
        t: DualType,
    ) -> FinalizedFaces3d<T> {
        assert!(state.tmp_faces.len() <= n_poly_faces - state.n_empty_faces);
        if matches!(t, DualType::Median) {
            assert_eq!(state.tmp_faces.len(), n_poly_faces - state.n_empty_faces);
        }

        let n = state.tmp_faces.len();
        let mut new_face_idx = vec![0; n];
        state
            .poly_to_face
            .iter()
            .filter(|&i| i.0 != usize::MAX)
            .for_each(|&i| new_face_idx[i.0] += 1);

        let mut count = 0;
        for i in &mut new_face_idx {
            if *i != 0 {
                assert!(*i <= 2);
                *i = count;
                count += 1;
            } else {
                *i = usize::MAX;
            }
        }
        if matches!(t, DualType::Median) {
            assert_eq!(count, n);
        }

        let mut faces = vec![Triangle::default(); count];
        let mut ftags = vec![0; count];
        for (face, (i_old, tag)) in state.tmp_faces.drain() {
            let i = new_face_idx[i_old];
            if i != usize::MAX {
                faces[i] = face;
                ftags[i] = tag;
            }
        }

        FinalizedFaces3d {
            faces,
            ftags,
            poly_to_face: state.poly_to_face,
            new_face_idx,
        }
    }

    fn compact_poly_to_face(
        msh: &impl Mesh<3, C = Tetrahedron<T>>,
        poly_to_face_ptr: &[usize],
        poly_to_face: &[(usize, bool)],
        new_face_idx: &[usize],
    ) -> (Vec<usize>, Vec<(usize, bool)>) {
        let n = poly_to_face.iter().filter(|&i| i.0 != usize::MAX).count();
        let mut new_ptr = Vec::with_capacity(poly_to_face_ptr.len());
        let mut new_map = Vec::with_capacity(n);
        new_ptr.push(0);
        for i_elem in 0..msh.n_verts() {
            for v in poly_to_face
                .iter()
                .take(poly_to_face_ptr[i_elem + 1])
                .skip(poly_to_face_ptr[i_elem])
            {
                if v.0 != usize::MAX {
                    new_map.push((new_face_idx[v.0], v.1));
                }
            }
            new_ptr.push(new_map.len());
        }
        (new_ptr, new_map)
    }

    fn collect_edges_and_normals(
        all_edges: &FxHashMap<Edge<T>, usize>,
        edge_normals: &[Vert3d],
    ) -> (Vec<Edge<T>>, Vec<Vert3d>) {
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

    #[must_use]
    pub fn vols(&self) -> impl ExactSizeIterator<Item = f64> + '_ {
        <Self as PolyMesh<3>>::vols_c::<Triangle<T>>(self)
    }

    #[must_use]
    pub fn par_vols(&self) -> impl IndexedParallelIterator<Item = f64> + '_ {
        <Self as PolyMesh<3>>::par_vols_c::<Triangle<T>>(self)
    }
}

impl<T: Idx> PolyMesh<3> for DualMesh3d<T> {
    fn poly_type(&self) -> PolyMeshType {
        PolyMeshType::Polyhedra
    }

    fn n_verts(&self) -> usize {
        self.verts.len()
    }

    fn vert(&self, i: usize) -> Vert3d {
        self.verts[i]
    }

    fn n_elems(&self) -> usize {
        self.elem_to_face_ptr.len() - 1
    }

    fn elem(&self, i: usize) -> impl ExactSizeIterator<Item = (usize, bool)> + Clone + Send {
        let start = self.elem_to_face_ptr[i];
        let end = self.elem_to_face_ptr[i + 1];
        self.elem_to_face[start..end].iter().copied()
    }

    fn etag(&self, i: usize) -> Tag {
        self.etags[i]
    }

    fn n_faces(&self) -> usize {
        self.faces.len()
    }

    fn face(&self, i: usize) -> impl ExactSizeIterator<Item = usize> + Clone + Send {
        self.faces[i].into_iter()
    }

    fn ftag(&self, i: usize) -> Tag {
        self.ftags[i]
    }

    fn face_type(&self) -> PolyFaceType {
        PolyFaceType::Simplices
    }

    fn elem_gfaces_c<C: Simplex>(
        &self,
        i: usize,
    ) -> Option<impl ExactSizeIterator<Item = C::GEOM<3>> + '_> {
        if C::DIM != 2 {
            return None;
        }
        let gfaces = self.elem(i).map(|(i, orient)| {
            let mut f = C::from_iter(self.face(i));
            if !orient {
                f.invert();
            }
            C::GEOM::from_iter(f.into_iter().map(|i| self.vert(i)))
        });
        Some(gfaces)
    }
}

impl<T: Idx> DualMesh<3> for DualMesh3d<T> {
    type C = Tetrahedron<T>;

    fn new(msh: &impl Mesh<3, C = Tetrahedron<T>>, t: DualType) -> Self {
        let all_edges = msh.edges();
        let all_faces = msh.all_faces();
        let n_poly_faces = 12 * msh.n_elems() + 6 * msh.n_faces();
        let vb = Self::init_vertices(msh, &all_edges, &all_faces, t);
        let poly_to_face_ptr = Self::init_poly_to_face_ptr(msh);

        let mut state = FaceBuildState3d {
            tmp_faces: FxHashMap::with_capacity_and_hasher(n_poly_faces, FxBuildHasher),
            poly_to_face: vec![(usize::MAX, true); poly_to_face_ptr[poly_to_face_ptr.len() - 1]],
            edge_normals: vec![Vert3d::zeros(); all_edges.len()],
            bdy_faces: Vec::with_capacity(msh.n_faces() * 6),
            n_empty_faces: 0,
        };

        Self::build_internal_faces(
            msh,
            &all_edges,
            &all_faces,
            &vb,
            &poly_to_face_ptr,
            &mut state,
        );
        Self::build_boundary_faces(
            msh,
            &all_edges,
            &all_faces,
            &vb,
            &poly_to_face_ptr,
            &mut state,
        );

        let bdy_faces = state.bdy_faces.clone();
        let edge_normals_raw = state.edge_normals.clone();
        let finalized = Self::finalize_faces(state, n_poly_faces, t);
        let (elem_to_face_ptr, elem_to_face) = Self::compact_poly_to_face(
            msh,
            &poly_to_face_ptr,
            &finalized.poly_to_face,
            &finalized.new_face_idx,
        );

        assert!(!elem_to_face.iter().any(|&i| i.0 == usize::MAX));
        let (edges, edge_normals) = Self::collect_edges_and_normals(&all_edges, &edge_normals_raw);

        Self {
            verts: vb.verts,
            faces: finalized.faces,
            ftags: finalized.ftags,
            elem_to_face_ptr,
            elem_to_face,
            etags: vec![1; msh.n_verts()],
            edges,
            edge_normals,
            bdy_faces,
        }
    }

    fn n_edges(&self) -> usize {
        self.edges.len()
    }

    fn edge(&self, i: usize) -> Edge<T> {
        self.edges[i]
    }

    fn edge_normal(&self, i: usize) -> Vert3d {
        self.edge_normals[i]
    }

    fn n_boundary_faces(&self) -> usize {
        self.bdy_faces.len()
    }
    fn par_boundary_faces(&self) -> impl IndexedParallelIterator<Item = (usize, Tag, Vert3d)> + '_ {
        self.bdy_faces.par_iter().copied()
    }
    fn boundary_faces(&self) -> impl ExactSizeIterator<Item = (usize, Tag, Vert3d)> + '_ {
        self.bdy_faces.iter().copied()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Vert3d, assert_delta,
        dual::{DualMesh, DualMesh3d, DualType, PolyMesh},
        mesh::{Hexahedron, Mesh, Mesh3d, Triangle, box_mesh},
    };
    use rayon::iter::ParallelIterator;

    #[test]
    fn test_dual_mesh_3d_simple_median() {
        let msh = box_mesh::<Mesh3d>(1.0, 2, 2.0, 2, 1.0, 2);
        let dual = DualMesh3d::new(&msh, DualType::Median);
        dual.check().unwrap();

        assert!((dual.par_vols().sum::<f64>() - 2.0) < 1e-10);

        dual.write_vtk("median3d.vtu").unwrap();

        // let (bdy, _): (BoundaryMesh3d, _) = dual.boundary();
        // bdy.write_vtk("median3d_bdy.vtu").unwrap();

        // let poly = SimplePolyMesh::<3>::simplify(&dual, true);
        // poly.write_vtk("median3d_simplified.vtu").unwrap();
    }

    #[test]
    fn test_dual_mesh_3d_simple_barth() {
        let msh = box_mesh::<Mesh3d>(1.0, 2, 2.0, 2, 1.0, 2);
        msh.write_vtk("mesh3d.vtu").unwrap();

        let dual = DualMesh3d::new(&msh, DualType::Median);
        dual.check().unwrap();

        assert!((dual.par_vols().sum::<f64>() - 2.0) < 1e-10);

        dual.write_vtk("barth3d.vtu").unwrap();

        // let (bdy, _): (BoundaryMesh3d, _) = dual.boundary();
        // bdy.write_vtk("barth3d_bdy.vtu").unwrap();

        // let poly = SimplePolyMesh::<3>::simplify(&dual, true);
        // poly.write_vtk("barth3d_simplified.vtu").unwrap();
    }

    #[test]
    fn test_split_elements_l_shape_3d() {
        let mut msh = Mesh3d::empty();

        // L-shape extruded on z in [0, 1] with no top-right block.
        // Coordinates are in {0,1,4} x {0,1,4} x {0,1} minus vertices at x=4,y=4.
        let verts = [
            Vert3d::new(0.0, 0.0, 0.0),
            Vert3d::new(1.0, 0.0, 0.0),
            Vert3d::new(4.0, 0.0, 0.0),
            Vert3d::new(0.0, 1.0, 0.0),
            Vert3d::new(1.0, 1.0, 0.0),
            Vert3d::new(4.0, 1.0, 0.0),
            Vert3d::new(0.0, 4.0, 0.0),
            Vert3d::new(1.0, 4.0, 0.0),
            Vert3d::new(0.0, 0.0, 1.0),
            Vert3d::new(1.0, 0.0, 1.0),
            Vert3d::new(4.0, 0.0, 1.0),
            Vert3d::new(0.0, 1.0, 1.0),
            Vert3d::new(1.0, 1.0, 1.0),
            Vert3d::new(4.0, 1.0, 1.0),
            Vert3d::new(0.0, 4.0, 1.0),
            Vert3d::new(1.0, 4.0, 1.0),
        ];
        msh.add_verts(verts.into_iter());

        let hexas = [
            // lower-left block: x in [0,1], y in [0,1]
            Hexahedron::new([0, 1, 4, 3, 8, 9, 12, 11]),
            // lower-right block: x in [1,4], y in [0,1]
            Hexahedron::new([1, 2, 5, 4, 9, 10, 13, 12]),
            // upper-left block: x in [0,1], y in [1,4]
            Hexahedron::new([3, 4, 7, 6, 11, 12, 15, 14]),
        ];
        msh.add_hexahedra(hexas.iter().copied(), (0..hexas.len()).map(|_| 1));
        msh.fix().unwrap();

        assert_delta!(msh.vol(), 7.0, 1e-10);

        let dual = DualMesh3d::new(&msh, DualType::Median);
        dual.check().unwrap();

        let split = dual.split_elements(&msh);
        split.check().unwrap();

        assert_delta!(split.vols_c::<Triangle<usize>>().sum::<f64>(), 7.0, 1e-10);
    }
}
