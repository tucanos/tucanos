//! Simplex meshes in D dimensions, represented by
//!   - the vertices
//!   - elements of type `C` and element tags
//!   - faces of type `C::FACE` and element tags
//!
//! F = C-1 cannot be imposed in rust stable
mod boundary_mesh_2d;
mod boundary_mesh_3d;
mod elements;
mod mesh_2d;
mod mesh_3d;

mod split;

mod hilbert;
pub mod partition;

pub mod gradient;

mod vector;

use log::{debug, warn};
use minimeshb::{reader::MeshbReader, writer::MeshbWriter};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use rayon::{
    prelude::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};

use crate::{
    Error, Result, Tag, Vertex,
    graph::CSRGraph,
    io::{VTUEncoding, VTUFile},
    mesh::gradient::{l2proj, least_squares},
    spatialindex::PointIndex,
};
pub use boundary_mesh_2d::{
    BoundaryMesh2d, QuadraticBoundaryMesh2d, circle_mesh, quadratic_circle_mesh,
    to_quadratic_edge_mesh,
};
pub use boundary_mesh_3d::{
    BoundaryMesh3d, QuadraticBoundaryMesh3d, quadratic_sphere_mesh, read_stl, sphere_mesh,
    to_quadratic_triangle_mesh,
};
pub use elements::{
    Hexahedron, Idx, Prism, Pyramid, Quadrangle,
    edge::{Edge, GEdge},
    node::{GNode, Node},
    quadratic_edge::{QuadraticEdge, QuadraticGEdge},
    quadratic_triangle::{QuadraticGTriangle, QuadraticTriangle},
    simplex::{GSimplex, Simplex, get_face_to_elem},
    tetrahedron::{GTetrahedron, Tetrahedron},
    to_simplices::{hex2tets, pri2tets, pyr2tets, qua2tris},
    triangle::{GTriangle, Triangle},
};
use hilbert::hilbert_indices;
pub use mesh_2d::{Mesh2d, nonuniform_rectangle_mesh, rectangle_mesh};
pub use mesh_3d::{Mesh3d, ball_mesh, box_mesh, nonuniform_box_mesh};
use partition::Partitioner;
use split::{split_edgs, split_tets, split_tris};
pub use vector::Vector;

pub(crate) fn sort_elem_min_ids<C: Simplex>(elems: impl ExactSizeIterator<Item = C>) -> Vec<usize> {
    let n_elems = elems.len();

    let min_ids = elems.map(|e| e.into_iter().min()).collect::<Vec<_>>();
    let mut indices = (0..n_elems).collect::<Vec<_>>();
    indices.sort_by_key(|&i| min_ids[i]);
    indices
}

/// Compute the maximum and average bandwidth of a connectivity
pub fn bandwidth<C: Simplex>(elems: impl ExactSizeIterator<Item = C>) -> (usize, f64) {
    let n_elems = elems.len();

    let (bmax, bmean) = elems.fold((0_usize, 0_usize), |a, e| {
        let max_id = e.into_iter().max().unwrap();
        let min_id = e.into_iter().min().unwrap();
        let tmp = max_id - min_id;
        (a.0.max(tmp), a.1 + tmp)
    });
    let bmean = bmean as f64 / n_elems as f64;
    (bmax, bmean)
}

/// Submesh of a `Mesh<D, C, F>`, with information about the vertices, element
/// and face ids in the parent mesh
pub struct SubMesh<const D: usize, M: Mesh<D>> {
    /// Mesh
    pub mesh: M,
    /// Indices of the vertices of `mesh` in the parent mesh
    pub parent_vert_ids: Vec<usize>,
    /// Indices of the element of `mesh` in the parent mesh
    pub parent_elem_ids: Vec<usize>,
    /// Indices of the faces of `mesh` in the parent mesh
    pub parent_face_ids: Vec<usize>,
}

impl<const D: usize, M: Mesh<D>> SubMesh<D, M> {
    /// Extract the elements with a given tag
    pub fn new<G: Fn(Tag) -> bool>(mesh: &M, filter: G) -> Self {
        let mut res = M::empty();
        let (parent_vert_ids, parent_elem_ids, parent_face_ids) =
            res.add(mesh, filter, |_| true, None);
        Self {
            mesh: res,
            parent_vert_ids,
            parent_elem_ids,
            parent_face_ids,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum GradientMethod {
    LinearLeastSquares(i32),
    QuadraticLeastSquares(i32),
    L2Projection,
}

pub type FixedTags = (FxHashMap<Tag, Tag>, FxHashMap<[Tag; 2], Tag>);
/// D-dimensional simplex mesh
pub trait Mesh<const D: usize>: Send + Sync + Sized {
    type C: Simplex;

    /// Create a new mesh from slices of Vertex, Cell and Face
    #[must_use]
    fn new(
        verts: &[Vertex<D>],
        elems: &[Self::C],
        etags: &[Tag],
        faces: &[<Self::C as Simplex>::FACE],
        ftags: &[Tag],
    ) -> Self {
        let mut res = Self::empty();
        res.add_verts(verts.iter().copied());
        res.add_elems(elems.iter().copied(), etags.iter().copied());
        res.add_faces(faces.iter().copied(), ftags.iter().copied());
        res
    }

    /// Create an empty mesh
    fn empty() -> Self;

    /// Number of vertices
    fn n_verts(&self) -> usize;

    /// Get the `i`th vertex
    fn vert(&self, i: usize) -> Vertex<D>;

    /// Parallel iterator over the vertices
    fn par_verts(&self) -> impl IndexedParallelIterator<Item = Vertex<D>> + Clone + '_;

    /// Sequential iterator over the vertices
    fn verts(&self) -> impl ExactSizeIterator<Item = Vertex<D>> + Clone + '_;

    /// Add vertices to the mesh
    fn add_verts(&mut self, v: impl ExactSizeIterator<Item = Vertex<D>>);

    /// Number of elements
    fn n_elems(&self) -> usize;

    /// Get the `i`th element
    fn elem(&self, i: usize) -> Self::C;

    /// Invert the `i`th element
    fn invert_elem(&mut self, i: usize);

    /// Parallel iterator over the mesh elements
    fn par_elems(&self) -> impl IndexedParallelIterator<Item = Self::C> + Clone + '_;

    /// Sequential iterator over the mesh elements
    fn elems(&self) -> impl ExactSizeIterator<Item = Self::C> + Clone + '_;

    /// Add elements to the mesh
    fn add_elems<I1: ExactSizeIterator<Item = Self::C>, I2: ExactSizeIterator<Item = Tag>>(
        &mut self,
        elems: I1,
        etags: I2,
    );

    /// Remove all the elements
    fn clear_elems(&mut self);

    /// Add elements to the mesh
    fn add_elems_and_tags(&mut self, elems_and_tags: impl ExactSizeIterator<Item = (Self::C, Tag)>);

    /// Get the tag of the `i`th element
    fn etag(&self, i: usize) -> Tag;

    /// Parallel iterator over the element tags
    fn par_etags(&self) -> impl IndexedParallelIterator<Item = Tag> + Clone + '_;

    /// Sequential iterator over the element tags
    fn etags(&self) -> impl ExactSizeIterator<Item = Tag> + Clone + '_;

    /// Get the vertices of element `e`
    fn gelem(&self, e: &Self::C) -> <Self::C as Simplex>::GEOM<D> {
        <Self::C as Simplex>::GEOM::from_iter(e.into_iter().map(|i| self.vert(i)))
    }

    /// Parallel iterator over element vertices
    fn par_gelems(
        &self,
    ) -> impl IndexedParallelIterator<Item = <Self::C as Simplex>::GEOM<D>> + Clone + '_ {
        self.par_elems().map(|e| self.gelem(&e))
    }

    /// Sequential iterator over element vertices
    fn gelems(&self) -> impl ExactSizeIterator<Item = <Self::C as Simplex>::GEOM<D>> + Clone + '_ {
        self.elems().map(|e| self.gelem(&e))
    }

    /// Number of faces
    fn n_faces(&self) -> usize;

    /// Get the `i`th face
    fn face(&self, i: usize) -> <Self::C as Simplex>::FACE;

    /// Invert the `i`th face
    fn invert_face(&mut self, i: usize);

    /// Parallel iterator over the faces
    fn par_faces(
        &self,
    ) -> impl IndexedParallelIterator<Item = <Self::C as Simplex>::FACE> + Clone + '_;

    /// Sequential itertor over the faces
    fn faces(&self) -> impl ExactSizeIterator<Item = <Self::C as Simplex>::FACE> + Clone + '_;

    /// Add faces to the mesh
    fn add_faces(
        &mut self,
        faces: impl ExactSizeIterator<Item = <Self::C as Simplex>::FACE>,
        ftags: impl ExactSizeIterator<Item = Tag>,
    );

    /// Clear all the mesh faces
    fn clear_faces(&mut self);

    /// Add faces to the mesh
    fn add_faces_and_tags(
        &mut self,
        faces_and_tags: impl ExactSizeIterator<Item = (<Self::C as Simplex>::FACE, Tag)>,
    );

    /// Get the tag of the `i`th face
    fn ftag(&self, i: usize) -> Tag;

    /// Parallel iterator over the mesh faces
    fn par_ftags(&self) -> impl IndexedParallelIterator<Item = Tag> + Clone + '_;

    /// Sequential iterator over the mesh faces
    fn ftags(&self) -> impl ExactSizeIterator<Item = Tag> + Clone + '_;

    /// Get the vertices of face `f`
    fn gface(
        &self,
        f: &<Self::C as Simplex>::FACE,
    ) -> <<Self::C as Simplex>::FACE as Simplex>::GEOM<D> {
        <<Self::C as Simplex>::FACE as Simplex>::GEOM::from_iter(
            f.into_iter().map(|i| self.vert(i)),
        )
    }

    /// Parallel iterator over face vertices
    fn par_gfaces(
        &self,
    ) -> impl IndexedParallelIterator<Item = <<Self::C as Simplex>::FACE as Simplex>::GEOM<D>> + Clone + '_
    {
        self.par_faces().map(|f| self.gface(&f))
    }

    /// Sequential iterator over face vertices
    fn gfaces(
        &self,
    ) -> impl ExactSizeIterator<Item = <<Self::C as Simplex>::FACE as Simplex>::GEOM<D>> + Clone + '_
    {
        self.faces().map(|f| self.gface(&f))
    }

    /// Compute the mesh edges
    /// a map from sorted edge `[i0, i1]` (`i0 < i1`) to the edge index is returned
    fn edges(&self) -> FxHashMap<Edge<<Self::C as Simplex>::T>, usize> {
        let mut res = FxHashMap::with_hasher(FxBuildHasher);

        for e in self.elems() {
            for edg in e.edges() {
                let edg = Edge::<<Self::C as Simplex>::T>::from_iter(edg.sorted());
                if !res.contains_key(&edg) {
                    res.insert(edg, res.len());
                }
            }
        }

        res
    }

    /// Compute the vertex-to-element connectivity
    fn vertex_to_elems(&self) -> CSRGraph {
        CSRGraph::transpose(self.elems(), Some(self.n_verts()))
    }

    /// Compute the vertex-to-vertex connectivity
    fn vertex_to_vertices(&self) -> CSRGraph {
        let edges = self.edges();
        CSRGraph::from_edges(
            edges.keys().map(|e| [e.get(0), e.get(1)]),
            Some(self.n_verts()),
        )
    }

    /// Check if faces can be oriented for the current values of D and C
    #[must_use]
    fn faces_are_oriented() -> bool {
        <Self::C as Simplex>::DIM == D
    }

    /// Compute all the mesh faces (boundary & internal)
    /// a map from sorted face `[i0, i1, ...]` (`i0 < i1 < ...`) to the face index and face elements
    /// (`e0` and `e1`) is returned.
    /// If the faces can be oriented, it is oriented outwards for `e0` and inwards for `e1`
    /// If the faces only belongs to one element, `i1 = usize::MAX`
    fn all_faces(&self) -> FxHashMap<<Self::C as Simplex>::FACE, [usize; 3]> {
        let mut res: FxHashMap<<Self::C as Simplex>::FACE, [usize; 3]> =
            FxHashMap::with_hasher(FxBuildHasher);
        let mut idx = 0;

        for (i_elem, e) in self.elems().enumerate() {
            for f in e.faces() {
                let tmp = f.sorted();
                if <Self::C as Simplex>::FACE::N_VERTS > 1 {
                    let i = if f.is_same(&tmp) { 1 } else { 2 };
                    match res.entry(tmp) {
                        std::collections::hash_map::Entry::Occupied(occupied_entry) => {
                            let arr = occupied_entry.into_mut();
                            assert_eq!(
                                arr[i],
                                usize::MAX,
                                "face {} ({:?}) belongs to {} ({:?}) and {} ({:?}) with orientation {i}",
                                arr[0],
                                tmp,
                                arr[i],
                                self.elem(arr[i]),
                                i_elem,
                                e
                            );
                            arr[i] = i_elem;
                        }
                        std::collections::hash_map::Entry::Vacant(vacant_entry) => {
                            let mut arr = [idx, usize::MAX, usize::MAX];
                            arr[i] = i_elem;
                            idx += 1;
                            vacant_entry.insert(arr);
                        }
                    }
                } else {
                    match res.entry(tmp) {
                        std::collections::hash_map::Entry::Occupied(occupied_entry) => {
                            let arr = occupied_entry.into_mut();
                            if arr[1] == usize::MAX {
                                arr[1] = i_elem;
                            } else {
                                assert_eq!(arr[2], usize::MAX);
                                arr[2] = i_elem;
                            }
                        }
                        std::collections::hash_map::Entry::Vacant(vacant_entry) => {
                            let arr = [idx, i_elem, usize::MAX];
                            idx += 1;
                            vacant_entry.insert(arr);
                        }
                    }
                }
            }
        }
        res
    }

    /// Compute element pairs corresponding to all the internal faces (for partitioning)
    fn element_pairs(&self, faces: &FxHashMap<<Self::C as Simplex>::FACE, [usize; 3]>) -> CSRGraph {
        let e2e = faces
            .iter()
            .map(|(_, &[_, i0, i1])| [i0, i1])
            .filter(|&[i0, i1]| i0 != usize::MAX && i1 != usize::MAX)
            .collect::<Vec<_>>();

        CSRGraph::from_edges(e2e.iter().copied(), Some(self.n_elems()))
    }

    /// Fixes the mesh topology and orientation.
    ///
    /// This adds missing boundary and internal faces, corrects orientation,
    /// and validates the result.
    fn fix(&mut self) -> Result<FixedTags> {
        let n = self.fix_elems_orientation();
        assert_eq!(n, 0);
        let all_faces = self.all_faces();
        let btags = self.tag_boundary_faces(&all_faces);
        let itags = self.tag_internal_faces(&all_faces);
        self.fix_faces_orientation(&all_faces);
        self.check(&all_faces)?;
        Ok((btags, itags))
    }

    /// Fix the orientation of elements (so that their volume is >0), return the number of
    /// elements fixed
    fn fix_elems_orientation(&mut self) -> usize {
        let flg = self
            .elems()
            .map(|e| self.gelem(&e).vol() < 0.0)
            .collect::<Vec<_>>();
        let n = flg
            .iter()
            .enumerate()
            .filter(|(_, f)| **f)
            .map(|(i, _)| self.invert_elem(i))
            .count();
        debug!("{n} elems reoriented");
        n
    }

    /// Fix the orientation
    /// - of boundary faces to be oriented outwards (if possible)
    /// - of internal faces to be oriented from the lower to the higher element tag
    fn fix_faces_orientation(
        &mut self,
        all_faces: &FxHashMap<<Self::C as Simplex>::FACE, [usize; 3]>,
    ) -> usize {
        let flg = self
            .faces()
            .map(|f| {
                let [_, i0, i1] = all_faces.get(&f.sorted()).unwrap();
                if *i0 == usize::MAX || *i1 == usize::MAX {
                    let i = if *i1 == usize::MAX { *i0 } else { *i1 };
                    let e = self.elem(i);
                    if e.faces().all(|f2| !f.is_same(&f2)) {
                        return true;
                    }
                }
                false
            })
            .collect::<Vec<_>>();

        let n = flg
            .iter()
            .enumerate()
            .filter(|(_, f)| **f)
            .map(|(i, _)| self.invert_face(i))
            .count();
        debug!("{n} faces reoriented");
        n
    }

    fn find_tag(tags: &mut FxHashSet<Tag>) -> Tag {
        let mut next = 1;
        while tags.contains(&next) || tags.contains(&(-next)) {
            next += 1;
        }
        tags.insert(next);

        next
    }

    /// Compute the faces that are connected to only one element and that are not already tagged
    fn tag_boundary_faces(
        &mut self,
        all_faces: &FxHashMap<<Self::C as Simplex>::FACE, [usize; 3]>,
    ) -> FxHashMap<Tag, Tag> {
        let mut res = FxHashMap::with_hasher(FxBuildHasher);

        let tagged_faces = self
            .par_faces()
            .zip(self.par_ftags())
            .map(|(f, t)| (f.sorted(), t))
            .collect::<FxHashMap<_, _>>();

        let mut tags = self.ftags().collect();

        // add untagged boundary faces
        for (f, &[_, i0, i1]) in all_faces {
            if (i0 == usize::MAX || i1 == usize::MAX) && !tagged_faces.contains_key(f) {
                let i = if i1 == usize::MAX { i0 } else { i1 };
                let e = self.elem(i);
                let mut f = *f;
                let mut ok = false;
                for f2 in e.faces() {
                    if f2.sorted().is_same(&f) {
                        f = f2;
                        ok = true;
                        break;
                    }
                }
                assert!(ok);
                let etag = self.etag(i);
                if let Some(&tmp) = res.get(&etag) {
                    self.add_faces(std::iter::once(f), std::iter::once(tmp));
                } else {
                    let tag = Self::find_tag(&mut tags);
                    res.insert(etag, tag);
                    self.add_faces(std::iter::once(f), std::iter::once(tag));
                }
            }
        }

        res
    }

    /// Compute the faces that are connected to elements with different tags and that are not already tagged
    fn tag_internal_faces(
        &mut self,
        all_faces: &FxHashMap<<Self::C as Simplex>::FACE, [usize; 3]>,
    ) -> FxHashMap<[Tag; 2], Tag> {
        let mut res = FxHashMap::with_hasher(FxBuildHasher);

        let tagged_faces = self
            .par_faces()
            .zip(self.par_ftags())
            .map(|(f, t)| (f.sorted(), t))
            .collect::<FxHashMap<_, _>>();

        let mut used_tags = self.ftags().collect();

        // check tagged internal faces
        for (f, &[_, i0, i1]) in all_faces {
            if i0 != usize::MAX && i1 != usize::MAX {
                let t0 = self.etag(i0);
                let t1 = self.etag(i1);
                if t0 != t1
                    && let Some(tag) = tagged_faces.get(f)
                {
                    let tags = if t0 < t1 { [t0, t1] } else { [t1, t0] };
                    if let Some(tmp) = res.get(&tags) {
                        assert_eq!(tag, tmp);
                    }
                }
            }
        }

        // add untagged internal faces
        for (f, &[_, i0, i1]) in all_faces {
            if i0 != usize::MAX && i1 != usize::MAX {
                let t0 = self.etag(i0);
                let t1 = self.etag(i1);
                if t0 != t1 && !tagged_faces.contains_key(f) {
                    let tags = if t0 < t1 { [t0, t1] } else { [t1, t0] };
                    let i = if t0 < t1 { i0 } else { i1 };
                    let e = self.elem(i);
                    let mut f = *f;
                    let mut ok = false;
                    for f2 in e.faces() {
                        if f2.sorted().is_same(&f) {
                            f = f2;
                            ok = true;
                            break;
                        }
                    }
                    assert!(ok);

                    if let Some(&tmp) = res.get(&tags) {
                        self.add_faces(std::iter::once(f), std::iter::once(tmp));
                    } else {
                        let tag = Self::find_tag(&mut used_tags);
                        res.insert(tags, tag);
                        self.add_faces(std::iter::once(f), std::iter::once(tag));
                    }
                }
            }
        }

        res
    }

    /// Check the mesh validity
    ///   - connectivity and tag sizes
    ///   - element to vertex connectivities
    ///   - element orientations
    ///   - boundary faces and faces connecting elements with different tags are present
    fn check(&self, all_faces: &FxHashMap<<Self::C as Simplex>::FACE, [usize; 3]>) -> Result<()> {
        // lengths
        if self.par_elems().len() != self.par_etags().len() {
            return Err(Error::from("Inconsistent sizes (elems)"));
        }
        if self.par_faces().len() != self.par_ftags().len() {
            return Err(Error::from("Inconsistent sizes (faces)"));
        }

        // indices & element volume
        for e in self.elems() {
            if !e.into_iter().all(|i| i < self.n_verts()) {
                return Err(Error::from("Invalid index in elems"));
            }
            let ge = self.gelem(&e);
            if ge.vol() < 0.0 {
                return Err(Error::from("Elem has a <0 volume"));
            }
        }
        for f in self.faces() {
            if !f.into_iter().all(|i| i < self.n_verts()) {
                return Err(Error::from("Invalid index in faces"));
            }
            if !all_faces.contains_key(&f.sorted()) {
                return Err(Error::from("Face belong to no element"));
            }
        }

        // tagged faces
        let tagged_faces = self
            .par_faces()
            .map(|f| f.sorted())
            .collect::<FxHashSet<_>>();

        for (f, [_, i0, i1]) in all_faces {
            if *i0 == usize::MAX || *i1 == usize::MAX {
                if !tagged_faces.contains(f) {
                    return Err(Error::from(&format!("Boundary face {f:?} not tagged")));
                }
            } else {
                let t0 = self.etag(*i0);
                let t1 = self.etag(*i1);
                if t0 != t1 && !tagged_faces.contains(f) {
                    return Err(Error::from(&format!(
                        "Internal boundary face {f:?} not tagged ({t0} / {t1})"
                    )));
                }
            }
        }

        for (f, t) in self.faces().zip(self.ftags()) {
            let gf = self.gface(&f);
            let fc = gf.center();
            let tmp = f.sorted();
            let [_, i0, i1] = all_faces.get(&tmp).unwrap();
            if *i0 != usize::MAX && *i1 != usize::MAX && self.etag(*i0) == self.etag(*i1) {
                return Err(Error::from(&format!(
                    "Tagged face inside the domain: center = {fc:?}",
                )));
            } else if *i0 == usize::MAX || *i1 == usize::MAX {
                let i = if *i1 == usize::MAX { *i0 } else { *i1 };
                let ge = self.gelem(&self.elem(i));
                let ec = ge.center();
                if Self::faces_are_oriented() {
                    let n = gf.normal(None);
                    if n.dot(&(fc - ec)) < 0.0 {
                        return Err(Error::from(&format!(
                            "Invalid face orientation: center = {fc:?}, normal = {n:?}, face = {f:?}, tag = {t}"
                        )));
                    }
                }
            }
        }

        // volumes
        if Self::faces_are_oriented() {
            let vol = self.par_gelems().map(|ge| ge.vol()).sum::<f64>();
            let vol2 = self
                .par_faces()
                .filter(|f| {
                    let f = f.sorted();
                    let [_, i0, i1] = all_faces.get(&f).unwrap();
                    *i0 == usize::MAX || *i1 == usize::MAX
                })
                .map(|f| {
                    let gf = self.gface(&f);
                    gf.center().dot(&gf.normal(None))
                })
                .sum::<f64>()
                / D as f64;
            if (vol - vol2).abs() > 1e-10 * vol {
                return Err(Error::from(&format!(
                    "Invalid volume : {vol} from elements, {vol2} from boundary faces"
                )));
            }
        }
        Ok(())
    }

    fn smooth(&self, method: GradientMethod, f: &[f64]) -> Vec<f64> {
        match method {
            GradientMethod::LinearLeastSquares(weight) => {
                least_squares::smooth(self, &self.vertex_to_vertices(), 1, weight, f)
            }
            GradientMethod::QuadraticLeastSquares(weight) => {
                least_squares::smooth(self, &self.vertex_to_vertices(), 2, weight, f)
            }
            GradientMethod::L2Projection => {
                unreachable!("Cannot use L2Proj for smoothing")
            }
        }
    }

    fn gradient(&self, method: GradientMethod, f: &[f64]) -> Vec<f64> {
        match method {
            GradientMethod::LinearLeastSquares(weight) => {
                least_squares::gradient(self, &self.vertex_to_vertices(), 1, weight, f).unwrap()
            }
            GradientMethod::QuadraticLeastSquares(weight) => {
                least_squares::gradient(self, &self.vertex_to_vertices(), 2, weight, f).unwrap()
            }
            GradientMethod::L2Projection => {
                l2proj::gradient_l2proj(self, &self.vertex_to_elems(), f)
            }
        }
    }

    fn hessian(&self, method: GradientMethod, f: &[f64]) -> Vec<f64> {
        match method {
            GradientMethod::LinearLeastSquares(_) => {
                unreachable!("Cannot use LinearLeastSquares to compute the hessian")
            }
            GradientMethod::QuadraticLeastSquares(weight) => {
                least_squares::hessian(self, &self.vertex_to_vertices(), weight, f).unwrap()
            }
            GradientMethod::L2Projection => {
                let v2e = self.vertex_to_elems();
                let grad = l2proj::gradient_l2proj(self, &v2e, f);
                l2proj::hessian_l2proj(self, &v2e, &grad)
            }
        }
    }

    /// Integrate `g(f)` over the mesh, where `f` is a field defined on the mesh vertices
    fn integrate<G: Fn(f64) -> f64 + Send + Sync>(&self, f: &[f64], op: G) -> f64 {
        self.par_elems()
            .map(|e| {
                let func = |x: &<<Self::C as Simplex>::GEOM<D> as GSimplex<D>>::BCOORDS| {
                    op(x.into_iter().zip(e).map(|(b, i)| b * f[i]).sum::<f64>())
                };
                self.gelem(&e).integrate(func)
            })
            .sum::<f64>()
    }

    /// Compute the norm of a field `f` defined on the mesh vertices
    fn norm(&self, f: &[f64]) -> f64 {
        self.integrate(f, |x| x.powi(2)).sqrt()
    }

    /// Reorder the mesh vertices
    #[must_use]
    fn reorder_vertices(&self, vert_indices: &[usize]) -> Self {
        assert_eq!(vert_indices.len(), self.n_verts());

        let mut new_vert_indices = vec![0; self.n_verts()];
        vert_indices
            .iter()
            .enumerate()
            .for_each(|(i, &new_i)| new_vert_indices[new_i] = i);
        let new_verts = vert_indices.iter().map(|&i| self.vert(i));
        let new_elems = self
            .elems()
            .map(|e| <Self::C as Simplex>::from_iter(e.into_iter().map(|i| new_vert_indices[i])));
        let new_faces = self.faces().map(|f| {
            <Self::C as Simplex>::FACE::from_iter(f.into_iter().map(|i| new_vert_indices[i]))
        });

        let mut res = Self::empty();
        res.add_verts(new_verts);
        res.add_elems(new_elems, self.etags());
        res.add_faces(new_faces, self.ftags());
        res
    }

    /// Reorder the mesh elements (in place)
    fn reorder_elems(&mut self, elem_indices: &[usize]) {
        assert_eq!(elem_indices.len(), self.n_elems());

        let new_elems = elem_indices
            .iter()
            .map(|&i| self.elem(i))
            .collect::<Vec<_>>();
        let new_etags = elem_indices
            .iter()
            .map(|&i| self.etag(i))
            .collect::<Vec<_>>();
        self.clear_elems();
        self.add_elems(new_elems.iter().copied(), new_etags.iter().copied());
    }

    /// Reorder the mesh faces (in place)
    fn reorder_faces(&mut self, face_indices: &[usize]) {
        assert_eq!(face_indices.len(), self.n_faces());

        let new_faces = face_indices
            .iter()
            .map(|&i| self.face(i))
            .collect::<Vec<_>>();
        let new_ftags = face_indices
            .iter()
            .map(|&i| self.ftag(i))
            .collect::<Vec<_>>();
        self.clear_faces();
        self.add_faces(new_faces.iter().copied(), new_ftags.iter().copied());
    }

    /// Reorder the mesh (RCM):
    ///   - RCM orderting based on the vertex-to-vertex connectivity is used for the mesh vertices
    ///   - elements and faces are sorted by their minimum vertex index
    fn reorder_rcm(&self) -> (Self, Vec<usize>, Vec<usize>, Vec<usize>) {
        let graph = self.vertex_to_vertices();
        let vert_ids = graph.reverse_cuthill_mckee();
        let mut res = self.reorder_vertices(&vert_ids);

        let elem_ids = sort_elem_min_ids(res.elems());
        res.reorder_elems(&elem_ids);

        let face_ids = sort_elem_min_ids(res.faces());
        res.reorder_faces(&face_ids);

        (res, vert_ids, elem_ids, face_ids)
    }

    /// Set the partition as etags from an usize slice
    fn set_partition(&mut self, part: &[usize]) {
        assert_eq!(self.n_elems(), part.len());
        self.etags_mut()
            .zip(part)
            .for_each(|(x, y)| *x = *y as Tag + 1);
    }

    /// Reorder the mesh (Hilbert):
    ///   - RCM orderting based on the vertex-to-vertex connectivity is used for the mesh vertices
    ///   - elements and faces are sorted by their minimum vertex index
    fn reorder_hilbert(&self) -> (Self, Vec<usize>, Vec<usize>, Vec<usize>) {
        let vert_ids = hilbert_indices(self.verts());
        let mut res = self.reorder_vertices(&vert_ids);

        let elem_ids = sort_elem_min_ids(res.elems());
        res.reorder_elems(&elem_ids);

        let face_ids = sort_elem_min_ids(res.faces());
        res.reorder_faces(&face_ids);

        (res, vert_ids, elem_ids, face_ids)
    }

    /// Get the i-th partition
    fn get_partition(&self, i: usize) -> SubMesh<D, Self> {
        SubMesh::new(self, |t| t == i as Tag + 1)
    }

    /// Partition the mesh (RCM ordering applied to the element to element connectivity)
    fn partition<P: Partitioner>(
        &mut self,
        n_parts: usize,
        weights: Option<Vec<f64>>,
    ) -> Result<(f64, f64)> {
        let partitioner = P::new(self, n_parts, weights)?;
        let parts = partitioner.compute()?;
        assert_eq!(parts.len(), self.n_elems());

        let quality = partitioner.partition_quality(&parts);
        let imbalance = partitioner.partition_imbalance(&parts);

        self.set_partition(&parts);

        Ok((quality, imbalance))
    }

    /// Randomly shuffle vertices, elements and faces
    #[must_use]
    fn random_shuffle(&self) -> Self {
        let mut rng = StdRng::seed_from_u64(1234);

        let mut vert_ids = (0..self.n_verts()).collect::<Vec<_>>();
        vert_ids.shuffle(&mut rng);

        let mut res = self.reorder_vertices(&vert_ids);

        let mut elem_ids = (0..self.n_elems()).collect::<Vec<_>>();
        elem_ids.shuffle(&mut rng);
        res.reorder_elems(&elem_ids);

        let mut face_ids = (0..self.n_faces()).collect::<Vec<_>>();
        face_ids.shuffle(&mut rng);
        res.reorder_faces(&face_ids);

        res
    }

    /// Import a mesh from a `.meshb` file
    fn from_meshb(file_name: &str) -> Result<Self> {
        let mut res = Self::empty();
        let mut reader = MeshbReader::new(file_name)?;

        res.add_verts(
            reader
                .read_vertices::<D>()?
                .map(|(x, _)| Vertex::<D>::from_column_slice(&x)),
        );

        match <Self::C as Simplex>::order() {
            1 => {
                match <Self::C as Simplex>::N_VERTS {
                    4 => {
                        if let Ok(iter) = reader.read_tetrahedra() {
                            res.add_elems_and_tags(
                                iter.map(|(e, t)| (<Self::C as Simplex>::from_iter(e), t as Tag)),
                            );
                        }
                    }
                    3 => {
                        if let Ok(iter) = reader.read_triangles() {
                            res.add_elems_and_tags(
                                iter.map(|(e, t)| (<Self::C as Simplex>::from_iter(e), t as Tag)),
                            );
                        }
                    }
                    2 => {
                        if let Ok(iter) = reader.read_edges() {
                            res.add_elems_and_tags(
                                iter.map(|(e, t)| (<Self::C as Simplex>::from_iter(e), t as Tag)),
                            );
                        }
                    }
                    _ => unimplemented!(),
                }

                match <Self::C as Simplex>::FACE::N_VERTS {
                    3 => {
                        if let Ok(iter) = reader.read_triangles() {
                            res.add_faces_and_tags(iter.map(|(e, t)| {
                                (<Self::C as Simplex>::FACE::from_iter(e), t as Tag)
                            }));
                        }
                    }
                    2 => {
                        if let Ok(iter) = reader.read_edges() {
                            res.add_faces_and_tags(iter.map(|(e, t)| {
                                (<Self::C as Simplex>::FACE::from_iter(e), t as Tag)
                            }));
                        }
                    }
                    1 => warn!("not reading faces when elements are edges"),
                    _ => unimplemented!(),
                }
            }
            2 => {
                match <Self::C as Simplex>::N_VERTS {
                    6 => {
                        if let Ok(iter) = reader.read_quadratic_triangles() {
                            res.add_elems_and_tags(
                                iter.map(|(e, t)| (<Self::C as Simplex>::from_iter(e), t as Tag)),
                            );
                        }
                    }
                    3 => {
                        if let Ok(iter) = reader.read_quadratic_edges() {
                            res.add_elems_and_tags(
                                iter.map(|(e, t)| (<Self::C as Simplex>::from_iter(e), t as Tag)),
                            );
                        }
                    }
                    _ => unimplemented!(),
                }

                match <Self::C as Simplex>::FACE::N_VERTS {
                    3 => {
                        if let Ok(iter) = reader.read_quadratic_edges() {
                            res.add_faces_and_tags(iter.map(|(e, t)| {
                                (<Self::C as Simplex>::FACE::from_iter(e), t as Tag)
                            }));
                        }
                    }
                    1 => warn!("not reading faces when elements are edges"),
                    _ => unimplemented!(),
                }
            }

            _ => unimplemented!(),
        }

        Ok(res)
    }

    /// Export the mesh to a `.meshb` file
    #[allow(clippy::unnecessary_fallible_conversions)]
    fn write_meshb(&self, file_name: &str) -> Result<()> {
        let mut writer = MeshbWriter::new(file_name, 3, D as u8)?;

        writer.write_vertices::<D, _, _>(
            self.verts().map(|x| std::array::from_fn(|i| x[i])),
            (0..self.n_verts()).map(|_| 1),
        )?;

        match <Self::C as Simplex>::order() {
            1 => {
                match <Self::C as Simplex>::N_VERTS {
                    4 => writer.write_tetrahedra(
                        self.elems().map(|x| std::array::from_fn(|i| x.get(i))),
                        self.etags().map(|x| x.try_into().unwrap()),
                    )?,
                    3 => writer.write_triangles(
                        self.elems().map(|x| std::array::from_fn(|i| x.get(i))),
                        self.etags().map(|x| x.try_into().unwrap()),
                    )?,
                    2 => writer.write_edges(
                        self.elems().map(|x| std::array::from_fn(|i| x.get(i))),
                        self.etags().map(|x| x.try_into().unwrap()),
                    )?,
                    _ => unimplemented!(),
                }

                match <Self::C as Simplex>::FACE::N_VERTS {
                    3 => writer.write_triangles(
                        self.faces().map(|x| std::array::from_fn(|i| x.get(i))),
                        self.ftags().map(|x| x.try_into().unwrap()),
                    )?,
                    2 => writer.write_edges(
                        self.faces().map(|x| std::array::from_fn(|i| x.get(i))),
                        self.ftags().map(|x| x.try_into().unwrap()),
                    )?,
                    1 => {
                        if self.n_faces() != 0 {
                            warn!("skip faces in meshb export");
                        }
                    }
                    _ => unimplemented!(),
                }
            }
            2 => {
                match <Self::C as Simplex>::N_VERTS {
                    6 => writer.write_quadratic_triangles(
                        self.elems().map(|x| std::array::from_fn(|i| x.get(i))),
                        self.etags().map(|x| x.try_into().unwrap()),
                    )?,
                    3 => writer.write_quadratic_edges(
                        self.elems().map(|x| std::array::from_fn(|i| x.get(i))),
                        self.etags().map(|x| x.try_into().unwrap()),
                    )?,
                    _ => unimplemented!(),
                }

                match <Self::C as Simplex>::FACE::N_VERTS {
                    3 => writer.write_quadratic_edges(
                        self.faces().map(|x| std::array::from_fn(|i| x.get(i))),
                        self.ftags().map(|x| x.try_into().unwrap()),
                    )?,
                    1 => {
                        if self.n_faces() != 0 {
                            warn!("skip faces in meshb export");
                        }
                    }
                    _ => unimplemented!(),
                }
            }
            _ => unimplemented!(),
        }

        writer.close();

        Ok(())
    }

    fn write_solb_it<const N: usize, G: FnMut(&[f64]) -> [f64; N]>(
        &self,
        arr: &[f64],
        file_name: &str,
        f: G,
    ) -> Result<()> {
        assert_eq!(arr.len(), N * self.n_verts());

        let mut writer = MeshbWriter::new(file_name, 3, D as u8)?;
        writer.write_solution(arr.chunks(N).map(f))?;
        writer.close();

        Ok(())
    }

    fn write_solb(&self, arr: &[f64], file_name: &str) -> Result<()> {
        let n_comp = arr.len() / self.n_verts();
        match D {
            2 => match n_comp {
                1 => self.write_solb_it::<1, _>(arr, file_name, |x| [x[0]])?,
                2 => self.write_solb_it::<2, _>(arr, file_name, |x| [x[0], x[1]])?,
                3 => self.write_solb_it::<3, _>(arr, file_name, |x| [x[0], x[2], x[1]])?,
                _ => unreachable!(),
            },
            3 => match n_comp {
                1 => self.write_solb_it::<1, _>(arr, file_name, |x| [x[0]])?,
                3 => self.write_solb_it::<3, _>(arr, file_name, |x| [x[0], x[1], x[2]])?,
                6 => self.write_solb_it::<6, _>(arr, file_name, |x| {
                    [x[0], x[3], x[1], x[5], x[4], x[2]]
                })?,
                _ => unreachable!(),
            },
            _ => unreachable!(),
        }

        Ok(())
    }

    fn read_solb_it<const N: usize, G: FnMut([f64; N]) -> [f64; N]>(
        mut reader: MeshbReader,
        f: G,
    ) -> Result<Vec<f64>> {
        let sol = reader.read_solution::<N>()?;
        Ok(sol.flat_map(f).collect())
    }

    fn read_solb(file_name: &str) -> Result<(Vec<f64>, usize)> {
        let mut reader = MeshbReader::new(file_name)?;
        let d = reader.dimension();
        assert_eq!(d, D as u8);
        let m = reader.get_solution_size()?;

        let res = match d {
            2 => match m {
                1 => Self::read_solb_it::<1, _>(reader, |x| [x[0]])?,
                2 => Self::read_solb_it::<2, _>(reader, |x| [x[0], x[1]])?,
                3 => Self::read_solb_it::<3, _>(reader, |x| [x[0], x[2], x[1]])?,
                _ => unreachable!(),
            },
            3 => match m {
                1 => Self::read_solb_it::<1, _>(reader, |x| [x[0]])?,
                3 => Self::read_solb_it::<3, _>(reader, |x| [x[0], x[1], x[2]])?,
                6 => Self::read_solb_it::<6, _>(reader, |x| [x[0], x[2], x[5], x[1], x[4], x[3]])?,
                _ => unreachable!(),
            },
            _ => unreachable!(),
        };

        Ok((res, m))
    }

    /// Export the mesh to a `.vtu` file
    fn write_vtk(&self, file_name: &str) -> Result<()> {
        let vtu = VTUFile::from_mesh(self, VTUEncoding::Binary);

        vtu.export(file_name)?;

        Ok(())
    }

    /// Build a `Mesh<D, C::FACE>` mesh containing faces such that `filter(tag)` is true
    /// Only the required vertices are present
    /// The following must be true: C2 = C - 1 and F2 = F - 1 (rust stable limitation)
    fn extract_faces<M: Mesh<D, C = <Self::C as Simplex>::FACE>, G: Fn(Tag) -> bool>(
        &self,
        filter: G,
    ) -> (M, Vec<usize>) {
        let mut new_ids = vec![usize::MAX; self.n_verts()];
        let mut vert_ids = Vec::new();
        let mut next = 0;

        let n_faces = self
            .faces()
            .zip(self.ftags())
            .filter(|(_, t)| filter(*t))
            .map(|(f, _)| {
                for i in f {
                    if new_ids[i] == usize::MAX {
                        new_ids[i] = next;
                        vert_ids.push(i);
                        next += 1;
                    }
                }
            })
            .count();
        let n_verts = next;

        let mut verts = vec![Vertex::<D>::zeros(); n_verts];
        let mut faces = Vec::with_capacity(n_faces);
        let mut ftags = Vec::with_capacity(n_faces);

        new_ids
            .iter()
            .enumerate()
            .filter(|&(_, j)| *j != usize::MAX)
            .for_each(|(i, &j)| verts[j] = self.vert(i));
        self.faces()
            .zip(self.ftags())
            .filter(|(f, _)| f.into_iter().all(|i| new_ids[i] != usize::MAX))
            .for_each(|(f, t)| {
                faces.push(<Self::C as Simplex>::FACE::from_iter(
                    f.into_iter().map(|i| new_ids[i]),
                ));
                ftags.push(t);
            });

        let mut res = M::empty();
        res.add_verts(verts.into_iter());
        res.add_elems(faces.into_iter(), ftags.iter().copied());

        (res, vert_ids)
    }

    /// Build a `Mesh<D, C2, F2>` mesh containing the boundary faces
    /// The following must be true: C2 = C - 1 and F2 = F - 1 (rust stable limitation)
    fn boundary<M: Mesh<D, C = <Self::C as Simplex>::FACE>>(&self) -> (M, Vec<usize>) {
        self.extract_faces(|_| true)
    }

    /// Split quandrangles and add them to the mesh (C == 3 or F == 3)
    fn add_quadrangles<
        I1: ExactSizeIterator<Item = Quadrangle<<Self::C as Simplex>::T>>,
        I2: ExactSizeIterator<Item = Tag>,
    >(
        &mut self,
        quads: I1,
        tags: I2,
    ) {
        if <Self::C as Simplex>::N_VERTS == 3 {
            let mut tmp = Vec::with_capacity(2 * quads.len());
            quads.zip(tags).for_each(|(q, t)| {
                let tris = qua2tris(&q);
                for tri in tris {
                    tmp.push((<Self::C as Simplex>::from_iter(tri), t));
                }
            });
            self.add_elems_and_tags(tmp.iter().copied());
        } else if <Self::C as Simplex>::FACE::N_VERTS == 3 {
            let mut tmp = Vec::with_capacity(2 * quads.len());
            quads.zip(tags).for_each(|(q, t)| {
                let tris = qua2tris(&q);
                for tri in tris {
                    tmp.push((<<Self::C as Simplex>::FACE as Simplex>::from_iter(tri), t));
                }
            });
            self.add_faces_and_tags(tmp.iter().copied());
        } else {
            unreachable!()
        }
    }

    /// Split hexahedra and add them to the mesh (C == 4)
    fn add_hexahedra<
        I1: ExactSizeIterator<Item = Hexahedron<<Self::C as Simplex>::T>>,
        I2: ExactSizeIterator<Item = Tag>,
    >(
        &mut self,
        hexs: I1,
        tags: I2,
    ) -> Vec<usize> {
        if <Self::C as Simplex>::N_VERTS == 4 {
            let mut tmp = Vec::with_capacity(6 * hexs.len());
            let mut ids = Vec::with_capacity(6 * hexs.len());
            hexs.zip(tags).enumerate().for_each(|(i, (q, t))| {
                let (tets, last_tet) = hex2tets(&q);
                for tet in tets {
                    tmp.push((<Self::C as Simplex>::from_iter(tet), t));
                    ids.push(i);
                }
                if let Some(last_tet) = last_tet {
                    tmp.push((<Self::C as Simplex>::from_iter(last_tet), t));
                    ids.push(i);
                }
            });
            self.add_elems_and_tags(tmp.iter().copied());
            ids
        } else {
            unreachable!()
        }
    }

    /// Split prisms and add them to the mesh (C == 4)
    fn add_prisms<
        I1: ExactSizeIterator<Item = Prism<<Self::C as Simplex>::T>>,
        I2: ExactSizeIterator<Item = Tag>,
    >(
        &mut self,
        pris: I1,
        tags: I2,
    ) {
        if <Self::C as Simplex>::N_VERTS == 4 {
            let mut tmp = Vec::with_capacity(3 * pris.len());
            pris.zip(tags).for_each(|(q, t)| {
                let tets = pri2tets(&q);
                for tet in tets {
                    tmp.push((<Self::C as Simplex>::from_iter(tet), t));
                }
            });
            self.add_elems_and_tags(tmp.iter().copied());
        } else {
            unreachable!()
        }
    }

    /// Split pyramids and add them to the mesh (C == 4)
    fn add_pyramids<
        I1: ExactSizeIterator<Item = Pyramid<<Self::C as Simplex>::T>>,
        I2: ExactSizeIterator<Item = Tag>,
    >(
        &mut self,
        pyrs: I1,
        tags: I2,
    ) {
        if <Self::C as Simplex>::N_VERTS == 4 {
            let mut tmp = Vec::with_capacity(3 * pyrs.len());
            pyrs.zip(tags).for_each(|(q, t)| {
                let tets = pyr2tets(&q);
                for tet in tets {
                    tmp.push((<Self::C as Simplex>::from_iter(tet), t));
                }
            });
            self.add_elems_and_tags(tmp.iter().copied());
        } else {
            unreachable!()
        }
    }

    /// Check that two meshes are equal
    ///   - same vertex coordinates (with tolerance `tol`)
    ///   - same connectivities and tags
    fn check_equals<M: Mesh<D, C = Self::C>>(&self, other: &M, tol: f64) -> Result<()> {
        for (i, (v0, v1)) in self.verts().zip(other.verts()).enumerate() {
            if (v0 - v1).norm() > tol {
                return Err(Error::from(&format!("Vertex {i}: {v0:?} != {v1:?}")));
            }
        }

        for (i, (e0, e1)) in self.elems().zip(other.elems()).enumerate() {
            if e0 != e1 {
                return Err(Error::from(&format!("Element {i}: {e0:?} != {e1:?}")));
            }
        }

        for (i, (t0, t1)) in self.etags().zip(other.etags()).enumerate() {
            if t0 != t1 {
                return Err(Error::from(&format!("Element tag {i}: {t0:?} != {t1:?}")));
            }
        }

        for (i, (e0, e1)) in self.faces().zip(other.faces()).enumerate() {
            if e0 != e1 {
                return Err(Error::from(&format!("Face {i}: {e0:?} != {e1:?}")));
            }
        }

        for (i, (t0, t1)) in self.ftags().zip(other.ftags()).enumerate() {
            if t0 != t1 {
                return Err(Error::from(&format!("Face tag {i}: {t0:?} != {t1:?}")));
            }
        }

        Ok(())
    }

    /// Split a mesh uniformly
    #[must_use]
    fn split(&self) -> Self {
        let mut res = Self::empty();

        let mut edges = self.edges();

        // Vertices
        res.add_verts(self.verts());
        let mut verts = vec![Vertex::<D>::zeros(); edges.len()];
        for (edg, &i) in &edges {
            let p0 = self.vert(edg.get(0));
            let p1 = self.vert(edg.get(1));
            verts[i] = 0.5 * (p0 + p1);
        }
        res.add_verts(verts.iter().copied());

        // add offset to verts
        for v in edges.values_mut() {
            *v += self.n_verts();
        }

        // Cells
        match <Self::C as Simplex>::N_VERTS {
            4 => {
                let (elems, etags) = split_tets(self.elems().zip(self.etags()), &edges);
                res.add_elems(elems.iter().copied(), etags.iter().copied());
            }
            3 => {
                let (elems, etags) = split_tris(self.elems().zip(self.etags()), &edges);
                res.add_elems(elems.iter().copied(), etags.iter().copied());
            }
            2 => {
                let (elems, etags) = split_edgs(self.elems().zip(self.etags()), &edges);
                res.add_elems(elems.iter().copied(), etags.iter().copied());
            }
            _ => unreachable!(),
        }

        // Faces
        match <Self::C as Simplex>::FACE::N_VERTS {
            3 => {
                let (faces, ftags) = split_tris(self.faces().zip(self.ftags()), &edges);
                res.add_faces(faces.iter().copied(), ftags.iter().copied());
            }
            2 => {
                let (faces, ftags) = split_edgs(self.faces().zip(self.ftags()), &edges);
                res.add_faces(faces.iter().copied(), ftags.iter().copied());
            }
            1 => {
                res.add_faces(self.faces(), self.ftags());
            }
            _ => unreachable!(),
        }
        res
    }

    /// Compute the skewness for all internal faces in the mesh
    /// Skewness is the normalized distance between a line that connects two
    /// adjacent cell centroids and the distance from that line to the shared
    /// face’s center.
    fn face_skewnesses(
        &self,
        all_faces: &FxHashMap<<Self::C as Simplex>::FACE, [usize; 3]>,
    ) -> impl Iterator<Item = (usize, usize, f64)> {
        all_faces
            .iter()
            .filter(|&(_, [_, i0, i1])| *i0 != usize::MAX && *i1 != usize::MAX)
            .map(|(f, &[_, i0, i1])| {
                let fc = self.gface(f).center();
                let ec0 = self.gelem(&self.elem(i0)).center();
                let ec1 = self.gelem(&self.elem(i1)).center();
                let e2f = fc - ec0;
                let l_e2f = e2f.norm();
                let e2e = ec1 - ec0;
                let l_e2e = e2e.norm();
                let tmp = e2e.dot(&e2f) / (l_e2e * l_e2f);
                let ang = f64::acos(tmp.clamp(-1., 1.));
                let s = l_e2f * ang.sin();
                (i0, i1, s / l_e2e)
            })
    }

    /// Compute the edge ratio for all the elements in the mesh
    #[must_use]
    fn edge_length_ratios(&self) -> impl ExactSizeIterator<Item = f64> + '_ {
        self.elems().map(move |e| {
            let mut l_min = f64::MAX;
            let mut l_max = 0.0_f64;
            for edg in e.edges() {
                let l = (self.vert(edg.get(0)) - self.vert(edg.get(1))).norm();
                l_min = l_min.min(l);
                l_max = l_max.max(l);
            }
            l_max / l_min
        })
    }

    /// Compute the ratio of inscribed radius to circumradius
    /// (normalized to be between 0 and 1) for all the elements in the mesh
    #[must_use]
    fn elem_gammas(&self) -> impl ExactSizeIterator<Item = f64> + '_ {
        self.gelems().map(|ge| ge.gamma())
    }

    /// Get the bounding box
    #[must_use]
    fn bounding_box(&self) -> (Vertex<D>, Vertex<D>) {
        let mut mini = self.verts().next().unwrap();
        let mut maxi = mini;
        for p in self.verts() {
            for j in 0..D {
                mini[j] = f64::min(mini[j], p[j]);
                maxi[j] = f64::max(maxi[j], p[j]);
            }
        }
        (mini, maxi)
    }

    /// Get the number of faces with a given tag
    #[must_use]
    fn n_tagged_faces(&self, tag: Tag) -> usize {
        self.ftags().filter(|&t| t == tag).count()
    }

    /// Return a bool vector that indicates wether a vertex in on a face
    #[must_use]
    fn boundary_flag(&self) -> Vec<bool> {
        let mut res = vec![false; self.n_verts()];
        self.faces().flatten().for_each(|i| res[i] = true);
        res
    }

    /// Add vertices, elements and faces from another mesh according to their tag
    ///   - only the elements with a tag t such that `element_filter(t)` is true are inserted
    ///   - among the faces belonging to these elements, only those with a tag such that `face_filter` is true are inserted
    ///   - if `merge_tol` is not None, vertices on the boundaries of `self` and `other` are merged if closer than the tolerance.
    ///
    /// NB: Some boundary faces in `self` or `other` may no longer be boundary faces in the result
    fn add<F1, F2>(
        &mut self,
        other: &Self,
        mut elem_filter: F1,
        mut face_filter: F2,
        merge_tol: Option<f64>,
    ) -> (Vec<usize>, Vec<usize>, Vec<usize>)
    where
        F1: FnMut(Tag) -> bool,
        F2: FnMut(Tag) -> bool,
    {
        let n_verts = self.n_verts();
        let n_verts_other = other.n_verts();
        let mut new_vert_ids = vec![usize::MAX; n_verts_other];

        for (e, t) in other.elems().zip(other.etags()) {
            if elem_filter(t) {
                for i in e {
                    new_vert_ids[i] = usize::MAX - 1;
                }
            }
        }

        // If needed, merge boundary vertices
        if let Some(merge_tol) = merge_tol {
            let other_flg = other.boundary_flag();
            let n = other_flg.iter().filter(|&&x| x).count();
            if n > 0 {
                let mut overts = Vec::with_capacity(n);
                let mut oids = Vec::with_capacity(n);
                for (i, &flg) in other_flg.iter().enumerate() {
                    if flg {
                        oids.push(i);
                        overts.push(other.vert(i));
                    }
                }
                let tree = PointIndex::new(overts.iter().copied());
                for (i, &flg) in self.boundary_flag().iter().enumerate() {
                    if flg {
                        let vx = self.vert(i);
                        let (i_other, _) = tree.nearest_vert(&vx);
                        let i_other = oids[i_other];
                        if (vx - other.vert(i_other)).norm() < merge_tol {
                            new_vert_ids[i_other] = i;
                        }
                    }
                }
            }
        }

        // number & add the new vertices
        let mut next = n_verts;
        let mut added_verts = Vec::new();
        new_vert_ids.iter_mut().enumerate().for_each(|(i, x)| {
            if *x == usize::MAX - 1 {
                added_verts.push(i);
                *x = next;
                next += 1;
                self.add_verts(std::iter::once(other.vert(i)));
            }
        });

        let mut added_elems = Vec::new();

        // keep track of the possible new faces
        let mut all_added_faces = FxHashSet::default();
        for (i, (e, t)) in other
            .elems()
            .zip(other.etags())
            .enumerate()
            .filter(|(_, (_, t))| elem_filter(*t))
        {
            added_elems.push(i);
            assert!(e.into_iter().all(|i| new_vert_ids[i] != usize::MAX));
            let e = <Self::C as Simplex>::from_iter(e.into_iter().map(|i| new_vert_ids[i]));
            self.add_elems(std::iter::once(e), std::iter::once(t));
            for face in e.faces() {
                all_added_faces.insert(face.sorted());
            }
        }

        let mut added_faces = Vec::new();
        for (i, (f, t)) in other
            .faces()
            .zip(other.ftags())
            .enumerate()
            .filter(|(_, (_, t))| face_filter(*t))
            .filter(|&(_, (f, _))| {
                if !f.into_iter().all(|i| new_vert_ids[i] != usize::MAX) {
                    return false;
                }
                let f =
                    <Self::C as Simplex>::FACE::from_iter(f.into_iter().map(|i| new_vert_ids[i]))
                        .sorted();
                all_added_faces.contains(&f)
            })
        {
            added_faces.push(i);
            let f = <Self::C as Simplex>::FACE::from_iter(f.into_iter().map(|i| new_vert_ids[i]));
            self.add_faces(std::iter::once(f), std::iter::once(t));
        }

        (added_verts, added_elems, added_faces)
    }

    /// Remove faces based on their tag
    fn remove_faces<F1: FnMut(Tag) -> bool>(&mut self, mut face_filter: F1) {
        let mut new_faces = Vec::new();
        let mut new_ftags = Vec::new();

        for (f, t) in self
            .faces()
            .zip(self.ftags())
            .filter(|(_, t)| !face_filter(*t))
        {
            new_faces.push(f);
            new_ftags.push(t);
        }
        self.clear_faces();
        self.add_faces(new_faces.iter().copied(), new_ftags.iter().copied());
    }

    /// Total mesh volume
    fn vol(&self) -> f64 {
        self.gelems().map(|ge| ge.vol()).sum::<f64>()
    }

    /// Convert a field defined at the element centers (P0) to a field defined at the vertices (P1)
    /// using a weighted average. For metric fields, use `elem_data_to_vertex_data_metric`
    /// vertex-to-element connectivity and volumes are required
    fn elem_data_to_vertex_data(&self, v2e: &CSRGraph, v: &[f64]) -> Vec<f64> {
        debug!("Convert element data to vertex data");

        let n_elems = self.n_elems();
        let n_verts = self.n_verts();
        assert_eq!(v.len() % n_elems, 0);

        let n_comp = v.len() / n_elems;

        let mut res = vec![0.; n_comp * n_verts];

        let vols = self.par_gelems().map(|ge| ge.vol()).collect::<Vec<_>>();

        res.par_chunks_mut(n_comp)
            .enumerate()
            .for_each(|(i_vert, vals)| {
                let mut tmp = 0.0;
                for i_elem in v2e.row(i_vert).iter().copied() {
                    let w = vols[i_elem] / <Self::C as Simplex>::N_VERTS as f64;
                    tmp += w;
                    for i_comp in 0..n_comp {
                        vals[i_comp] += w * v[n_comp * i_elem + i_comp];
                    }
                }
                for v in vals.iter_mut() {
                    *v /= tmp;
                }
            });

        res
    }

    /// Convert a field defined at the vertices (P1) to a field defined at the element centers (P0)
    /// For metric fields, use `elem_data_to_vertex_data_metric`
    fn vertex_data_to_elem_data(&self, v: &[f64]) -> Vec<f64> {
        debug!("Convert vertex data to element data");
        let n_elems = self.n_elems();
        let n_verts = self.n_verts();
        assert_eq!(v.len() % n_verts, 0);

        let n_comp = v.len() / n_verts;

        let mut res = vec![0.; n_comp * n_elems];

        let f = 1. / <Self::C as Simplex>::N_VERTS as f64;
        res.par_chunks_mut(n_comp)
            .zip(self.par_elems())
            .for_each(|(vals, e)| {
                for i_comp in 0..n_comp {
                    for i_vert in e {
                        vals[i_comp] += f * v[n_comp * i_vert + i_comp];
                    }
                }
            });

        res
    }

    /// Sequential iterator over the vertices
    fn verts_mut(&mut self) -> impl ExactSizeIterator<Item = &mut Vertex<D>> + '_;

    /// Sequential iterator over the mesh elements
    fn elems_mut<'a>(&'a mut self) -> impl ExactSizeIterator<Item = &'a mut Self::C> + 'a
    where
        Self::C: 'a;

    /// Sequential iterator over the element tags
    fn etags_mut(&mut self) -> impl ExactSizeIterator<Item = &mut Tag> + '_;

    /// Sequential itertor over the faces
    fn faces_mut<'a>(
        &'a mut self,
    ) -> impl ExactSizeIterator<Item = &'a mut <Self::C as Simplex>::FACE> + 'a
    where
        Self::C: 'a;

    /// Sequential iterator over the mesh faces
    fn ftags_mut(&mut self) -> impl ExactSizeIterator<Item = &mut Tag> + '_;
}

/// Generic meshes implemented with Vecs
#[derive(Clone)]
pub struct GenericMesh<const D: usize, C: Simplex> {
    verts: Vector<Vertex<D>>,
    elems: Vector<C>,
    etags: Vector<Tag>,
    faces: Vector<C::FACE>,
    ftags: Vector<Tag>,
}

impl<const D: usize, C: Simplex> GenericMesh<D, C> {
    #[must_use]
    pub fn from_vecs(
        verts: Vec<Vertex<D>>,
        elems: Vec<C>,
        etags: Vec<Tag>,
        faces: Vec<C::FACE>,
        ftags: Vec<Tag>,
    ) -> Self {
        Self {
            verts: verts.into(),
            elems: elems.into(),
            etags: etags.into(),
            faces: faces.into(),
            ftags: ftags.into(),
        }
    }

    #[must_use]
    pub const fn new(
        verts: Vector<Vertex<D>>,
        elems: Vector<C>,
        etags: Vector<Tag>,
        faces: Vector<C::FACE>,
        ftags: Vector<Tag>,
    ) -> Self {
        Self {
            verts,
            elems,
            etags,
            faces,
            ftags,
        }
    }
}

impl<const D: usize, C: Simplex> Mesh<D> for GenericMesh<D, C> {
    type C = C;
    fn empty() -> Self {
        Self {
            verts: Vec::new().into(),
            elems: Vec::new().into(),
            etags: Vec::new().into(),
            faces: Vec::new().into(),
            ftags: Vec::new().into(),
        }
    }

    fn n_verts(&self) -> usize {
        self.verts.len()
    }

    fn vert(&self, i: usize) -> Vertex<D> {
        self.verts.index(i)
    }

    fn verts(&self) -> impl ExactSizeIterator<Item = Vertex<D>> + Clone + '_ {
        self.verts.iter()
    }

    fn par_verts(&self) -> impl IndexedParallelIterator<Item = Vertex<D>> + Clone + '_ {
        self.verts.par_iter()
    }

    fn add_verts(&mut self, v: impl ExactSizeIterator<Item = Vertex<D>>) {
        self.verts.extend(v);
    }

    fn n_elems(&self) -> usize {
        self.elems.len()
    }

    fn elem(&self, i: usize) -> C {
        self.elems.index(i)
    }

    fn invert_elem(&mut self, i: usize) {
        self.elems.index_mut(i).invert();
    }

    fn elems(&self) -> impl ExactSizeIterator<Item = C> + Clone + '_ {
        self.elems.iter()
    }

    fn par_elems(&self) -> impl IndexedParallelIterator<Item = C> + Clone + '_ {
        self.elems.par_iter()
    }

    fn etag(&self, i: usize) -> Tag {
        self.etags.index(i)
    }

    fn etags(&self) -> impl ExactSizeIterator<Item = Tag> + Clone + '_ {
        self.etags.iter()
    }

    fn par_etags(&self) -> impl IndexedParallelIterator<Item = Tag> + Clone + '_ {
        self.etags.par_iter()
    }

    fn add_elems<I1: ExactSizeIterator<Item = C>, I2: ExactSizeIterator<Item = Tag>>(
        &mut self,
        elems: I1,
        etags: I2,
    ) {
        self.elems.extend(elems);
        self.etags.extend(etags);
    }

    fn clear_elems(&mut self) {
        self.elems.clear();
        self.etags.clear();
    }

    fn add_elems_and_tags(&mut self, elems_and_tags: impl ExactSizeIterator<Item = (C, Tag)>) {
        self.elems.reserve(elems_and_tags.len());
        self.etags.reserve(elems_and_tags.len());
        for (e, t) in elems_and_tags {
            self.elems.push(e);
            self.etags.push(t);
        }
    }

    fn n_faces(&self) -> usize {
        self.faces.len()
    }

    fn face(&self, i: usize) -> C::FACE {
        self.faces.index(i)
    }

    fn invert_face(&mut self, i: usize) {
        self.faces.index_mut(i).invert();
    }

    fn faces(&self) -> impl ExactSizeIterator<Item = C::FACE> + Clone + '_ {
        self.faces.iter()
    }

    fn par_faces(&self) -> impl IndexedParallelIterator<Item = C::FACE> + Clone + '_ {
        self.faces.par_iter()
    }

    fn ftag(&self, i: usize) -> Tag {
        self.ftags.index(i)
    }

    fn ftags(&self) -> impl ExactSizeIterator<Item = Tag> + Clone + '_ {
        self.ftags.iter()
    }

    fn par_ftags(&self) -> impl IndexedParallelIterator<Item = Tag> + Clone + '_ {
        self.ftags.par_iter()
    }

    fn add_faces(
        &mut self,
        faces: impl ExactSizeIterator<Item = C::FACE>,
        ftags: impl ExactSizeIterator<Item = Tag>,
    ) {
        self.faces.extend(faces);
        self.ftags.extend(ftags);
    }

    fn clear_faces(&mut self) {
        self.faces.clear();
        self.ftags.clear();
    }

    fn add_faces_and_tags(
        &mut self,
        faces_and_tags: impl ExactSizeIterator<Item = (C::FACE, Tag)>,
    ) {
        self.faces.reserve(faces_and_tags.len());
        self.ftags.reserve(faces_and_tags.len());
        for (e, t) in faces_and_tags {
            self.faces.push(e);
            self.ftags.push(t);
        }
    }

    fn verts_mut(&mut self) -> impl ExactSizeIterator<Item = &mut Vertex<D>> + '_ {
        self.verts.iter_mut()
    }

    fn elems_mut<'a>(&'a mut self) -> impl ExactSizeIterator<Item = &'a mut C> + 'a
    where
        C: 'a,
    {
        self.elems.iter_mut()
    }

    fn etags_mut(&mut self) -> impl ExactSizeIterator<Item = &mut Tag> + '_ {
        self.etags.iter_mut()
    }

    fn faces_mut<'a>(&'a mut self) -> impl ExactSizeIterator<Item = &'a mut C::FACE> + 'a
    where
        C: 'a,
    {
        self.faces.iter_mut()
    }

    fn ftags_mut(&mut self) -> impl ExactSizeIterator<Item = &mut Tag> + '_ {
        self.ftags.iter_mut()
    }
}
