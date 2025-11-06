//! Simplex meshes in D dimensions, represented by
//!   - the vertices
//!   - elements of type `C` and element tags
//!   - faces of type `C::FACE` and element tags
//!
//! F = C-1 cannot be imposed in rust stable
mod boundary_mesh_2d;
mod boundary_mesh_3d;
mod mesh_2d;
mod mesh_3d;
mod simplices;
mod to_simplices;
pub mod twovec;

mod split;

mod hilbert;
pub mod partition;

pub mod least_squares;

use derive_more::{AsRef, From, Index, IndexMut, IntoIterator};
use log::{debug, warn};
use minimeshb::{reader::MeshbReader, writer::MeshbWriter};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};
use std::convert::Infallible;
use std::fmt::{Debug, Display};
use std::num::TryFromIntError;
use std::ops::Mul;
use std::{
    hash::Hash,
    marker::PhantomData,
    ops::{Add, AddAssign, Sub},
};

use crate::{
    Error, Result, Tag, Vertex,
    graph::CSRGraph,
    io::{VTUEncoding, VTUFile},
    spatialindex::PointIndex,
};
pub use boundary_mesh_2d::BoundaryMesh2d;
pub use boundary_mesh_3d::{BoundaryMesh3d, read_stl};
use hilbert::hilbert_indices;
use least_squares::LeastSquaresGradient;
pub use mesh_2d::{Mesh2d, nonuniform_rectangle_mesh, rectangle_mesh};
pub use mesh_3d::{Mesh3d, box_mesh, nonuniform_box_mesh};
use partition::Partitioner;
pub use simplices::{GSimplex, Simplex, get_face_to_elem};
use split::{split_edgs, split_tets, split_tris};
pub use to_simplices::{hex2tets, pri2tets, pyr2tets, qua2tris};

pub trait Idx:
    TryInto<usize, Error = Self::ConvertError>
    + TryFrom<usize, Error = Self::ConvertError>
    + Eq
    + PartialEq
    + Ord
    + Clone
    + Copy
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Display
    + Debug
    + PartialOrd
    + Hash
    + Default
    + Send
    + Sync
    + serde::Serialize
    + 'static
{
    type ConvertError: std::error::Error + Debug;
    const MAX: Self;
    const ONE: Self;
    const ZERO: Self;

    fn range_from_zero(self) -> std::ops::Range<usize> {
        0..self.try_into().unwrap()
    }
    fn iter_from_zero<IV>(self) -> impl ExactSizeIterator<Item = IV>
    where
        IV: TryFrom<usize>,
        <IV as TryFrom<usize>>::Error: Debug,
    {
        (0..self.try_into().unwrap()).map(|x| x.try_into().unwrap())
    }
}

impl Idx for usize {
    const MAX: Self = Self::MAX;
    type ConvertError = Infallible;
    const ONE: Self = 1;
    const ZERO: Self = 0;
}

impl Idx for u32 {
    const MAX: Self = Self::MAX;
    type ConvertError = TryFromIntError;
    const ONE: Self = 1;
    const ZERO: Self = 0;
}

/// Hexahedron
#[derive(
    Default, Clone, Copy, PartialEq, Eq, Hash, Debug, Index, IndexMut, IntoIterator, From, AsRef,
)]
#[as_ref(forward)]
pub struct Hexahedron<T: Idx = usize>([T; 8]);
/// Prism
#[derive(
    Default, Clone, Copy, PartialEq, Eq, Hash, Debug, Index, IndexMut, IntoIterator, From, AsRef,
)]
#[as_ref(forward)]
pub struct Prism<T: Idx = usize>([T; 6]);
/// Pyramid
#[derive(
    Default, Clone, Copy, PartialEq, Eq, Hash, Debug, Index, IndexMut, IntoIterator, From, AsRef,
)]
#[as_ref(forward)]
pub struct Pyramid<T: Idx = usize>([T; 5]);
/// Quadrangle
#[derive(
    Default, Clone, Copy, PartialEq, Eq, Hash, Debug, Index, IndexMut, IntoIterator, From, AsRef,
)]
#[as_ref(forward)]
pub struct Quadrangle<T: Idx = usize>([T; 4]);

/// Tetrahedron
#[derive(
    Default, Clone, Copy, PartialEq, Eq, Hash, Debug, Index, IndexMut, IntoIterator, From, AsRef,
)]
#[as_ref(forward)]
pub struct Tetrahedron<T: Idx = usize>([T; 4]);
#[derive(Clone, Copy, Debug, Index, IndexMut, IntoIterator, From, AsRef)]
#[as_ref(forward)]
pub struct GTetrahedron<const D: usize>([Vertex<D>; 4]);
/// Triangle
#[derive(
    Default, Clone, Copy, PartialEq, Eq, Hash, Debug, Index, IndexMut, IntoIterator, From, AsRef,
)]
#[as_ref(forward)]
pub struct Triangle<T: Idx = usize>([T; 3]);
#[derive(Clone, Copy, Debug, Index, IndexMut, IntoIterator, From, AsRef)]
#[as_ref(forward)]
pub struct GTriangle<const D: usize>([Vertex<D>; 3]);
/// Edge
#[derive(
    Default, Clone, Copy, PartialEq, Eq, Hash, Debug, Index, IndexMut, IntoIterator, From, AsRef,
)]
#[as_ref(forward)]
pub struct Edge<T: Idx = usize>([T; 2]);
#[derive(Clone, Copy, Debug, Index, IndexMut, IntoIterator, From, AsRef)]
#[as_ref(forward)]
pub struct GEdge<const D: usize>([Vertex<D>; 2]);
/// Node
#[derive(
    Default, Clone, Copy, PartialEq, Eq, Hash, Debug, Index, IndexMut, IntoIterator, From, AsRef,
)]
#[as_ref(forward)]
pub struct Node<T: Idx = usize>([T; 1]);
#[derive(Clone, Copy, Debug, Index, IndexMut, IntoIterator, From, AsRef)]
#[as_ref(forward)]
pub struct GNode<const D: usize>([Vertex<D>; 1]);

pub(crate) fn sort_elem_min_ids<T: Idx, C: Simplex<T>, I: ExactSizeIterator<Item = C>>(
    elems: I,
) -> Vec<T> {
    let n_elems = elems.len();

    let min_ids = elems.map(|e| e.into_iter().min()).collect::<Vec<_>>();
    let mut indices = (0..n_elems)
        .map(|x| x.try_into().unwrap())
        .collect::<Vec<T>>();
    indices.sort_by_key(|&i| min_ids[i.try_into().unwrap()]);
    indices
}

/// Compute the maximum and average bandwidth of a connectivity
pub fn bandwidth<T: Idx, C: Simplex<T>, I: ExactSizeIterator<Item = C>>(elems: I) -> (usize, f64) {
    let n_elems = elems.len();

    let (bmax, bmean) = elems.fold((0_usize, 0_usize), |a, e| {
        let max_id = e.into_iter().max().unwrap();
        let min_id = e.into_iter().min().unwrap();
        let tmp = (max_id - min_id).try_into().unwrap();
        (a.0.max(tmp), a.1 + tmp)
    });
    let bmean = bmean as f64 / n_elems as f64;
    (bmax, bmean)
}

// /// Collect elements into a fixed shape
// pub fn collect_elems<'a, C: Cell, const C1: usize, I: ExactSizeIterator<Item = &'a Cell<C0>>>(
//     elems: I,
// ) -> Vec<C1>> {
//     assert_eq!(C0, C1);
//     let mut res = Vec::with_capacity(elems.len());
//     let mut new = [0; C1];
//     for e in elems {
//         new.copy_from_slice(e);
//         res.push(new);
//     }
//     res
// }

/// Submesh of a `Mesh<D, C, F>`, with information about the vertices, element
/// and face ids in the parent mesh
pub struct SubMesh<T: Idx, const D: usize, C: Simplex<T>, M: Mesh<T, D, C>> {
    /// Mesh
    pub mesh: M,
    /// Indices of the vertices of `mesh` in the parent mesh
    pub parent_vert_ids: Vec<usize>,
    /// Indices of the element of `mesh` in the parent mesh
    pub parent_elem_ids: Vec<usize>,
    /// Indices of the faces of `mesh` in the parent mesh
    pub parent_face_ids: Vec<usize>,
    _t: PhantomData<T>,
    _c: PhantomData<C>,
}

impl<T: Idx, const D: usize, C: Simplex<T>, M: Mesh<T, D, C>> SubMesh<T, D, C, M> {
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
            _c: PhantomData::<C>,
            _t: PhantomData::<T>,
        }
    }
}

/// D-dimensional mesh containing simplices with C nodes
/// F = C-1 is given explicitely to be usable with rust stable
pub trait Mesh<T: Idx, const D: usize, C: Simplex<T>>: Send + Sync + Sized {
    /// Create a new mesh from slices of Vertex, Cell and Face
    #[must_use]
    fn new(
        verts: &[Vertex<D>],
        elems: &[C],
        etags: &[Tag],
        faces: &[C::FACE],
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
    fn n_verts(&self) -> T;

    /// Get the `i`th vertex
    fn vert(&self, i: T) -> Vertex<D>;

    /// Parallel iterator over the vertices
    fn par_verts(&self) -> impl IndexedParallelIterator<Item = Vertex<D>> + Clone + '_;

    /// Sequential iterator over the vertices
    fn verts(&self) -> impl ExactSizeIterator<Item = Vertex<D>> + Clone + '_;

    /// Add vertices to the mesh
    fn add_verts<I: ExactSizeIterator<Item = Vertex<D>>>(&mut self, v: I);

    /// Number of elements
    fn n_elems(&self) -> T;

    /// Get the `i`th element
    fn elem(&self, i: T) -> C;

    /// Invert the `i`th element
    fn invert_elem(&mut self, i: T);

    /// Parallel iterator over the mesh elements
    fn par_elems(&self) -> impl IndexedParallelIterator<Item = C> + Clone + '_;

    /// Sequential iterator over the mesh elements
    fn elems(&self) -> impl ExactSizeIterator<Item = C> + Clone + '_;

    /// Add elements to the mesh
    fn add_elems<I1: ExactSizeIterator<Item = C>, I2: ExactSizeIterator<Item = Tag>>(
        &mut self,
        elems: I1,
        etags: I2,
    );

    /// Remove all the elements
    fn clear_elems(&mut self);

    /// Add elements to the mesh
    fn add_elems_and_tags<I: ExactSizeIterator<Item = (C, Tag)>>(&mut self, elems_and_tags: I);

    /// Get the tag of the `i`th element
    fn etag(&self, i: T) -> Tag;

    /// Parallel iterator over the element tags
    fn par_etags(&self) -> impl IndexedParallelIterator<Item = Tag> + Clone + '_;

    /// Sequential iterator over the element tags
    fn etags(&self) -> impl ExactSizeIterator<Item = Tag> + Clone + '_;

    /// Get the vertices of element `e`
    fn gelem(&self, e: &C) -> C::GEOM<D> {
        let mut res = C::GEOM::default();

        for (j, k) in e.into_iter().enumerate() {
            res[j] = self.vert(k);
        }
        res
    }

    /// Parallel iterator over element vertices
    fn par_gelems(&self) -> impl IndexedParallelIterator<Item = C::GEOM<D>> + Clone + '_ {
        self.par_elems().map(|e| self.gelem(&e))
    }

    /// Sequential iterator over element vertices
    fn gelems(&self) -> impl ExactSizeIterator<Item = C::GEOM<D>> + Clone + '_ {
        self.elems().map(|e| self.gelem(&e))
    }

    /// Number of faces
    fn n_faces(&self) -> T;

    /// Get the `i`th face
    fn face(&self, i: T) -> C::FACE;

    /// Invert the `i`th face
    fn invert_face(&mut self, i: T);

    /// Parallel iterator over the faces
    fn par_faces(&self) -> impl IndexedParallelIterator<Item = C::FACE> + Clone + '_;

    /// Sequential itertor over the faces
    fn faces(&self) -> impl ExactSizeIterator<Item = C::FACE> + Clone + '_;
    /// Add faces to the mesh
    fn add_faces<I1: ExactSizeIterator<Item = C::FACE>, I2: ExactSizeIterator<Item = Tag>>(
        &mut self,
        faces: I1,
        ftags: I2,
    );

    /// Clear all the mesh faces
    fn clear_faces(&mut self);

    /// Add faces to the mesh
    fn add_faces_and_tags<I: ExactSizeIterator<Item = (C::FACE, Tag)>>(
        &mut self,
        faces_and_tags: I,
    );

    /// Get the tag of the `i`th face
    fn ftag(&self, i: T) -> Tag;

    /// Parallel iterator over the mesh faces
    fn par_ftags(&self) -> impl IndexedParallelIterator<Item = Tag> + Clone + '_;

    /// Sequential iterator over the mesh faces
    fn ftags(&self) -> impl ExactSizeIterator<Item = Tag> + Clone + '_;

    /// Get the vertices of face `f`
    fn gface(&self, f: &C::FACE) -> <C::FACE as Simplex<T>>::GEOM<D> {
        let mut res = <C::FACE as Simplex<T>>::GEOM::default();
        for j in 0..C::FACE::N_VERTS {
            res[j] = self.vert(f[j]);
        }
        res
    }

    /// Parallel iterator over face vertices
    fn par_gfaces(
        &self,
    ) -> impl IndexedParallelIterator<Item = <C::FACE as Simplex<T>>::GEOM<D>> + Clone + '_ {
        self.par_faces().map(|f| self.gface(&f))
    }

    /// Sequential iterator over face vertices
    fn gfaces(
        &self,
    ) -> impl ExactSizeIterator<Item = <C::FACE as Simplex<T>>::GEOM<D>> + Clone + '_ {
        self.faces().map(|f| self.gface(&f))
    }

    /// Compute the mesh edges
    /// a map from sorted edge `[i0, i1]` (`i0 < i1`) to the edge index is returned
    fn edges(&self) -> FxHashMap<Edge<T>, T> {
        let mut res = FxHashMap::with_hasher(FxBuildHasher);

        for e in self.elems() {
            for edg in e.edges() {
                let edg = edg.sorted();
                if !res.contains_key(&edg) {
                    res.insert(edg, res.len().try_into().unwrap());
                }
            }
        }

        res
    }

    /// Compute the vertex-to-element connectivity
    fn vertex_to_elems(&self) -> CSRGraph<T> {
        CSRGraph::transpose(self.elems(), Some(self.n_verts()))
    }

    /// Compute the vertex-to-vertex connectivity
    fn vertex_to_vertices(&self) -> CSRGraph<T> {
        let edges = self.edges();
        CSRGraph::from_edges(edges.keys().map(|e| [e[0], e[1]]), Some(self.n_verts()))
    }

    /// Check if faces can be oriented for the current values of D and C
    #[must_use]
    fn faces_are_oriented() -> bool {
        C::FACE::DIM == D
    }

    /// Compute all the mesh faces (boundary & internal)
    /// a map from sorted face `[i0, i1, ...]` (`i0 < i1 < ...`) to the face index and face elements
    /// (`e0` and `e1`) is returned.
    /// If the faces can be oriented, it is oriented outwards for `e0` and inwards for `e1`
    /// If the faces only belongs to one element, `i1 = T::MAX`
    fn all_faces(&self) -> FxHashMap<C::FACE, [T; 3]> {
        let mut res: FxHashMap<C::FACE, [T; 3]> = FxHashMap::with_hasher(FxBuildHasher);
        let mut idx = 0;

        for (i_elem, e) in self.elems().enumerate() {
            for f in e.faces() {
                let tmp = f.sorted();
                if C::FACE::N_VERTS > 1 {
                    let i = if f.is_same(&tmp) { 1 } else { 2 };
                    match res.entry(tmp) {
                        std::collections::hash_map::Entry::Occupied(occupied_entry) => {
                            let arr = occupied_entry.into_mut();
                            assert_eq!(
                                arr[i],
                                T::MAX,
                                "face {} belongs to {} and {} with orientation {i}",
                                arr[0],
                                arr[i],
                                i_elem
                            );
                            arr[i] = i_elem.try_into().unwrap();
                        }
                        std::collections::hash_map::Entry::Vacant(vacant_entry) => {
                            let mut arr = [idx.try_into().unwrap(), T::MAX, T::MAX];
                            arr[i] = i_elem.try_into().unwrap();
                            idx += 1;
                            vacant_entry.insert(arr);
                        }
                    }
                } else {
                    match res.entry(tmp) {
                        std::collections::hash_map::Entry::Occupied(occupied_entry) => {
                            let arr = occupied_entry.into_mut();
                            if arr[1] == T::MAX {
                                arr[1] = i_elem.try_into().unwrap();
                            } else {
                                assert_eq!(arr[2], T::MAX);
                                arr[2] = i_elem.try_into().unwrap();
                            }
                        }
                        std::collections::hash_map::Entry::Vacant(vacant_entry) => {
                            let arr = [idx.try_into().unwrap(), i_elem.try_into().unwrap(), T::MAX];
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
    fn element_pairs(&self, faces: &FxHashMap<C::FACE, [T; 3]>) -> CSRGraph<T> {
        let e2e = faces
            .iter()
            .map(|(_, &[_, i0, i1])| [i0, i1])
            .filter(|&[i0, i1]| i0 != T::MAX && i1 != T::MAX)
            .collect::<Vec<_>>();

        CSRGraph::from_edges(e2e.iter().copied(), Some(self.n_elems()))
    }

    /// Fix the mesh
    ///   - add missing boundary faces
    ///   - add missing internal faces
    ///   - fix orientation
    ///
    /// and check the valitity
    #[allow(clippy::type_complexity)]
    fn fix(&mut self) -> Result<(FxHashMap<Tag, Tag>, FxHashMap<[Tag; 2], Tag>)> {
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
            .map(|(i, _)| self.invert_elem(i.try_into().unwrap()))
            .count();
        debug!("{n} elems reoriented");
        n
    }

    /// Fix the orientation
    /// - of boundary faces to be oriented outwards (if possible)
    /// - of internal faces to be oriented from the lower to the higher element tag
    fn fix_faces_orientation(&mut self, all_faces: &FxHashMap<C::FACE, [T; 3]>) -> usize {
        if Self::faces_are_oriented() {
            let flg = self
                .faces()
                .map(|f| {
                    let gf = self.gface(&f);
                    let fc = gf.center();
                    let normal = gf.center();

                    let [_, i0, i1] = all_faces.get(&f.sorted()).unwrap();
                    let i = if *i0 == T::MAX || *i1 == T::MAX {
                        if *i1 == T::MAX { *i0 } else { *i1 }
                    } else {
                        let t0 = self.etag(*i0);
                        let t1 = self.etag(*i1);
                        assert_ne!(t0, t1);
                        if t0 < t1 { *i0 } else { *i1 }
                    };
                    let ge = self.gelem(&self.elem(i));
                    let ec = ge.center();

                    normal.dot(&(fc - ec)) < 0.0
                })
                .collect::<Vec<_>>();

            let n = flg
                .iter()
                .enumerate()
                .filter(|(_, f)| **f)
                .map(|(i, _)| self.invert_face(i.try_into().unwrap()))
                .count();
            debug!("{n} faces reoriented");
            return n;
        }
        0
    }

    /// Compute the faces that are connected to only one element and that are not already tagged
    fn tag_boundary_faces(
        &mut self,
        all_faces: &FxHashMap<C::FACE, [T; 3]>,
    ) -> FxHashMap<Tag, Tag> {
        let mut res = FxHashMap::with_hasher(FxBuildHasher);

        let tagged_faces = self
            .par_faces()
            .zip(self.par_ftags())
            .map(|(f, t)| (f.sorted(), t))
            .collect::<FxHashMap<_, _>>();

        let mut next_tag = -self.par_ftags().max().unwrap_or(0) - 1;

        // add untagged boundary faces
        for (f, &[_, i0, i1]) in all_faces {
            if i0 == T::MAX && !tagged_faces.contains_key(f) {
                let etag = self.etag(i1);
                if let Some(&tmp) = res.get(&etag) {
                    self.add_faces(std::iter::once(f).copied(), std::iter::once(tmp));
                } else {
                    res.insert(etag, next_tag);
                    self.add_faces(std::iter::once(f).copied(), std::iter::once(next_tag));
                    next_tag -= 1;
                }
            }
            if i1 == T::MAX && !tagged_faces.contains_key(f) {
                let etag = self.etag(i0);
                if let Some(&tmp) = res.get(&etag) {
                    self.add_faces(std::iter::once(f).copied(), std::iter::once(tmp));
                } else {
                    res.insert(etag, next_tag);
                    self.add_faces(std::iter::once(f).copied(), std::iter::once(next_tag));
                    next_tag -= 1;
                }
            }
        }

        res
    }

    /// Compute the faces that are connected to elements with different tags and that are not already tagged
    fn tag_internal_faces(
        &mut self,
        all_faces: &FxHashMap<C::FACE, [T; 3]>,
    ) -> FxHashMap<[Tag; 2], Tag> {
        let mut res = FxHashMap::with_hasher(FxBuildHasher);

        let tagged_faces = self
            .par_faces()
            .zip(self.par_ftags())
            .map(|(f, t)| (f.sorted(), t))
            .collect::<FxHashMap<_, _>>();

        let mut next_tag = -self.par_ftags().max().unwrap_or(0) - 1;

        // check tagged internal faces
        for (f, &[_, i0, i1]) in all_faces {
            if i0 != T::MAX && i1 != T::MAX {
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
            if i0 != T::MAX && i1 != T::MAX {
                let t0 = self.etag(i0);
                let t1 = self.etag(i1);
                if t0 != t1 && !tagged_faces.contains_key(f) {
                    let tags = if t0 < t1 { [t0, t1] } else { [t1, t0] };
                    if let Some(&tmp) = res.get(&tags) {
                        self.add_faces(std::iter::once(f).copied(), std::iter::once(tmp));
                    } else {
                        res.insert(tags, next_tag);
                        self.add_faces(std::iter::once(f).copied(), std::iter::once(next_tag));
                        next_tag -= 1;
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
    fn check_simple(&self) -> Result<()> {
        self.check(&self.all_faces())
    }

    /// Check the mesh validity
    ///   - connectivity and tag sizes
    ///   - element to vertex connectivities
    ///   - element orientations
    ///   - boundary faces and faces connecting elements with different tags are present
    fn check(&self, all_faces: &FxHashMap<C::FACE, [T; 3]>) -> Result<()> {
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
            if *i0 == T::MAX || *i1 == T::MAX {
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

        for f in self.faces() {
            let gf = self.gface(&f);
            let fc = gf.center();
            let tmp = f.sorted();
            let [_, i0, i1] = all_faces.get(&tmp).unwrap();
            if *i0 != T::MAX && *i1 != T::MAX && self.etag(*i0) == self.etag(*i1) {
                return Err(Error::from(&format!(
                    "Tagged face inside the domain: center = {fc:?}",
                )));
            } else if *i0 == T::MAX || *i1 == T::MAX {
                let i = if *i1 == T::MAX { *i0 } else { *i1 };
                let ge = self.gelem(&self.elem(i));
                let ec = ge.center();
                if Self::faces_are_oriented() {
                    let n = gf.normal();
                    if n.dot(&(fc - ec)) < 0.0 {
                        return Err(Error::from(&format!(
                            "Invalid face orientation: center = {fc:?}, normal = {n:?}, face = {f:?}"
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
                    *i0 == T::MAX || *i1 == T::MAX
                })
                .map(|f| {
                    let gf = self.gface(&f);
                    gf.center().dot(&gf.normal())
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

    /// Compute the gradient of a field defined of the mesh vertices using weighted
    /// least squares
    fn gradient(
        &self,
        v2v: &CSRGraph<T>,
        weight: i32,
        f: &[f64],
    ) -> impl IndexedParallelIterator<Item = Vertex<D>>
    where
        nalgebra::Const<D>: nalgebra::Dim,
    {
        self.n_verts()
            .range_from_zero()
            .into_par_iter()
            .map(move |i| {
                let x = self.vert(i.try_into().unwrap());
                let neighbors = v2v.row(i.try_into().unwrap());
                let dx = neighbors.iter().map(|&j| self.vert(j) - x);
                let ls = LeastSquaresGradient::new(weight, dx).unwrap();
                let df = neighbors.iter().map(|&j| f[j.try_into().unwrap()] - f[i]);
                ls.gradient(df)
            })
    }

    /// Integrate `g(f)` over the mesh, where `f` is a field defined on the mesh vertices
    fn integrate<G: Fn(f64) -> f64 + Send + Sync>(&self, f: &[f64], op: G) -> f64 {
        let (qw, qp) = C::quadrature();
        debug_assert!(qp.iter().all(|x| x.len() == C::N_VERTS - 1));

        self.par_elems()
            .map(|e| {
                let res = qw
                    .iter()
                    .zip(qp.iter())
                    .map(|(w, pt)| {
                        let x0 = f[e[0].try_into().unwrap()];
                        let mut x_pt = x0;
                        for (&b, j) in pt.iter().zip(e.into_iter().skip(1)) {
                            let mut dx = f[j.try_into().unwrap()] - x0;
                            dx *= b;
                            x_pt += dx;
                        }
                        *w * op(x_pt)
                    })
                    .sum::<f64>();
                self.gelem(&e).vol() * res
            })
            .sum::<f64>()
    }

    /// Compute the norm of a field `f` defined on the mesh vertices
    fn norm(&self, f: &[f64]) -> f64 {
        self.integrate(f, |x| x.powi(2)).sqrt()
    }

    /// Reorder the mesh vertices
    #[must_use]
    fn reorder_vertices(&self, vert_indices: &[T]) -> Self {
        assert_eq!(vert_indices.len(), self.n_verts().try_into().unwrap());

        let mut new_vert_indices: Vec<T> = vec![T::ZERO; self.n_verts().try_into().unwrap()];
        vert_indices.iter().enumerate().for_each(|(i, &new_i)| {
            new_vert_indices[new_i.try_into().unwrap()] = i.try_into().unwrap();
        });
        let new_verts = vert_indices.iter().map(|&i| self.vert(i));
        let new_elems = self.elems().map(|mut e| {
            for i in 0..C::N_VERTS {
                e[i] = new_vert_indices[e[i].try_into().unwrap()];
            }
            e
        });
        let new_faces = self.faces().map(|mut f| {
            for i in 0..C::FACE::N_VERTS {
                f[i] = new_vert_indices[f[i].try_into().unwrap()];
            }
            f
        });

        let mut res = Self::empty();
        res.add_verts(new_verts);
        res.add_elems(new_elems, self.etags());
        res.add_faces(new_faces, self.ftags());
        res
    }

    /// Reorder the mesh elements (in place)
    fn reorder_elems(&mut self, elem_indices: &[T]) {
        assert_eq!(elem_indices.len(), self.n_elems().try_into().unwrap());

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
    fn reorder_faces(&mut self, face_indices: &[T]) {
        assert_eq!(face_indices.len(), self.n_faces().try_into().unwrap());

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
    fn reorder_rcm(&self) -> (Self, Vec<T>, Vec<T>, Vec<T>) {
        let graph = self.vertex_to_vertices();
        let vert_ids = graph.reverse_cuthill_mckee();
        let mut res = self.reorder_vertices(&vert_ids);

        let elem_ids = sort_elem_min_ids(res.elems());
        res.reorder_elems(&elem_ids);

        let face_ids = sort_elem_min_ids(res.faces());
        res.reorder_faces(&face_ids);

        (res, vert_ids, elem_ids, face_ids)
    }

    /// Set the element tags
    fn set_etags<I: ExactSizeIterator<Item = Tag>>(&mut self, tags: I);

    /// Set the partition as etags from an usize slice
    fn set_partition(&mut self, part: &[T]) {
        assert_eq!(self.n_elems().try_into().unwrap(), part.len());
        self.set_etags(part.iter().map(|&x| x.try_into().unwrap() as Tag + 1));
    }

    /// Reorder the mesh (Hilbert):
    ///   - RCM orderting based on the vertex-to-vertex connectivity is used for the mesh vertices
    ///   - elements and faces are sorted by their minimum vertex index
    fn reorder_hilbert(&self) -> (Self, Vec<T>, Vec<T>, Vec<T>) {
        let vert_ids = hilbert_indices::<T, _, _>(self.verts());
        let mut res = self.reorder_vertices(&vert_ids);

        let elem_ids = sort_elem_min_ids(res.elems());
        res.reorder_elems(&elem_ids);

        let face_ids = sort_elem_min_ids(res.faces());
        res.reorder_faces(&face_ids);

        (res, vert_ids, elem_ids, face_ids)
    }

    /// Get the i-th partition
    fn get_partition(&self, i: usize) -> SubMesh<T, D, C, Self> {
        SubMesh::new(self, |t| t == i as Tag + 1)
    }

    /// Partition the mesh (RCM ordering applied to the element to element connectivity)
    fn partition<P: Partitioner<T>>(
        &mut self,
        n_parts: T,
        weights: Option<Vec<f64>>,
    ) -> Result<(f64, f64)> {
        let partitioner = P::new(self, n_parts, weights)?;
        let parts = partitioner.compute()?;
        assert_eq!(parts.len(), self.n_elems().try_into().unwrap());

        let quality = partitioner.partition_quality(&parts);
        let imbalance = partitioner.partition_imbalance(&parts);

        self.set_partition(&parts);

        Ok((quality, imbalance))
    }

    /// Randomly shuffle vertices, elements and faces
    #[must_use]
    fn random_shuffle(&self) -> Self {
        let mut rng = StdRng::seed_from_u64(1234);

        let mut vert_ids: Vec<_> = self.n_verts().iter_from_zero().collect();
        vert_ids.shuffle(&mut rng);

        let mut res = self.reorder_vertices(&vert_ids);

        let mut elem_ids: Vec<_> = self.n_elems().iter_from_zero().collect();
        elem_ids.shuffle(&mut rng);
        res.reorder_elems(&elem_ids);

        let mut face_ids: Vec<_> = self.n_faces().iter_from_zero().collect();
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

        match C::N_VERTS {
            4 => {
                if let Ok(iter) = reader.read_tetrahedra() {
                    res.add_elems_and_tags(iter.map(|(e, t)| (C::from_iter(e), t as Tag)));
                }
            }
            3 => {
                if let Ok(iter) = reader.read_triangles() {
                    res.add_elems_and_tags(iter.map(|(e, t)| (C::from_iter(e), t as Tag)));
                }
            }
            2 => {
                if let Ok(iter) = reader.read_edges() {
                    res.add_elems_and_tags(iter.map(|(e, t)| (C::from_iter(e), t as Tag)));
                }
            }
            _ => unimplemented!(),
        }

        match C::FACE::N_VERTS {
            3 => {
                if let Ok(iter) = reader.read_triangles() {
                    res.add_faces_and_tags(iter.map(|(e, t)| (C::FACE::from_iter(e), t as Tag)));
                }
            }
            2 => {
                if let Ok(iter) = reader.read_edges() {
                    res.add_faces_and_tags(iter.map(|(e, t)| (C::FACE::from_iter(e), t as Tag)));
                }
            }
            1 => warn!("not reading faces when elements are edges"),
            _ => unimplemented!(),
        }

        Ok(res)
    }

    /// Export the mesh to a `.meshb` file
    #[allow(clippy::unnecessary_fallible_conversions)]
    fn write_meshb(&self, file_name: &str) -> Result<()> {
        let mut writer = MeshbWriter::new(file_name, 3, D as u8)?;

        writer.write_vertices::<D, _, _>(
            self.verts().map(|x| {
                let mut tmp = [0.0; D];
                tmp.copy_from_slice(x.as_ref());
                tmp
            }),
            self.n_verts().range_from_zero().map(|_| 1),
        )?;

        match C::N_VERTS {
            4 => writer.write_tetrahedra(
                self.elems()
                    .map(|x| std::array::from_fn(|i| x[i].try_into().unwrap())),
                self.etags().map(|x| x.try_into().unwrap()),
            )?,
            3 => writer.write_triangles(
                self.elems()
                    .map(|x| std::array::from_fn(|i| x[i].try_into().unwrap())),
                self.etags().map(|x| x.try_into().unwrap()),
            )?,
            2 => writer.write_edges(
                self.elems()
                    .map(|x| std::array::from_fn(|i| x[i].try_into().unwrap())),
                self.etags().map(|x| x.try_into().unwrap()),
            )?,
            _ => unimplemented!(),
        }

        match C::FACE::N_VERTS {
            3 => writer.write_triangles(
                self.faces()
                    .map(|x| std::array::from_fn(|i| x[i].try_into().unwrap())),
                self.ftags().map(|x| x.try_into().unwrap()),
            )?,
            2 => writer.write_edges(
                self.faces()
                    .map(|x| std::array::from_fn(|i| x[i].try_into().unwrap())),
                self.ftags().map(|x| x.try_into().unwrap()),
            )?,
            1 => {
                if self.n_faces().try_into().unwrap() != 0 {
                    warn!("skip faces in meshb export");
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
        assert_eq!(arr.len(), N * self.n_verts().try_into().unwrap());

        let mut writer = MeshbWriter::new(file_name, 3, D as u8)?;
        writer.write_solution(arr.chunks(N).map(f))?;
        writer.close();

        Ok(())
    }

    fn write_solb(&self, arr: &[f64], file_name: &str) -> Result<()> {
        let n_comp = arr.len() / self.n_verts().try_into().unwrap();
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
    fn extract_faces<M: Mesh<T, D, C::FACE>, G: Fn(Tag) -> bool>(&self, filter: G) -> (M, Vec<T>) {
        let mut new_ids = vec![T::MAX; self.n_verts().try_into().unwrap()];
        let mut vert_ids = Vec::new();
        let mut next = T::ZERO;

        let n_faces = self
            .faces()
            .zip(self.ftags())
            .filter(|(_, t)| filter(*t))
            .map(|(f, _)| {
                for i in f {
                    if new_ids[i.try_into().unwrap()] == T::MAX {
                        new_ids[i.try_into().unwrap()] = next;
                        vert_ids.push(i);
                        next += T::ONE;
                    }
                }
            })
            .count();
        let n_verts = next;

        let mut verts = vec![Vertex::<D>::zeros(); n_verts.try_into().unwrap()];
        let mut faces = Vec::with_capacity(n_faces);
        let mut ftags = Vec::with_capacity(n_faces);

        new_ids
            .iter()
            .enumerate()
            .filter(|&(_, j)| *j != T::MAX)
            .for_each(|(i, &j)| verts[j.try_into().unwrap()] = self.vert(i.try_into().unwrap()));
        self.faces()
            .zip(self.ftags())
            .filter(|(f, _)| {
                f.into_iter()
                    .all(|i| new_ids[i.try_into().unwrap()] != T::MAX)
            })
            .for_each(|(f, t)| {
                faces.push(C::FACE::from_iter(
                    f.into_iter().map(|i| new_ids[i.try_into().unwrap()]),
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
    fn boundary<M: Mesh<T, D, C::FACE>>(&self) -> (M, Vec<T>) {
        self.extract_faces(|_| true)
    }

    /// Split quandrangles and add them to the mesh (C == 3 or F == 3)
    fn add_quadrangles<
        I1: ExactSizeIterator<Item = Quadrangle<T>>,
        I2: ExactSizeIterator<Item = Tag>,
    >(
        &mut self,
        quads: I1,
        tags: I2,
    ) {
        if C::N_VERTS == 3 {
            let mut tmp = Vec::with_capacity(2 * quads.len());
            quads.zip(tags).for_each(|(q, t)| {
                let tris = qua2tris(&q);
                for tri in tris {
                    tmp.push((C::from_other(tri), t));
                }
            });
            self.add_elems_and_tags(tmp.iter().copied());
        } else if C::FACE::N_VERTS == 3 {
            let mut tmp = Vec::with_capacity(2 * quads.len());
            quads.zip(tags).for_each(|(q, t)| {
                let tris = qua2tris(&q);
                for tri in tris {
                    tmp.push((<C::FACE as Simplex<T>>::from_other(tri), t));
                }
            });
            self.add_faces_and_tags(tmp.iter().copied());
        } else {
            unreachable!()
        }
    }

    /// Split hexahedra and add them to the mesh (C == 4)
    fn add_hexahedra<
        I1: ExactSizeIterator<Item = Hexahedron<T>>,
        I2: ExactSizeIterator<Item = Tag>,
    >(
        &mut self,
        hexs: I1,
        tags: I2,
    ) -> Vec<usize> {
        if C::N_VERTS == 4 {
            let mut tmp = Vec::with_capacity(6 * hexs.len());
            let mut ids = Vec::with_capacity(6 * hexs.len());
            hexs.zip(tags).enumerate().for_each(|(i, (q, t))| {
                let (tets, last_tet) = hex2tets(&q);
                for tet in tets {
                    tmp.push((C::from_other(tet), t));
                    ids.push(i);
                }
                if let Some(last_tet) = last_tet {
                    tmp.push((C::from_other(last_tet), t));
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
    fn add_prisms<I1: ExactSizeIterator<Item = Prism<T>>, I2: ExactSizeIterator<Item = Tag>>(
        &mut self,
        pris: I1,
        tags: I2,
    ) {
        if C::N_VERTS == 4 {
            let mut tmp = Vec::with_capacity(3 * pris.len());
            pris.zip(tags).for_each(|(q, t)| {
                let tets = pri2tets(&q);
                for tet in tets {
                    tmp.push((C::from_other(tet), t));
                }
            });
            self.add_elems_and_tags(tmp.iter().copied());
        } else {
            unreachable!()
        }
    }

    /// Split pyramids and add them to the mesh (C == 4)
    fn add_pyramids<I1: ExactSizeIterator<Item = Pyramid<T>>, I2: ExactSizeIterator<Item = Tag>>(
        &mut self,
        pyrs: I1,
        tags: I2,
    ) {
        if C::N_VERTS == 4 {
            let mut tmp = Vec::with_capacity(3 * pyrs.len());
            pyrs.zip(tags).for_each(|(q, t)| {
                let tets = pyr2tets(&q);
                for tet in tets {
                    tmp.push((C::from_other(tet), t));
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
    fn check_equals<M: Mesh<T, D, C>>(&self, other: &M, tol: f64) -> Result<()> {
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
            let p0 = self.vert(edg[0]);
            let p1 = self.vert(edg[1]);
            verts[i.try_into().unwrap()] = 0.5 * (p0 + p1);
        }
        res.add_verts(verts.iter().copied());

        // add offset to verts
        for v in edges.values_mut() {
            *v += self.n_verts();
        }

        // Cells
        match C::N_VERTS {
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
        match C::FACE::N_VERTS {
            3 => {
                let (faces, ftags) = split_tris(self.faces().zip(self.ftags()), &edges);
                res.add_faces(faces.iter().copied(), ftags.iter().copied());
            }
            2 => {
                let (faces, ftags) = split_edgs(self.faces().zip(self.ftags()), &edges);
                res.add_faces(faces.iter().copied(), ftags.iter().copied());
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
        all_faces: &FxHashMap<C::FACE, [T; 3]>,
    ) -> impl Iterator<Item = (T, T, f64)> {
        all_faces
            .iter()
            .filter(|&(_, [_, i0, i1])| *i0 != T::MAX && *i1 != T::MAX)
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
                let l = (self.vert(edg[0]) - self.vert(edg[1])).norm();
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
        let mut res = vec![false; self.n_verts().try_into().unwrap()];
        self.faces()
            .flatten()
            .for_each(|i| res[i.try_into().unwrap()] = true);
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
        let mut new_vert_ids = vec![T::MAX; n_verts_other.try_into().unwrap()];

        for (e, t) in other.elems().zip(other.etags()) {
            if elem_filter(t) {
                for i in e {
                    new_vert_ids[i.try_into().unwrap()] = T::MAX - T::ONE;
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
                        overts.push(other.vert(i.try_into().unwrap()));
                    }
                }
                let tree = PointIndex::new(overts.iter().copied());
                for (i, &flg) in self.boundary_flag().iter().enumerate() {
                    if flg {
                        let vx = self.vert(i.try_into().unwrap());
                        let (i_other, _) = tree.nearest_vert(&vx);
                        let i_other = oids[i_other];
                        if (vx - other.vert(i_other.try_into().unwrap())).norm() < merge_tol {
                            new_vert_ids[i_other] = i.try_into().unwrap();
                        }
                    }
                }
            }
        }

        // number & add the new vertices
        let mut next = n_verts;
        let mut added_verts = Vec::new();
        new_vert_ids.iter_mut().enumerate().for_each(|(i, x)| {
            if *x == T::MAX - T::ONE {
                added_verts.push(i);
                *x = next;
                next += T::ONE;
                self.add_verts(std::iter::once(other.vert(i.try_into().unwrap())));
            }
        });

        let mut added_elems = Vec::new();

        // keep track of the possible new faces
        let mut all_added_faces = FxHashSet::default();
        for (i, (mut e, t)) in other
            .elems()
            .zip(other.etags())
            .enumerate()
            .filter(|(_, (_, t))| elem_filter(*t))
        {
            added_elems.push(i);
            for i in 0..C::N_VERTS {
                e[i] = new_vert_ids[e[i].try_into().unwrap()];
            }
            self.add_elems(std::iter::once(e), std::iter::once(t));
            for face in e.faces() {
                all_added_faces.insert(face.sorted());
            }
        }

        let mut added_faces = Vec::new();
        for (i, (mut f, t)) in other
            .faces()
            .zip(other.ftags())
            .enumerate()
            .filter(|(_, (_, t))| face_filter(*t))
            .filter(|&(_, (mut f, _))| {
                for i in 0..C::FACE::N_VERTS {
                    f[i] = new_vert_ids[f[i].try_into().unwrap()];
                }
                all_added_faces.contains(&f.sorted())
            })
        {
            added_faces.push(i);
            for i in 0..C::FACE::N_VERTS {
                f[i] = new_vert_ids[f[i].try_into().unwrap()];
            }
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
}

/// D-dimensional mesh containing simplices with C nodes
/// F = C-1 is given explicitely to be usable with rust stable
pub trait MutMesh<T: Idx, const D: usize, C: Simplex<T>>: Mesh<T, D, C> {
    /// Sequential iterator over the vertices
    fn verts_mut(&mut self) -> impl ExactSizeIterator<Item = &mut Vertex<D>> + '_;

    /// Sequential iterator over the mesh elements
    fn elems_mut<'a>(&'a mut self) -> impl ExactSizeIterator<Item = &'a mut C> + 'a
    where
        C: 'a;

    /// Sequential iterator over the element tags
    fn etags_mut(&mut self) -> impl ExactSizeIterator<Item = &mut Tag> + '_;

    /// Sequential itertor over the faces
    fn faces_mut<'a>(&'a mut self) -> impl ExactSizeIterator<Item = &'a mut C::FACE> + 'a
    where
        C: 'a;

    /// Sequential iterator over the mesh faces
    fn ftags_mut(&mut self) -> impl ExactSizeIterator<Item = &mut Tag> + '_;
}

/// Generic meshes implemented with Vecs
pub struct GenericMesh<T: Idx, const D: usize, C: Simplex<T>> {
    verts: Vec<Vertex<D>>,
    elems: Vec<C>,
    etags: Vec<Tag>,
    faces: Vec<C::FACE>,
    ftags: Vec<Tag>,
}

impl<T: Idx, const D: usize, C: Simplex<T>> Mesh<T, D, C> for GenericMesh<T, D, C> {
    fn empty() -> Self {
        Self {
            verts: Vec::new(),
            elems: Vec::new(),
            etags: Vec::new(),
            faces: Vec::new(),
            ftags: Vec::new(),
        }
    }

    fn n_verts(&self) -> T {
        self.verts.len().try_into().unwrap()
    }

    fn vert(&self, i: T) -> Vertex<D> {
        self.verts[i.try_into().unwrap()]
    }

    fn verts(&self) -> impl ExactSizeIterator<Item = Vertex<D>> + Clone + '_ {
        self.verts.iter().copied()
    }

    fn par_verts(&self) -> impl IndexedParallelIterator<Item = Vertex<D>> + Clone + '_ {
        self.verts.par_iter().cloned()
    }

    fn add_verts<I: ExactSizeIterator<Item = Vertex<D>>>(&mut self, v: I) {
        self.verts.extend(v);
    }

    fn n_elems(&self) -> T {
        self.elems.len().try_into().unwrap()
    }

    fn elem(&self, i: T) -> C {
        self.elems[i.try_into().unwrap()]
    }

    fn invert_elem(&mut self, i: T) {
        self.elems[i.try_into().unwrap()].invert();
    }

    fn elems(&self) -> impl ExactSizeIterator<Item = C> + Clone + '_ {
        self.elems.iter().copied()
    }

    fn par_elems(&self) -> impl IndexedParallelIterator<Item = C> + Clone + '_ {
        self.elems.par_iter().cloned()
    }

    fn etag(&self, i: T) -> Tag {
        self.etags[i.try_into().unwrap()]
    }

    fn etags(&self) -> impl ExactSizeIterator<Item = Tag> + Clone + '_ {
        self.etags.iter().copied()
    }

    fn set_etags<I: ExactSizeIterator<Item = Tag>>(&mut self, tags: I) {
        self.etags.iter_mut().zip(tags).for_each(|(x, y)| *x = y);
    }

    fn par_etags(&self) -> impl IndexedParallelIterator<Item = Tag> + Clone + '_ {
        self.etags.par_iter().cloned()
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

    fn add_elems_and_tags<I: ExactSizeIterator<Item = (C, Tag)>>(&mut self, elems_and_tags: I) {
        self.elems.reserve(elems_and_tags.len());
        self.etags.reserve(elems_and_tags.len());
        for (e, t) in elems_and_tags {
            self.elems.push(e);
            self.etags.push(t);
        }
    }

    fn n_faces(&self) -> T {
        self.faces.len().try_into().unwrap()
    }

    fn face(&self, i: T) -> C::FACE {
        self.faces[i.try_into().unwrap()]
    }

    fn invert_face(&mut self, i: T) {
        self.faces[i.try_into().unwrap()].invert();
    }

    fn faces(&self) -> impl ExactSizeIterator<Item = C::FACE> + Clone + '_ {
        self.faces.iter().copied()
    }

    fn par_faces(&self) -> impl IndexedParallelIterator<Item = C::FACE> + Clone + '_ {
        self.faces.par_iter().cloned()
    }

    fn ftag(&self, i: T) -> Tag {
        self.ftags[i.try_into().unwrap()]
    }

    fn ftags(&self) -> impl ExactSizeIterator<Item = Tag> + Clone + '_ {
        self.ftags.iter().copied()
    }

    fn par_ftags(&self) -> impl IndexedParallelIterator<Item = Tag> + Clone + '_ {
        self.ftags.par_iter().cloned()
    }

    fn add_faces<I1: ExactSizeIterator<Item = C::FACE>, I2: ExactSizeIterator<Item = Tag>>(
        &mut self,
        faces: I1,
        ftags: I2,
    ) {
        self.faces.extend(faces);
        self.ftags.extend(ftags);
    }

    fn clear_faces(&mut self) {
        self.faces.clear();
        self.ftags.clear();
    }

    fn add_faces_and_tags<I: ExactSizeIterator<Item = (C::FACE, Tag)>>(
        &mut self,
        faces_and_tags: I,
    ) {
        self.faces.reserve(faces_and_tags.len());
        self.ftags.reserve(faces_and_tags.len());
        for (e, t) in faces_and_tags {
            self.faces.push(e);
            self.ftags.push(t);
        }
    }
}

impl<T: Idx, const D: usize, C: Simplex<T>> MutMesh<T, D, C> for GenericMesh<T, D, C> {
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
