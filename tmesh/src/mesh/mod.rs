//! Simplex meshes in D dimensions, represented by
//!   - the vertices
//!   - elements of type `Cell<C>` and element tags
//!   - faces of type `Face<F>` and element tags
//!
//! F = C-1 cannot be imposed in rust stable
mod boundary_mesh_2d;
mod boundary_mesh_3d;
mod mesh_2d;
mod mesh_3d;
mod simplices;
mod to_simplices;
mod twovec;

mod split;

mod hilbert;
pub mod partition;

pub mod least_squares;

/// Cell
pub type Cell<const C: usize> = [usize; C];
/// Face
pub type Face<const F: usize> = Cell<F>;

/// Hexahedron
pub type Hexahedron = Cell<8>;
/// Prism
pub type Prism = Cell<6>;
/// Pyramid
pub type Pyramid = Cell<5>;
/// Quadrangle
pub type Quadrangle = Cell<4>;

/// Tetrahedron
pub type Tetrahedron = Cell<4>;
/// Triangle
pub type Triangle = Cell<3>;
/// Edge
pub type Edge = Cell<2>;
/// Node
pub type Node = Cell<1>;

pub use boundary_mesh_2d::BoundaryMesh2d;
pub use boundary_mesh_3d::{BoundaryMesh3d, read_stl};
use hilbert::hilbert_indices;
use least_squares::LeastSquaresGradient;
pub use mesh_2d::{Mesh2d, nonuniform_rectangle_mesh, rectangle_mesh};
pub use mesh_3d::{Mesh3d, box_mesh, nonuniform_box_mesh};
use partition::Partitioner;
pub(crate) use simplices::{EDGE_FACES, TETRA_FACES, TRIANGLE_FACES};
pub use simplices::{Simplex, get_face_to_elem};
use split::{split_edgs, split_tets, split_tris};
pub use to_simplices::{hex2tets, pri2tets, pyr2tets, qua2tris};

use crate::{
    Error, Result, Tag, Vertex,
    graph::CSRGraph,
    io::{VTUEncoding, VTUFile},
    spatialindex::PointIndex,
};
use log::debug;
use minimeshb::{reader::MeshbReader, writer::MeshbWriter};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};

/// Compute the center of a cell
#[must_use]
pub fn cell_center<const D: usize, const N: usize>(v: &[Vertex<D>; N]) -> Vertex<D> {
    let res = v.iter().copied().sum::<Vertex<D>>();
    (1.0 / N as f64) * res
}

/// Compute a cell point based on its barycentric coordinates
#[must_use]
pub fn cell_vertex<const D: usize, const N: usize>(
    v: &[Vertex<D>; N],
    bcoords: [f64; N],
) -> Vertex<D> {
    bcoords.iter().zip(v.iter()).map(|(&w, v)| w * v).sum()
}

pub(crate) fn sort_elem_min_ids<const C: usize, I: ExactSizeIterator<Item = Cell<C>>>(
    elems: I,
) -> Vec<usize> {
    let n_elems = elems.len();

    let min_ids = elems.map(|e| e.iter().copied().min()).collect::<Vec<_>>();
    let mut indices = (0..n_elems).collect::<Vec<_>>();
    indices.sort_by_key(|&i| min_ids[i]);
    indices
}

/// Compute the maximum and average bandwidth of a connectivity
pub fn bandwidth<const C: usize, I: ExactSizeIterator<Item = Cell<C>>>(elems: I) -> (usize, f64) {
    let n_elems = elems.len();

    let (bmax, bmean) = elems.fold((0, 0), |a, e| {
        let max_id = e.iter().copied().max().unwrap();
        let min_id = e.iter().copied().min().unwrap();
        let tmp = max_id - min_id;
        (a.0.max(tmp), a.1 + tmp)
    });
    let bmean = bmean as f64 / n_elems as f64;
    (bmax, bmean)
}

/// Collect elements into a fixed shape
pub fn collect_elems<
    'a,
    const C0: usize,
    const C1: usize,
    I: ExactSizeIterator<Item = &'a Cell<C0>>,
>(
    elems: I,
) -> Vec<Cell<C1>> {
    assert_eq!(C0, C1);
    let mut res = Vec::with_capacity(elems.len());
    let mut new = [0; C1];
    for e in elems {
        new.copy_from_slice(e);
        res.push(new);
    }
    res
}

/// Submesh of a `Mesh<D, C, F>`, with information about the vertices, element
/// and face ids in the parent mesh
pub struct SubMesh<const D: usize, const C: usize, const F: usize, M: Mesh<D, C, F>>
where
    Cell<C>: Simplex<C>,
    Cell<F>: Simplex<F>,
{
    /// Mesh
    pub mesh: M,
    /// Indices of the vertices of `mesh` in the parent mesh
    pub parent_vert_ids: Vec<usize>,
    /// Indices of the element of `mesh` in the parent mesh
    pub parent_elem_ids: Vec<usize>,
    /// Indices of the faces of `mesh` in the parent mesh
    pub parent_face_ids: Vec<usize>,
}

impl<const D: usize, const C: usize, const F: usize, M: Mesh<D, C, F>> SubMesh<D, C, F, M>
where
    Cell<C>: Simplex<C>,
    Cell<F>: Simplex<F>,
{
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

/// D-dimensional mesh containing simplices with C nodes
/// F = C-1 is given explicitely to be usable with rust stable
pub trait Mesh<const D: usize, const C: usize, const F: usize>: Send + Sync + Sized
where
    Cell<C>: Simplex<C>,
    Cell<F>: Simplex<F>,
{
    /// Create a new mesh from slices of Vertex, Cell and Face
    #[must_use]
    fn new(
        verts: &[Vertex<D>],
        elems: &[Cell<C>],
        etags: &[Tag],
        faces: &[Face<F>],
        ftags: &[Tag],
    ) -> Self {
        let mut res = Self::empty();
        res.add_verts(verts.iter().copied());
        res.add_elems(elems.iter().copied(), etags.iter().copied());
        res.add_faces(faces.iter().copied(), ftags.iter().copied());
        res
    }

    /// Get a vector of faces ( arrays of size C - 1) of the element (0, .., C-1)
    /// If faces can be oriented, they are oriented outwards
    #[must_use]
    fn elem_to_faces() -> Vec<Face<F>> {
        match C {
            4 => TETRA_FACES
                .iter()
                .map(|x| x.as_slice().try_into().unwrap())
                .collect(),
            3 => TRIANGLE_FACES
                .iter()
                .map(|x| x.as_slice().try_into().unwrap())
                .collect(),
            2 => EDGE_FACES
                .iter()
                .map(|x| x.as_slice().try_into().unwrap())
                .collect(),
            _ => unreachable!(),
        }
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
    fn add_verts<I: ExactSizeIterator<Item = Vertex<D>>>(&mut self, v: I);

    /// Number of elements
    fn n_elems(&self) -> usize;

    /// Get the `i`th element
    fn elem(&self, i: usize) -> Cell<C>;

    /// Invert the `i`th element
    fn invert_elem(&mut self, i: usize);

    /// Parallel iterator over the mesh elements
    fn par_elems(&self) -> impl IndexedParallelIterator<Item = Cell<C>> + Clone + '_;

    /// Sequential iterator over the mesh elements
    fn elems(&self) -> impl ExactSizeIterator<Item = Cell<C>> + Clone + '_;

    /// Add elements to the mesh
    fn add_elems<I1: ExactSizeIterator<Item = Cell<C>>, I2: ExactSizeIterator<Item = Tag>>(
        &mut self,
        elems: I1,
        etags: I2,
    );

    /// Remove all the elements
    fn clear_elems(&mut self);

    /// Add elements to the mesh
    fn add_elems_and_tags<I: ExactSizeIterator<Item = (Cell<C>, Tag)>>(
        &mut self,
        elems_and_tags: I,
    );

    /// Get the tag of the `i`th element
    fn etag(&self, i: usize) -> Tag;

    /// Parallel iterator over the element tags
    fn par_etags(&self) -> impl IndexedParallelIterator<Item = Tag> + Clone + '_;

    /// Sequential iterator over the element tags
    fn etags(&self) -> impl ExactSizeIterator<Item = Tag> + Clone + '_;

    /// Get the vertices of element `e`
    fn gelem(&self, e: &Cell<C>) -> [Vertex<D>; C] {
        let mut res = [self.vert(0); C];

        for (j, &k) in e.iter().enumerate() {
            res[j] = self.vert(k);
        }
        res
    }

    /// Parallel iterator over element vertices
    fn par_gelems(&self) -> impl IndexedParallelIterator<Item = [Vertex<D>; C]> + Clone + '_ {
        self.par_elems().map(|e| self.gelem(&e))
    }

    /// Sequential iterator over element vertices
    fn gelems(&self) -> impl ExactSizeIterator<Item = [Vertex<D>; C]> + Clone + '_ {
        self.elems().map(|e| self.gelem(&e))
    }

    /// Number of faces
    fn n_faces(&self) -> usize;

    /// Get the `i`th face
    fn face(&self, i: usize) -> Face<F>;

    /// Invert the `i`th face
    fn invert_face(&mut self, i: usize);

    /// Parallel iterator over the faces
    fn par_faces(&self) -> impl IndexedParallelIterator<Item = Face<F>> + Clone + '_;

    /// Sequential itertor over the faces
    fn faces(&self) -> impl ExactSizeIterator<Item = Face<F>> + Clone + '_;
    /// Add faces to the mesh
    fn add_faces<I1: ExactSizeIterator<Item = Face<F>>, I2: ExactSizeIterator<Item = Tag>>(
        &mut self,
        faces: I1,
        ftags: I2,
    );

    /// Clear all the mesh faces
    fn clear_faces(&mut self);

    /// Add faces to the mesh
    fn add_faces_and_tags<I: ExactSizeIterator<Item = (Face<F>, Tag)>>(
        &mut self,
        faces_and_tags: I,
    );

    /// Get the tag of the `i`th face
    fn ftag(&self, i: usize) -> Tag;

    /// Parallel iterator over the mesh faces
    fn par_ftags(&self) -> impl IndexedParallelIterator<Item = Tag> + Clone + '_;

    /// Sequential iterator over the mesh faces
    fn ftags(&self) -> impl ExactSizeIterator<Item = Tag> + Clone + '_;

    /// Get the vertices of face `f`
    fn gface(&self, f: &Face<F>) -> [Vertex<D>; F] {
        let mut res = [self.vert(0); F];
        for j in 0..F {
            res[j] = self.vert(f[j]);
        }
        res
    }

    /// Parallel iterator over face vertices
    fn par_gfaces(&self) -> impl IndexedParallelIterator<Item = [Vertex<D>; F]> + Clone + '_ {
        self.par_faces().map(|f| self.gface(&f))
    }

    /// Sequential iterator over face vertices
    fn gfaces(&self) -> impl ExactSizeIterator<Item = [Vertex<D>; F]> + Clone + '_ {
        self.faces().map(|f| self.gface(&f))
    }

    /// Compute the mesh edges
    /// a map from sorted edge `[i0, i1]` (`i0 < i1`) to the edge index is returned
    fn edges(&self) -> FxHashMap<Edge, usize> {
        let elem_to_edges = Cell::<C>::edges();

        let mut res = FxHashMap::with_hasher(FxBuildHasher);

        for edg in self.elems().flat_map(|e| {
            elem_to_edges
                .iter()
                .map(move |&[i, j]| [e[i], e[j]].sorted())
        }) {
            if !res.contains_key(&edg) {
                res.insert(edg, res.len());
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
        CSRGraph::from_edges(edges.keys().copied(), Some(self.n_verts()))
    }

    /// Check if faces can be oriented for the current values of D and C
    #[must_use]
    fn faces_are_oriented() -> bool {
        F == D
    }

    /// Compute all the mesh faces (boundary & internal)
    /// a map from sorted face `[i0, i1, ...]` (`i0 < i1 < ...`) to the face index and face elements
    /// (`e0` and `e1`) is returned.
    /// If the faces can be oriented, it is oriented outwards for `e0` and inwards for `e1`
    /// If the faces only belongs to one element, `i1 = usize::MAX`
    fn all_faces(&self) -> FxHashMap<Face<F>, [usize; 3]> {
        let mut res: FxHashMap<Face<F>, [usize; 3]> = FxHashMap::with_hasher(FxBuildHasher);
        let mut idx = 0;
        let elem_to_faces = Self::elem_to_faces();

        for (i_elem, e) in self.elems().enumerate() {
            for face in &elem_to_faces {
                let mut tmp = [0; F];
                for (i, &j) in face.iter().enumerate() {
                    tmp[i] = e[j];
                }
                if Self::faces_are_oriented() {
                    let n = Face::<F>::normal(&self.gface(&tmp));
                    tmp.sort_unstable();
                    let n_ref = Face::<F>::normal(&self.gface(&tmp));
                    let is_dot_pos = n.dot(&n_ref) > 0.0;

                    if let Some(arr) = res.get_mut(&tmp) {
                        if is_dot_pos {
                            arr[1] = i_elem;
                        } else {
                            assert_eq!(arr[2], usize::MAX);
                            arr[2] = i_elem;
                        }
                    } else {
                        if is_dot_pos {
                            res.insert(tmp, [idx, i_elem, usize::MAX]);
                        } else {
                            res.insert(tmp, [idx, usize::MAX, i_elem]);
                        }
                        idx += 1;
                    }
                } else {
                    tmp.sort_unstable();
                    if let Some(arr) = res.get_mut(&tmp) {
                        assert_eq!(arr[2], usize::MAX);
                        arr[2] = i_elem;
                    } else {
                        res.insert(tmp, [idx, i_elem, usize::MAX]);
                        idx += 1;
                    }
                }
            }
        }
        res
    }

    /// Compute element pairs corresponding to all the internal faces (for partitioning)
    fn element_pairs(&self, faces: &FxHashMap<Face<F>, [usize; 3]>) -> CSRGraph {
        let e2e = faces
            .iter()
            .map(|(_, &[_, i0, i1])| [i0, i1])
            .filter(|&[i0, i1]| i0 != usize::MAX && i1 != usize::MAX)
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
            .map(|e| Cell::<C>::vol(&self.gelem(&e)) < 0.0)
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
    fn fix_faces_orientation(&mut self, all_faces: &FxHashMap<Face<F>, [usize; 3]>) -> usize {
        if Self::faces_are_oriented() {
            let flg = self
                .faces()
                .map(|f| {
                    let gf = self.gface(&f);
                    let fc = cell_center(&gf);
                    let normal = Face::<F>::normal(&gf);

                    let [_, i0, i1] = all_faces.get(&f.sorted()).unwrap();
                    let i = if *i0 == usize::MAX || *i1 == usize::MAX {
                        if *i1 == usize::MAX { *i0 } else { *i1 }
                    } else {
                        let t0 = self.etag(*i0);
                        let t1 = self.etag(*i1);
                        assert_ne!(t0, t1);
                        if t0 < t1 { *i0 } else { *i1 }
                    };
                    let ge = self.gelem(&self.elem(i));
                    let ec = cell_center(&ge);

                    normal.dot(&(fc - ec)) < 0.0
                })
                .collect::<Vec<_>>();

            let n = flg
                .iter()
                .enumerate()
                .filter(|(_, f)| **f)
                .map(|(i, _)| self.invert_face(i))
                .count();
            debug!("{n} faces reoriented");
            return n;
        }
        0
    }

    /// Compute the faces that are connected to only one element and that are not already tagged
    fn tag_boundary_faces(
        &mut self,
        all_faces: &FxHashMap<Face<F>, [usize; 3]>,
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
            if i0 == usize::MAX && !tagged_faces.contains_key(f) {
                let etag = self.etag(i1);
                if let Some(&tmp) = res.get(&etag) {
                    self.add_faces(std::iter::once(f).copied(), std::iter::once(tmp));
                } else {
                    res.insert(etag, next_tag);
                    self.add_faces(std::iter::once(f).copied(), std::iter::once(next_tag));
                    next_tag -= 1;
                }
            }
            if i1 == usize::MAX && !tagged_faces.contains_key(f) {
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
        all_faces: &FxHashMap<Face<F>, [usize; 3]>,
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
    fn check(&self, all_faces: &FxHashMap<Face<F>, [usize; 3]>) -> Result<()> {
        // lengths
        if self.par_elems().len() != self.par_etags().len() {
            return Err(Error::from("Inconsistent sizes (elems)"));
        }
        if self.par_faces().len() != self.par_ftags().len() {
            return Err(Error::from("Inconsistent sizes (faces)"));
        }

        // indices & element volume
        for e in self.elems() {
            if !e.iter().all(|&i| i < self.n_verts()) {
                return Err(Error::from("Invalid index in elems"));
            }
            let ge = self.gelem(&e);
            if Cell::<C>::vol(&ge) < 0.0 {
                return Err(Error::from("Elem has a <0 volume"));
            }
        }
        for f in self.faces() {
            if !f.iter().all(|&i| i < self.n_verts()) {
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

        for f in self.faces() {
            let gf = self.gface(&f);
            let fc = cell_center(&gf);
            let tmp = f.sorted();
            let [_, i0, i1] = all_faces.get(&tmp).unwrap();
            if *i0 != usize::MAX && *i1 != usize::MAX && self.etag(*i0) == self.etag(*i1) {
                return Err(Error::from(&format!(
                    "Tagged face inside the domain: center = {fc:?}",
                )));
            } else if *i0 == usize::MAX || *i1 == usize::MAX {
                let i = if *i1 == usize::MAX { *i0 } else { *i1 };
                let ge = self.gelem(&self.elem(i));
                let ec = cell_center(&ge);
                if Self::faces_are_oriented() {
                    let n = Face::<F>::normal(&gf);
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
            let vol = self.par_gelems().map(|ge| Cell::<C>::vol(&ge)).sum::<f64>();
            let vol2 = self
                .par_faces()
                .filter(|f| {
                    let f = f.sorted();
                    let [_, i0, i1] = all_faces.get(&f).unwrap();
                    *i0 == usize::MAX || *i1 == usize::MAX
                })
                .map(|f| {
                    let gf = self.gface(&f);
                    cell_center(&gf).dot(&Face::<F>::normal(&gf))
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
        v2v: &CSRGraph,
        weight: i32,
        f: &[f64],
    ) -> impl IndexedParallelIterator<Item = Vertex<D>>
    where
        nalgebra::Const<D>: nalgebra::Dim,
    {
        (0..self.n_verts()).into_par_iter().map(move |i| {
            let x = self.vert(i);
            let neighbors = v2v.row(i);
            let dx = neighbors.iter().map(|&j| self.vert(j) - x);
            let ls = LeastSquaresGradient::new(weight, dx).unwrap();
            let df = neighbors.iter().map(|&j| f[j] - f[i]);
            ls.gradient(df)
        })
    }

    /// Integrate `g(f)` over the mesh, where `f` is a field defined on the mesh vertices
    fn integrate<G: Fn(f64) -> f64 + Send + Sync>(&self, f: &[f64], op: G) -> f64 {
        let (qw, qp) = Cell::<C>::quadrature();
        debug_assert!(qp.iter().all(|x| x.len() == C - 1));

        self.par_elems()
            .map(|e| {
                let res = qw
                    .iter()
                    .zip(qp.iter())
                    .map(|(w, pt)| {
                        let x0 = f[e[0]];
                        let mut x_pt = x0;
                        for (&b, &j) in pt.iter().zip(e.iter().skip(1)) {
                            let mut dx = f[j] - x0;
                            dx *= b;
                            x_pt += dx;
                        }
                        *w * op(x_pt)
                    })
                    .sum::<f64>();
                Cell::<C>::vol(&self.gelem(&e)) * res
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
        let new_elems = self.elems().map(|mut e| {
            for idx in &mut e {
                *idx = new_vert_indices[*idx];
            }
            e
        });
        let new_faces = self.faces().map(|mut f| {
            for idx in &mut f {
                *idx = new_vert_indices[*idx];
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

    /// Set the element tags
    fn set_etags<I: ExactSizeIterator<Item = Tag>>(&mut self, tags: I);

    /// Set the partition as etags from an usize slice
    fn set_partition(&mut self, part: &[usize]) {
        assert_eq!(self.n_elems(), part.len());
        self.set_etags(part.iter().map(|&x| x as Tag + 1));
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
    fn get_partition(&self, i: usize) -> SubMesh<D, C, F, Self> {
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

        match C {
            4 => res.add_elems_and_tags(reader.read_tetrahedra()?.map(|(e, t)| {
                let mut tmp = [0; C];
                tmp.copy_from_slice(&e);
                (tmp, t as Tag)
            })),
            3 => res.add_elems_and_tags(reader.read_triangles()?.map(|(e, t)| {
                let mut tmp = [0; C];
                tmp.copy_from_slice(&e);
                (tmp, t as Tag)
            })),
            2 => res.add_elems_and_tags(reader.read_edges()?.map(|(e, t)| {
                let mut tmp = [0; C];
                tmp.copy_from_slice(&e);
                (tmp, t as Tag)
            })),
            _ => unimplemented!(),
        }

        match F {
            3 => res.add_faces_and_tags(reader.read_triangles()?.map(|(e, t)| {
                let mut tmp = [0; F];
                tmp.copy_from_slice(&e);
                (tmp, t as Tag)
            })),
            2 => res.add_faces_and_tags(reader.read_edges()?.map(|(e, t)| {
                let mut tmp = [0; F];
                tmp.copy_from_slice(&e);
                (tmp, t as Tag)
            })),
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
                tmp.copy_from_slice(x.as_slice());
                tmp
            }),
            (0..self.n_verts()).map(|_| 1),
        )?;

        match C {
            4 => writer.write_tetrahedra(
                self.elems().map(|x| {
                    let mut tmp = [0; 4];
                    tmp.copy_from_slice(x.as_slice());
                    tmp
                }),
                self.etags().map(|x| x.try_into().unwrap()),
            )?,
            3 => writer.write_triangles(
                self.elems().map(|x| {
                    let mut tmp = [0; 3];
                    tmp.copy_from_slice(x.as_slice());
                    tmp
                }),
                self.etags().map(|x| x.try_into().unwrap()),
            )?,
            2 => writer.write_edges(
                self.elems().map(|x| {
                    let mut tmp = [0; 2];
                    tmp.copy_from_slice(x.as_slice());
                    tmp
                }),
                self.etags().map(|x| x.try_into().unwrap()),
            )?,
            _ => unimplemented!(),
        }

        match F {
            3 => writer.write_triangles(
                self.faces().map(|x| {
                    let mut tmp = [0; 3];
                    tmp.copy_from_slice(x.as_slice());
                    tmp
                }),
                self.ftags().map(|x| x.try_into().unwrap()),
            )?,
            2 => writer.write_edges(
                self.faces().map(|x| {
                    let mut tmp = [0; 2];
                    tmp.copy_from_slice(x.as_slice());
                    tmp
                }),
                self.ftags().map(|x| x.try_into().unwrap()),
            )?,
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

    /// Build a `Mesh<D, C2, F2>` mesh containing faces such that `filter(tag)` is true
    /// Only the required vertices are present
    /// The following must be true: C2 = C - 1 and F2 = F - 1 (rust stable limitation)
    fn extract_faces<const C2: usize, const F2: usize, M: Mesh<D, C2, F2>, G: Fn(Tag) -> bool>(
        &self,
        filter: G,
    ) -> (M, Vec<usize>)
    where
        Cell<C2>: Simplex<C2>,
        Cell<F2>: Simplex<F2>,
    {
        assert_eq!(C2, C - 1);
        assert_eq!(F2, F - 1);
        let mut new_ids = vec![usize::MAX; self.n_verts()];
        let mut vert_ids = Vec::new();
        let mut next = 0;

        let n_faces = self
            .faces()
            .zip(self.ftags())
            .filter(|(_, t)| filter(*t))
            .map(|(f, _)| {
                for &i in &f {
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
            .filter(|(f, _)| f.iter().all(|&i| new_ids[i] != usize::MAX))
            .for_each(|(f, t)| {
                faces.push(std::array::from_fn(|i| new_ids[f[i]]));
                ftags.push(t);
            });

        let mut res = M::empty();
        res.add_verts(verts.iter().copied());
        res.add_elems(faces.iter().copied(), ftags.iter().copied());

        (res, vert_ids)
    }

    /// Build a `Mesh<D, C2, F2>` mesh containing the boundary faces
    /// The following must be true: C2 = C - 1 and F2 = F - 1 (rust stable limitation)
    fn boundary<const C2: usize, const F2: usize, M: Mesh<D, C2, F2>>(&self) -> (M, Vec<usize>)
    where
        Cell<C2>: Simplex<C2>,
        Cell<F2>: Simplex<F2>,
    {
        self.extract_faces(|_| true)
    }

    /// Split quandrangles and add them to the mesh (C == 3 or F == 3)
    fn add_quadrangles<
        I1: ExactSizeIterator<Item = Quadrangle>,
        I2: ExactSizeIterator<Item = Tag>,
    >(
        &mut self,
        quads: I1,
        tags: I2,
    ) {
        if C == 3 {
            let mut tmp = Vec::with_capacity(2 * quads.len());
            quads.zip(tags).for_each(|(q, t)| {
                let tris = qua2tris(&q);
                for tri in tris {
                    let mut tmp_tri = [0; C];
                    tmp_tri.copy_from_slice(&tri);
                    tmp.push((tmp_tri, t));
                }
            });
            self.add_elems_and_tags(tmp.iter().copied());
        } else if F == 3 {
            let mut tmp = Vec::with_capacity(2 * quads.len());
            quads.zip(tags).for_each(|(q, t)| {
                let tris = qua2tris(&q);
                for tri in tris {
                    let mut tmp_tri = [0; F];
                    tmp_tri.copy_from_slice(&tri);
                    tmp.push((tmp_tri, t));
                }
            });
            self.add_faces_and_tags(tmp.iter().copied());
        } else {
            unreachable!()
        }
    }

    /// Split hexahedra and add them to the mesh (C == 4)
    fn add_hexahedra<
        I1: ExactSizeIterator<Item = Hexahedron>,
        I2: ExactSizeIterator<Item = Tag>,
    >(
        &mut self,
        hexs: I1,
        tags: I2,
    ) -> Vec<usize> {
        if C == 4 {
            let mut tmp = Vec::with_capacity(6 * hexs.len());
            let mut ids = Vec::with_capacity(6 * hexs.len());
            hexs.zip(tags).enumerate().for_each(|(i, (q, t))| {
                let (tets, last_tet) = hex2tets(&q);
                for tet in tets {
                    let mut tmp_tet = [0; C];
                    tmp_tet.copy_from_slice(&tet);
                    tmp.push((tmp_tet, t));
                    ids.push(i);
                }
                if let Some(last_tet) = last_tet {
                    let mut tmp_tet = [0; C];
                    tmp_tet.copy_from_slice(&last_tet);
                    tmp.push((tmp_tet, t));
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
    fn add_prisms<I1: ExactSizeIterator<Item = Prism>, I2: ExactSizeIterator<Item = Tag>>(
        &mut self,
        pris: I1,
        tags: I2,
    ) {
        if C == 4 {
            let mut tmp = Vec::with_capacity(3 * pris.len());
            pris.zip(tags).for_each(|(q, t)| {
                let tets = pri2tets(&q);
                for tet in tets {
                    let mut tmp_tet = [0; C];
                    tmp_tet.copy_from_slice(&tet);
                    tmp.push((tmp_tet, t));
                }
            });
            self.add_elems_and_tags(tmp.iter().copied());
        } else {
            unreachable!()
        }
    }

    /// Split pyramids and add them to the mesh (C == 4)
    fn add_pyramids<I1: ExactSizeIterator<Item = Pyramid>, I2: ExactSizeIterator<Item = Tag>>(
        &mut self,
        pyrs: I1,
        tags: I2,
    ) {
        if C == 4 {
            let mut tmp = Vec::with_capacity(3 * pyrs.len());
            pyrs.zip(tags).for_each(|(q, t)| {
                let tets = pyr2tets(&q);
                for tet in tets {
                    let mut tmp_tet = [0; C];
                    tmp_tet.copy_from_slice(&tet);
                    tmp.push((tmp_tet, t));
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
    fn check_equals<M: Mesh<D, C, F>>(&self, other: &M, tol: f64) -> Result<()> {
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
            verts[i] = 0.5 * (p0 + p1);
        }
        res.add_verts(verts.iter().copied());

        // add offset to verts
        for v in edges.values_mut() {
            *v += self.n_verts();
        }

        // Cells
        match C {
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
        match F {
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
        all_faces: &FxHashMap<Face<F>, [usize; 3]>,
    ) -> impl Iterator<Item = (usize, usize, f64)> {
        all_faces
            .iter()
            .filter(|&(_, [_, i0, i1])| *i0 != usize::MAX && *i1 != usize::MAX)
            .map(|(f, &[_, i0, i1])| {
                let fc = cell_center(&self.gface(f));
                let ec0 = cell_center(&self.gelem(&self.elem(i0)));
                let ec1 = cell_center(&self.gelem(&self.elem(i1)));
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
        let elem_to_edges = Cell::<C>::edges();
        self.elems().map(move |e| {
            let mut l_min = f64::MAX;
            let mut l_max = 0.0_f64;
            for &[i0, i1] in &elem_to_edges {
                let l = (self.vert(e[i1]) - self.vert(e[i0])).norm();
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
        self.gelems().map(|ge| Cell::<C>::gamma(&ge))
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
                for &i in &e {
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
        let elem_to_faces = Self::elem_to_faces();
        for (i, (mut e, t)) in other
            .elems()
            .zip(other.etags())
            .enumerate()
            .filter(|(_, (_, t))| elem_filter(*t))
        {
            added_elems.push(i);
            for i in &mut e {
                *i = new_vert_ids[*i];
            }
            self.add_elems(std::iter::once(e), std::iter::once(t));
            for face in &elem_to_faces {
                let mut tmp = [0; F];
                for (i, &j) in face.iter().enumerate() {
                    tmp[i] = e[j];
                }
                all_added_faces.insert(tmp.sorted());
            }
        }

        let mut added_faces = Vec::new();
        for (i, (mut f, t)) in other
            .faces()
            .zip(other.ftags())
            .enumerate()
            .filter(|(_, (_, t))| face_filter(*t))
            .filter(|&(_, (mut f, _))| {
                for i in &mut f {
                    *i = new_vert_ids[*i];
                }
                all_added_faces.contains(&f.sorted())
            })
        {
            added_faces.push(i);
            for i in &mut f {
                *i = new_vert_ids[*i];
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
pub trait MutMesh<const D: usize, const C: usize, const F: usize>: Mesh<D, C, F>
where
    Cell<C>: Simplex<C>,
    Cell<F>: Simplex<F>,
{
    /// Sequential iterator over the vertices
    fn verts_mut(&mut self) -> impl ExactSizeIterator<Item = &mut Vertex<D>> + '_;

    /// Sequential iterator over the mesh elements
    fn elems_mut(&mut self) -> impl ExactSizeIterator<Item = &mut Cell<C>> + '_;

    /// Sequential iterator over the element tags
    fn etags_mut(&mut self) -> impl ExactSizeIterator<Item = &mut Tag> + '_;

    /// Sequential itertor over the faces
    fn faces_mut(&mut self) -> impl ExactSizeIterator<Item = &mut Face<F>> + '_;

    /// Sequential iterator over the mesh faces
    fn ftags_mut(&mut self) -> impl ExactSizeIterator<Item = &mut Tag> + '_;
}

/// Generic meshes implemented with Vecs
pub struct GenericMesh<const D: usize, const C: usize, const F: usize> {
    verts: Vec<Vertex<D>>,
    elems: Vec<Cell<C>>,
    etags: Vec<Tag>,
    faces: Vec<Face<F>>,
    ftags: Vec<Tag>,
}

impl<const D: usize, const C: usize, const F: usize> Mesh<D, C, F> for GenericMesh<D, C, F>
where
    Cell<C>: Simplex<C>,
    Face<F>: Simplex<F>,
{
    fn empty() -> Self {
        Self {
            verts: Vec::new(),
            elems: Vec::new(),
            etags: Vec::new(),
            faces: Vec::new(),
            ftags: Vec::new(),
        }
    }

    fn n_verts(&self) -> usize {
        self.verts.len()
    }

    fn vert(&self, i: usize) -> Vertex<D> {
        self.verts[i]
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

    fn n_elems(&self) -> usize {
        self.elems.len()
    }

    fn elem(&self, i: usize) -> Cell<C> {
        self.elems[i]
    }

    fn invert_elem(&mut self, i: usize) {
        self.elems[i].invert();
    }

    fn elems(&self) -> impl ExactSizeIterator<Item = Cell<C>> + Clone + '_ {
        self.elems.iter().copied()
    }

    fn par_elems(&self) -> impl IndexedParallelIterator<Item = Cell<C>> + Clone + '_ {
        self.elems.par_iter().cloned()
    }

    fn etag(&self, i: usize) -> Tag {
        self.etags[i]
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

    fn add_elems<I1: ExactSizeIterator<Item = Cell<C>>, I2: ExactSizeIterator<Item = Tag>>(
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

    fn add_elems_and_tags<I: ExactSizeIterator<Item = (Cell<C>, Tag)>>(
        &mut self,
        elems_and_tags: I,
    ) {
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

    fn face(&self, i: usize) -> Face<F> {
        self.faces[i]
    }

    fn invert_face(&mut self, i: usize) {
        self.faces[i].invert();
    }

    fn faces(&self) -> impl ExactSizeIterator<Item = Face<F>> + Clone + '_ {
        self.faces.iter().copied()
    }

    fn par_faces(&self) -> impl IndexedParallelIterator<Item = Face<F>> + Clone + '_ {
        self.faces.par_iter().cloned()
    }

    fn ftag(&self, i: usize) -> Tag {
        self.ftags[i]
    }

    fn ftags(&self) -> impl ExactSizeIterator<Item = Tag> + Clone + '_ {
        self.ftags.iter().copied()
    }

    fn par_ftags(&self) -> impl IndexedParallelIterator<Item = Tag> + Clone + '_ {
        self.ftags.par_iter().cloned()
    }

    fn add_faces<I1: ExactSizeIterator<Item = Face<F>>, I2: ExactSizeIterator<Item = Tag>>(
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

    fn add_faces_and_tags<I: ExactSizeIterator<Item = (Face<F>, Tag)>>(
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

impl<const D: usize, const C: usize, const F: usize> MutMesh<D, C, F> for GenericMesh<D, C, F>
where
    Cell<C>: Simplex<C>,
    Face<F>: Simplex<F>,
{
    fn verts_mut(&mut self) -> impl ExactSizeIterator<Item = &mut Vertex<D>> + '_ {
        self.verts.iter_mut()
    }

    fn elems_mut(&mut self) -> impl ExactSizeIterator<Item = &mut Cell<C>> + '_ {
        self.elems.iter_mut()
    }

    fn etags_mut(&mut self) -> impl ExactSizeIterator<Item = &mut Tag> + '_ {
        self.etags.iter_mut()
    }

    fn faces_mut(&mut self) -> impl ExactSizeIterator<Item = &mut Face<F>> + '_ {
        self.faces.iter_mut()
    }

    fn ftags_mut(&mut self) -> impl ExactSizeIterator<Item = &mut Tag> + '_ {
        self.ftags.iter_mut()
    }
}
