//! Simplex meshes in D dimensions, represented by
//!   - the vertices
//!   - elements of type `Cell<C>` and element tags
//!   - faces of type `Face<F>` and element tags
//!
//! F = C-1 cannot be imposed in rust stable
use crate::least_squares::LeastSquaresGradient;
use crate::simplices::{Simplex, EDGE_FACES, TETRA_FACES, TRIANGLE_FACES};
use crate::to_simplices::{hex2tets, pri2tets, pyr2tets, qua2tris};
use crate::vtu_output::{Encoding, VTUFile};
use crate::{graph::CSRGraph, Cell, Edge, Error, Face, Result, Tag, Vertex};
use crate::{Hexahedron, Quadrangle};
use log::debug;
use minimeshb::reader::MeshbReader;
use minimeshb::writer::MeshbWriter;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};

pub(crate) fn cell_center<const D: usize, const N: usize>(v: [&Vertex<D>; N]) -> Vertex<D> {
    let res = v.iter().cloned().sum::<Vertex<D>>();
    (1.0 / N as f64) * res
}

pub(crate) fn cell_vertex<const D: usize, const N: usize>(
    v: [&Vertex<D>; N],
    bcoords: [f64; N],
) -> Vertex<D> {
    bcoords.iter().zip(v.iter()).map(|(&w, &&v)| w * v).sum()
}

pub(crate) fn sort_elem_min_ids<const C: usize, I: ExactSizeIterator<Item = [usize; C]>>(
    elems: I,
) -> Vec<usize> {
    let n_elems = elems.len();

    let min_ids = elems.map(|e| e.iter().cloned().min()).collect::<Vec<_>>();
    let mut indices = (0..n_elems).collect::<Vec<_>>();
    indices.sort_by_key(|&i| min_ids[i]);
    indices
}

/// Compute the maximum and average bandwidth of a connectivity
pub fn bandwidth<const C: usize, I: ExactSizeIterator<Item = [usize; C]>>(
    elems: I,
) -> (usize, f64) {
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

/// D-dimensional mesh containing simplices with C nodes
/// F = C-1 is given explicitely to be usable with rust stable
pub trait Mesh<const D: usize, const C: usize, const F: usize>: Sync + Sized
where
    Cell<C>: Simplex<C>,
    Cell<F>: Simplex<F>,
{
    /// Get a vector of faces ( arrays of size C - 1) of the element (0, .., C-1)
    /// If faces can be oriented, they are oriented outwards
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
    fn vert(&self, i: usize) -> &Vertex<D>;

    /// Parallel iterator over the vertices
    fn par_verts(&self) -> impl IndexedParallelIterator<Item = &Vertex<D>> + Clone + '_ {
        (0..self.n_verts()).into_par_iter().map(|i| self.vert(i))
    }

    /// Sequential iterator over the vertices
    fn verts(&self) -> impl ExactSizeIterator<Item = &Vertex<D>> + Clone + '_ {
        (0..self.n_verts()).map(|i| self.vert(i))
    }

    /// Add vertices to the mesh
    fn add_verts<I: ExactSizeIterator<Item = Vertex<D>>>(&mut self, v: I);

    /// Number of elements
    fn n_elems(&self) -> usize;

    /// Get the `i`th element
    fn elem(&self, i: usize) -> &Cell<C>;

    /// Parallel iterator over the mesh elements
    fn par_elems(&self) -> impl IndexedParallelIterator<Item = &Cell<C>> + Clone + '_ {
        (0..self.n_elems()).into_par_iter().map(|i| self.elem(i))
    }

    /// Sequential iterator over the mesh elements
    fn elems(&self) -> impl ExactSizeIterator<Item = &Cell<C>> + Clone + '_ {
        (0..self.n_elems()).map(|i| self.elem(i))
    }

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

    /// Invert the `i`th element
    fn invert_elem(&mut self, i: usize);

    /// Get the tag of the `i`th element
    fn etag(&self, i: usize) -> Tag;

    /// Parallel iterator over the element tags
    fn par_etags(&self) -> impl IndexedParallelIterator<Item = Tag> + Clone + '_ {
        (0..self.n_elems()).into_par_iter().map(|i| self.etag(i))
    }

    /// Sequential iterator over the element tags
    fn etags(&self) -> impl ExactSizeIterator<Item = Tag> + Clone + '_ {
        (0..self.n_elems()).map(|i| self.etag(i))
    }

    /// Get the vertices of element `e`
    fn gelem(&self, e: &Cell<C>) -> [&Vertex<D>; C] {
        let mut res = [self.vert(0); C];

        for (j, &k) in e.iter().enumerate() {
            res[j] = self.vert(k);
        }
        res
    }

    /// Parallel iterator over element vertices
    fn par_gelems(&self) -> impl IndexedParallelIterator<Item = [&Vertex<D>; C]> + Clone + '_ {
        self.par_elems().map(|e| self.gelem(e))
    }

    /// Sequential iterator over element vertices
    fn gelems(&self) -> impl ExactSizeIterator<Item = [&Vertex<D>; C]> + Clone + '_ {
        self.elems().map(|e| self.gelem(e))
    }

    /// Number of faces
    fn n_faces(&self) -> usize;

    /// Get the `i`th face
    fn face(&self, i: usize) -> &Face<F>;

    /// Parallel iterator over the faces
    fn par_faces(&self) -> impl IndexedParallelIterator<Item = &Face<F>> + Clone + '_ {
        (0..self.n_faces()).into_par_iter().map(|i| self.face(i))
    }

    /// Sequential itertor over the faces
    fn faces(&self) -> impl ExactSizeIterator<Item = &Face<F>> + Clone + '_ {
        (0..self.n_faces()).map(|i| self.face(i))
    }

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

    /// Invert the `i`th face
    fn invert_face(&mut self, i: usize);

    /// Get the tag of the `i`th face
    fn ftag(&self, i: usize) -> Tag;

    /// Parallel iterator over the mesh faces
    fn par_ftags(&self) -> impl IndexedParallelIterator<Item = Tag> + Clone + '_ {
        (0..self.n_faces()).into_par_iter().map(|i| self.ftag(i))
    }

    /// Sequential iterator over the mesh faces
    fn ftags(&self) -> impl ExactSizeIterator<Item = Tag> + Clone + '_ {
        (0..self.n_faces()).map(|i| self.ftag(i))
    }

    /// Get the vertices of face `f`
    fn gface(&self, f: &Face<F>) -> [&Vertex<D>; F] {
        let mut res = [self.vert(0); F];
        for j in 0..F {
            res[j] = self.vert(f[j]);
        }
        res
    }

    /// Parallel iterator over face vertices
    fn par_gfaces(&self) -> impl IndexedParallelIterator<Item = [&Vertex<D>; F]> + Clone + '_ {
        self.par_faces().map(|f| self.gface(f))
    }

    /// Sequential iterator over face vertices
    fn gfaces(&self) -> impl ExactSizeIterator<Item = [&Vertex<D>; F]> + Clone + '_ {
        self.faces().map(|f| self.gface(f))
    }

    /// Compute the mesh edges
    /// a map from sorted edge `[i0, i1]` (`i0 < i1`) to the edge index is returned
    fn compute_edges(&self) -> FxHashMap<Edge, usize> {
        let elem_to_edges = Cell::<C>::edges();

        let mut res = FxHashMap::with_hasher(FxBuildHasher);

        for edg in self
            .elems()
            .flat_map(|e| elem_to_edges.iter().map(|&[i, j]| [e[i], e[j]].sorted()))
        {
            if !res.contains_key(&edg) {
                res.insert(edg, res.len());
            }
        }

        res
    }

    /// Compute the vertex-to-element connectivity
    fn compute_vertex_to_elems(&self) -> CSRGraph {
        CSRGraph::transpose(self.elems())
    }

    /// Compute the vertex-to-vertex connectivity
    fn compute_vertex_to_vertices(&self) -> CSRGraph {
        let edges = self.compute_edges();
        CSRGraph::from_edges(edges.keys())
    }

    /// Check if faces can be oriented for the current values of D and C
    fn faces_are_oriented() -> bool {
        F == D
    }

    /// Compute all the mesh faces (boundary & internal)
    /// a map from sorted face `[i0, i1, ...]` (`i0 < i1 < ...`) to the face index and face elements
    /// (`e0` and `e1`) is returned.
    /// If the faces can be oriented, it is oriented outwards for `e0` and inwards for `e1`
    /// If the faces only belongs to one element, `i1 = usize::MAX`
    fn compute_faces(&self) -> FxHashMap<Face<F>, [usize; 3]> {
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
                    let n = Face::<F>::normal(self.gface(&tmp));
                    tmp.sort();
                    let n_ref = Face::<F>::normal(self.gface(&tmp));
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
                    tmp.sort();
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

    /// Fix the orientation of elements (so that their volume is >0) and of faces
    /// to be oriented outwards (if possible)
    fn fix_orientation(&mut self, all_faces: &FxHashMap<Face<F>, [usize; 3]>) {
        let flg: Vec<_> = self
            .par_elems()
            .map(|e| Cell::<C>::vol(self.gelem(e)) < 0.0)
            .collect();
        flg.iter()
            .enumerate()
            .filter(|(_, &f)| f)
            .for_each(|(i, _)| self.invert_elem(i));

        if Self::faces_are_oriented() {
            let flg: Vec<_> = self
                .par_faces()
                .map(|f| {
                    let gf = self.gface(f);
                    let fc = cell_center(gf);
                    let n = Face::<F>::normal(gf);

                    let f = f.sorted();
                    let [_, i0, i1] = all_faces.get(&f).unwrap();
                    assert!(*i0 == usize::MAX || *i1 == usize::MAX);
                    let i = if *i1 == usize::MAX { *i0 } else { *i1 };
                    let ge = self.gelem(self.elem(i));
                    let ec = cell_center(ge);

                    n.dot(&(fc - ec)) < 0.0
                })
                .collect();
            let n = flg
                .iter()
                .enumerate()
                .filter(|(_, &f)| f)
                .map(|(i, _)| {
                    self.invert_face(i);
                })
                .count();
            debug!("{n} faces reoriented");
        }
    }

    /// Compute the faces that are connected to elements with different tags
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

        let mut next_tag = self.par_ftags().max().unwrap_or(0) + 1;

        // check tagged internal faces
        for (f, [_, i0, i1]) in all_faces {
            if *i0 != usize::MAX && *i1 != usize::MAX {
                let t0 = self.etag(*i0);
                let t1 = self.etag(*i1);
                if t0 != t1 {
                    if let Some(tag) = tagged_faces.get(f) {
                        let tags = if t0 < t1 { [t0, t1] } else { [t1, t0] };
                        if let Some(tmp) = res.get(&tags) {
                            assert_eq!(tag, tmp);
                        }
                    }
                }
            }
        }

        // add untagged internal faces
        for (f, [_, i0, i1]) in all_faces {
            if *i0 != usize::MAX && *i1 != usize::MAX {
                let t0 = self.etag(*i0);
                let t1 = self.etag(*i1);
                if t0 != t1 && !tagged_faces.contains_key(f) {
                    let tags = if t0 < t1 { [t0, t1] } else { [t1, t0] };
                    if let Some(&tmp) = res.get(&tags) {
                        self.add_faces(std::iter::once(f).cloned(), std::iter::once(tmp));
                    } else {
                        res.insert(tags, next_tag);
                        self.add_faces(std::iter::once(f).cloned(), std::iter::once(next_tag));
                        next_tag += 1;
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
            let ge = self.gelem(e);
            if Cell::<C>::vol(ge) < 0.0 {
                return Err(Error::from("Elem has a <0 volume"));
            }
        }
        for f in self.faces() {
            if !f.iter().all(|&i| i < self.n_verts()) {
                return Err(Error::from("Invalid index in faces"));
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
            let gf = self.gface(f);
            let fc = cell_center(gf);
            let tmp = f.sorted();
            let [_, i0, i1] = all_faces.get(&tmp).unwrap();
            if *i0 != usize::MAX && *i1 != usize::MAX && self.etag(*i0) == self.etag(*i1) {
                return Err(Error::from(&format!(
                    "Tagged face inside the domain: center = {fc:?}",
                )));
            }
            let i = if *i1 == usize::MAX { *i0 } else { *i1 };
            let ge = self.gelem(self.elem(i));
            let ec = cell_center(ge);
            if Self::faces_are_oriented() {
                let n = Face::<F>::normal(gf);
                if n.dot(&(fc - ec)) < 0.0 {
                    return Err(Error::from(&format!(
                        "Invalid face orientation: center = {fc:?}"
                    )));
                }
            }
        }

        // volumes
        if Self::faces_are_oriented() {
            let vol = self.par_gelems().map(Cell::<C>::vol).sum::<f64>();
            let vol2 = self
                .par_gfaces()
                .map(|gf| cell_center(gf).dot(&Face::<F>::normal(gf)))
                .sum::<f64>()
                / D as f64;
            if (vol - vol2).abs() > 1e-10 * vol {
                return Err(Error::from("Invalid volume"));
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
            let x = *self.vert(i);
            let neighbors = v2v.row(i);
            let dx = neighbors.iter().map(|&j| *self.vert(j) - x);
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
                Cell::<C>::vol(self.gelem(e)) * res
            })
            .sum::<f64>()
    }

    /// Compute the norm of a field `f` defined on the mesh vertices
    fn norm(&self, f: &[f64]) -> f64 {
        self.integrate(f, |x| x.powi(2)).sqrt()
    }

    /// Reorder the mesh vertices
    fn reorder_vertices(&self, vert_indices: &[usize]) -> Self {
        assert_eq!(vert_indices.len(), self.n_verts());

        let mut new_vert_indices = vec![0; self.n_verts()];
        vert_indices
            .iter()
            .enumerate()
            .for_each(|(i, &new_i)| new_vert_indices[new_i] = i);
        let new_verts = vert_indices.iter().map(|&i| *self.vert(i));
        let new_elems = self.elems().map(|e| {
            let mut e = *e;
            e.iter_mut().for_each(|idx| *idx = new_vert_indices[*idx]);
            e
        });
        let new_faces = self.faces().map(|f| {
            let mut f = *f;
            f.iter_mut().for_each(|idx| *idx = new_vert_indices[*idx]);
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
            .map(|&i| *self.elem(i))
            .collect::<Vec<_>>();
        let new_etags = elem_indices
            .iter()
            .map(|&i| self.etag(i))
            .collect::<Vec<_>>();
        self.clear_elems();
        self.add_elems(new_elems.iter().cloned(), new_etags.iter().cloned());
    }

    /// Reorder the mesh faces (in place)
    fn reorder_faces(&mut self, face_indices: &[usize]) {
        assert_eq!(face_indices.len(), self.n_faces());

        let new_faces = face_indices
            .iter()
            .map(|&i| *self.face(i))
            .collect::<Vec<_>>();
        let new_ftags = face_indices
            .iter()
            .map(|&i| self.ftag(i))
            .collect::<Vec<_>>();
        self.clear_faces();
        self.add_faces(new_faces.iter().cloned(), new_ftags.iter().cloned());
    }

    /// Reorder the mesh:
    ///   - RCM orderting based on the vertex-to-vertex connectivity is used for the mesh vertices
    ///   - elements and faces are sorted by their minimum vertex index
    fn reorder_rcm(&self) -> (Self, Vec<usize>, Vec<usize>, Vec<usize>) {
        let graph = self.compute_vertex_to_vertices();
        let vert_ids = graph.reverse_cuthill_mckee();
        let mut res = self.reorder_vertices(&vert_ids);

        let elem_ids = sort_elem_min_ids(res.elems().cloned());
        res.reorder_elems(&elem_ids);

        let face_ids = sort_elem_min_ids(res.faces().cloned());
        res.reorder_faces(&face_ids);

        (res, vert_ids, elem_ids, face_ids)
    }

    /// Randomly shuffle vertices, elements and faces
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
    fn write_meshb(&self, file_name: &str) -> Result<()> {
        let mut writer = MeshbWriter::new(file_name, 4, D as u8)?;

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
                self.etags().map(|x| x.into()),
            )?,
            3 => writer.write_triangles(
                self.elems().map(|x| {
                    let mut tmp = [0; 3];
                    tmp.copy_from_slice(x.as_slice());
                    tmp
                }),
                self.etags().map(|x| x.into()),
            )?,
            2 => writer.write_edges(
                self.elems().map(|x| {
                    let mut tmp = [0; 2];
                    tmp.copy_from_slice(x.as_slice());
                    tmp
                }),
                self.etags().map(|x| x.into()),
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
                self.ftags().map(|x| x.into()),
            )?,
            2 => writer.write_edges(
                self.faces().map(|x| {
                    let mut tmp = [0; 2];
                    tmp.copy_from_slice(x.as_slice());
                    tmp
                }),
                self.ftags().map(|x| x.into()),
            )?,
            _ => unimplemented!(),
        }

        writer.close();

        Ok(())
    }

    /// Export the mesh to a `.vtu` file
    fn write_vtk(&self, file_name: &str) -> Result<()> {
        let vtu = VTUFile::from_mesh(self, Encoding::Binary);

        vtu.export(file_name)?;

        Ok(())
    }

    /// Build a `Mesh<D, C, F>` mesh containing elements such that `filter(tag)` is true
    /// Only the required vertices are present
    fn extract_elems<G: Fn(Tag) -> bool, M: Mesh<D, C, F>>(&self, filter: G) -> M
    where
        Self: std::marker::Sized,
    {
        let mut new_ids = vec![usize::MAX; self.n_verts()];
        let mut next = 0;

        let n_elems = self
            .elems()
            .zip(self.etags())
            .filter(|(_, t)| filter(*t))
            .map(|(e, _)| {
                e.iter().for_each(|&i| {
                    if new_ids[i] == usize::MAX {
                        new_ids[i] = next;
                        next += 1;
                    }
                })
            })
            .count();
        let n_verts = next;
        let n_faces = self
            .faces()
            .filter(|f| f.iter().all(|&i| new_ids[i] != usize::MAX))
            .count();

        let mut verts = vec![Vertex::<D>::zeros(); n_verts];
        let mut faces = Vec::with_capacity(n_faces);
        let mut ftags = Vec::with_capacity(n_faces);
        let mut elems = Vec::with_capacity(n_elems);
        let mut etags = Vec::with_capacity(n_elems);

        new_ids
            .iter()
            .enumerate()
            .filter(|(_, &j)| j != usize::MAX)
            .for_each(|(i, &j)| verts[j] = *self.vert(i));
        self.faces()
            .zip(self.ftags())
            .filter(|(f, _)| f.iter().all(|&i| new_ids[i] != usize::MAX))
            .for_each(|(f, t)| {
                faces.push(std::array::from_fn(|i| new_ids[f[i]]));
                ftags.push(t);
            });
        self.elems()
            .zip(self.etags())
            .filter(|(e, _)| e.iter().all(|&i| new_ids[i] != usize::MAX))
            .for_each(|(e, t)| {
                elems.push(std::array::from_fn(|i| new_ids[e[i]]));
                etags.push(t);
            });

        let mut res = M::empty();
        res.add_verts(verts.iter().copied());
        res.add_faces(faces.iter().copied(), ftags.iter().copied());
        res.add_elems(elems.iter().copied(), etags.iter().copied());

        res
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
                f.iter().for_each(|&i| {
                    if new_ids[i] == usize::MAX {
                        new_ids[i] = next;
                        vert_ids.push(i);
                        next += 1;
                    }
                })
            })
            .count();
        let n_verts = next;

        let mut verts = vec![Vertex::<D>::zeros(); n_verts];
        let mut faces = Vec::with_capacity(n_faces);
        let mut ftags = Vec::with_capacity(n_faces);

        new_ids
            .iter()
            .enumerate()
            .filter(|(_, &j)| j != usize::MAX)
            .for_each(|(i, &j)| verts[j] = *self.vert(i));
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
            self.add_elems_and_tags(tmp.iter().cloned());
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
            self.add_faces_and_tags(tmp.iter().cloned());
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
            self.add_elems_and_tags(tmp.iter().cloned());
            ids
        } else {
            unreachable!()
        }
    }

    /// Split prisms and add them to the mesh (C == 4)
    fn add_prisms<I1: ExactSizeIterator<Item = Hexahedron>, I2: ExactSizeIterator<Item = Tag>>(
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
            self.add_elems_and_tags(tmp.iter().cloned());
        } else {
            unreachable!()
        }
    }

    /// Split pyramids and add them to the mesh (C == 4)
    fn add_pyramids<I1: ExactSizeIterator<Item = Hexahedron>, I2: ExactSizeIterator<Item = Tag>>(
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
            self.add_elems_and_tags(tmp.iter().cloned());
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
}
