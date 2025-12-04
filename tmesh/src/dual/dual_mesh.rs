//! Dual meshes
//! Both and edge-based representation (used for cell-vertex FV schemes)
//!  - edges of the original mesh, and the normals (scaled by the area) of the faces built around
//!    the edge
//!  - boundary faces
//!
//! and an explicit polygonal meshes (`PolyMesh<D>, where all the faces are of type `Face<F>`)
//! are built
use super::PolyMesh;
use crate::{
    Error, Result, Tag, Vertex,
    mesh::{Edge, GSimplex, Mesh, Simplex},
};
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};

/// Types of dual cells
#[derive(Clone, Copy, Debug)]
pub enum DualType {
    /// Classical median cells, going through edge, face and element geometric centers
    Median,
    /// Barth cells, as described in
    /// T. Barth, Aspects of unstructured grids and finite-volume solvers for the Euler
    /// and Navier-Stokes equations, Technical report 787, AGARD, 1992
    Barth,
    /// Modified Barth cells: if a barycentric coordinates of the center given by Barth
    /// cells is smaller that a threshold, it is clipped to 0
    ThresholdBarth(f64),
}

#[derive(Clone, Copy, Debug)]
pub enum DualCellCenter<const D: usize, C: Simplex> {
    Vertex(Vertex<D>),
    Face(C::FACE),
}

/// Dual of a `Mesh<D, C, F>`
pub trait DualMesh<const D: usize>: PolyMesh<D> {
    type C: Simplex;

    /// Compute the dual of `mesh`
    fn new(msh: &impl Mesh<D, C = Self::C>, t: DualType) -> Self;

    /// Display element `i`
    fn print_elem_info(&self, i: usize) {
        println!("Dual element {i}");
        self.par_edges_and_normals()
            .filter(|&(e, _)| e.get(0) == i || e.get(1) == i)
            .for_each(|(e, n)| {
                println!(
                    "  edges: {} - {} : n = {:?} (nrm = {})",
                    e.get(0),
                    e.get(1),
                    n,
                    n.norm()
                );
            });
        self.par_boundary_faces()
            .filter(|&(i0, _, _)| i0 == i)
            .for_each(|(_, tag, n)| {
                println!(
                    "  boundary face: tag = {},: n = {:?} (nrm = {})",
                    tag,
                    n,
                    n.norm()
                );
            });
        println!("  vol: {:.2e}", self.vol(i));
    }

    /// Get the vertex coordinates for face `f`
    fn gface(
        &self,
        f: &<Self::C as Simplex>::FACE,
    ) -> <<Self::C as Simplex>::FACE as Simplex>::GEOM<D> {
        <<Self::C as Simplex>::FACE as Simplex>::GEOM::from_iter(
            f.into_iter().map(|i| self.vert(i)),
        )
    }

    /// Parallel iterator over the faces vertex coordinates
    fn par_gfaces(
        &self,
    ) -> impl IndexedParallelIterator<Item = <<Self::C as Simplex>::FACE as Simplex>::GEOM<D>> + '_
    {
        self.par_faces().map(|f| {
            let f = <Self::C as Simplex>::FACE::from_iter(f);
            self.gface(&f)
        })
    }

    /// Sequential iterator over the faces vertex coordinates
    fn gfaces(
        &self,
    ) -> impl ExactSizeIterator<Item = <<Self::C as Simplex>::FACE as Simplex>::GEOM<D>> + '_ {
        self.faces().map(|f| {
            let f = <Self::C as Simplex>::FACE::from_iter(f);
            self.gface(&f)
        })
    }

    /// Number of edges
    fn n_edges(&self) -> usize;

    /// Get the `i`th edge
    fn edge(&self, i: usize) -> Edge<<Self::C as Simplex>::T>;

    /// Parallel itertator over the edges
    fn par_edges(
        &self,
    ) -> impl IndexedParallelIterator<Item = Edge<<Self::C as Simplex>::T>> + '_ + Clone {
        (0..self.n_edges()).into_par_iter().map(|i| self.edge(i))
    }

    /// Sequential iterator over the edges
    fn edges(&self) -> impl ExactSizeIterator<Item = Edge<<Self::C as Simplex>::T>> + '_ + Clone {
        (0..self.n_edges()).map(|i| self.edge(i))
    }

    /// Get the normal associated with the `i`th edge
    fn edge_normal(&self, i: usize) -> Vertex<D>;

    /// Parallel iterator over the edges and edge normals
    fn par_edges_and_normals(
        &self,
    ) -> impl IndexedParallelIterator<Item = (Edge<<Self::C as Simplex>::T>, Vertex<D>)> + '_ {
        (0..self.n_edges())
            .into_par_iter()
            .map(|i| (self.edge(i), self.edge_normal(i)))
    }

    /// Sequential iterator over the edges and edge normals
    fn edges_and_normals(
        &self,
    ) -> impl ExactSizeIterator<Item = (Edge<<Self::C as Simplex>::T>, Vertex<D>)> + '_ {
        (0..self.n_edges()).map(|i| (self.edge(i), self.edge_normal(i)))
    }

    /// Number of boundary faces
    fn n_boundary_faces(&self) -> usize;

    /// Parallel iterator over the boundary faces
    fn par_boundary_faces(
        &self,
    ) -> impl IndexedParallelIterator<Item = (usize, Tag, Vertex<D>)> + '_;

    /// Sequential iterator over the boundary faces
    fn boundary_faces(&self) -> impl ExactSizeIterator<Item = (usize, Tag, Vertex<D>)> + '_;

    /// Get the volume of the `i`th cell
    fn vol(&self, i: usize) -> f64 {
        self.elem(i)
            .map(|(i, orient)| {
                let mut f = <Self::C as Simplex>::FACE::from_iter(self.face(i));
                if !orient {
                    f.invert();
                }
                self.gface(&f)
            })
            .map(|gf| gf.center().dot(&gf.normal()))
            .sum::<f64>()
            / D as f64
    }

    /// Parallel iterator over cell volumes
    fn par_vols(&self) -> impl IndexedParallelIterator<Item = f64> + '_ {
        (0..self.n_elems()).into_par_iter().map(|i| self.vol(i))
    }

    /// Sequential iterator over cell volumes
    fn vols(&self) -> impl ExactSizeIterator<Item = f64> + '_ {
        (0..self.n_elems()).map(|i| self.vol(i))
    }

    /// Check if polygonal cell `e` is closed
    fn is_closed(&self, e: impl ExactSizeIterator<Item = (usize, bool)>) -> bool {
        let mut res = [0.0; D];

        e.map(|(i, orient)| {
            let mut f = <Self::C as Simplex>::FACE::from_iter(self.face(i));
            if !orient {
                f.invert();
            }
            self.gface(&f)
        })
        .for_each(|gf| {
            let n = gf.normal();
            res.iter_mut().zip(n.iter()).for_each(|(x, y)| *x += y);
        });
        res.iter().map(|x| x.abs()).sum::<f64>() < 1e-10
    }

    /// Check the validity of the dual mesh
    ///  - consistent number of faces and faces tags
    ///  - consistent face to vertex connectivity
    ///  - consistent element to face connectivity
    ///  - closed elements
    ///  - unique faces
    fn check(&self) -> Result<()> {
        // lengths
        if self.par_faces().len() != self.par_ftags().len() {
            return Err(Error::from("Inconsistent sizes (faces)"));
        }

        // indices
        if self.par_faces().any(|mut f| f.any(|i| i >= self.n_verts())) {
            return Err(Error::from("Inconsistent indices (faces)"));
        }

        // faces
        if self
            .par_elems()
            .any(|mut e| e.any(|x| x.0 >= self.n_faces()))
        {
            return Err(Error::from("Inconsistent indices (elems)"));
        }

        // closed elements
        for (i, (e, v)) in self.elems().zip(self.vols()).enumerate() {
            if v < 0.0 {
                let e = e.collect::<Vec<_>>();
                return Err(Error::from(&format!(
                    "Element {i} invalid: vol={v} < 0  ({e:?})"
                )));
            }
            if !self.is_closed(e.clone()) {
                let e = e.collect::<Vec<_>>();
                return Err(Error::from(&format!("Element {i} not closed ({e:?})")));
            }
        }

        // all faces appear only once
        let mut faces = FxHashMap::with_hasher(FxBuildHasher);
        for (i, f) in self.faces().enumerate() {
            assert_eq!(f.len(), <Self::C as Simplex>::FACE::N_VERTS);
            let res = <Self::C as Simplex>::FACE::from_iter(f.clone()).sorted();
            if let std::collections::hash_map::Entry::Vacant(e) = faces.entry(res) {
                e.insert(i);
            } else {
                let j = *faces.get(&res).unwrap();
                let f = f.collect::<Vec<_>>();
                let f_j = self.face(j).collect::<Vec<_>>();
                return Err(Error::from(&format!(
                    "Face {i} ({f:?}) = face {j} ({f_j:?})"
                )));
            }
        }

        // faces appear at most once in 1 or 2 elements
        let mut flg = vec![0; self.n_faces()];
        for (i_elem, e) in self.elems().enumerate() {
            let mut tmp = FxHashSet::with_capacity_and_hasher(e.len(), FxBuildHasher);
            for (i, sgn) in e {
                if tmp.contains(&i) {
                    return Err(Error::from(&format!(
                        "Face {i} appears multiple times in element {i_elem}"
                    )));
                }
                tmp.insert(i);
                if flg[i] == 0 {
                    if sgn {
                        flg[i] = 1;
                    } else {
                        flg[i] = -1;
                    }
                } else if flg[i] == 1 {
                    if sgn {
                        return Err(Error::from(&format!("Face {i} appears twice with True")));
                    }
                    flg[i] = 2;
                } else if flg[i] == -1 {
                    if !sgn {
                        return Err(Error::from(&format!("Face {i} appears twice with False")));
                    }
                    flg[i] = 2;
                } else if flg[i] == 2 {
                    return Err(Error::from(&format!("Face {i} appears 3 times")));
                } else {
                    unreachable!()
                }
            }
        }

        Ok(())
    }

    /// Return a `Mesh<D, C2, F2>` containing the faces such that `filter(tag)` is true.
    fn extract_faces<M: Mesh<D, C = <Self::C as Simplex>::FACE>, G: Fn(Tag) -> bool>(
        &self,
        filter: G,
    ) -> (M, Vec<usize>) {
        // find out if faces needs to be inverted (boundary faces will only be seen once)
        let mut flg = vec![true; self.n_faces()];
        self.elems().flatten().for_each(|(i, x)| flg[i] = x);

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
            .zip(flg.iter())
            .filter(|((f, _), _)| f.clone().all(|i| new_ids[i] != usize::MAX))
            .for_each(|((f, t), &invert)| {
                let mut f = <Self::C as Simplex>::FACE::from_iter(f.map(|i| new_ids[i]));
                if invert {
                    f.invert();
                }
                faces.push(f);
                ftags.push(t);
            });

        let mut res = M::empty();
        res.add_verts(verts.iter().copied());
        res.add_elems(faces.iter().copied(), ftags.iter().copied());

        (res, vert_ids)
    }

    /// Return a `Mesh<D, C2, F2>` containing all the boundary faces.
    fn boundary<M: Mesh<D, C = <Self::C as Simplex>::FACE>>(&self) -> (M, Vec<usize>) {
        self.extract_faces(|t| t > 0)
    }
}
