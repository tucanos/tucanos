//! Dual meshes
//! Both and edge-based representation (used for cell-vertex FV schemes)
//!  - edges of the original mesh, and the normals (scaled by the area) of the faces built around
//!    the edge
//!  - boundary faces
//!
//! and an explicit polygonal meshes (`PolyMesh<D>`, where all the faces are of type `Face<F>`)
//! are built
use super::PolyMesh;
use crate::{
    Tag, Vertex,
    dual::{PolyMeshType, SimplePolyMesh, poly_mesh::PolyFaceType},
    mesh::{Edge, GSimplex, Mesh, Simplex},
};
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use rustc_hash::{FxBuildHasher, FxHashMap};

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

/// Dual of a `Mesh<D>`
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
        println!("  vol: {:.2e}", self.vol_c::<<Self::C as Simplex>::FACE>(i));
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

    /// Return a `Mesh<D>` (with element type `C::FACE`) containing the faces
    /// such that `filter(tag)` is true.
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

    /// Return a `Mesh<D>` (with element type `C::FACE`) containing all the
    /// boundary faces.
    fn boundary<M: Mesh<D, C = <Self::C as Simplex>::FACE>>(&self) -> (M, Vec<usize>) {
        self.extract_faces(|t| t > 0)
    }

    /// Return the id of a split face and its orientation.
    ///
    /// Algorithm:
    /// 1. Build a canonical key by sorting the node ids (topology-only identity).
    /// 2. Reuse an existing face if the key is already in `new_faces`.
    /// 3. Compute orientation as `key.is_same(face)`.
    /// 4. If needed, create the sorted face `key` in `res`, cache its id, and return `(id, orient)`.
    ///
    /// This deduplicates faces shared by neighboring cones during element splitting.
    #[must_use]
    fn get_or_insert_split_face(
        res: &mut SimplePolyMesh<D>,
        new_faces: &mut FxHashMap<<Self::C as Simplex>::FACE, (usize, bool)>,
        face: &<Self::C as Simplex>::FACE,
    ) -> (usize, bool) {
        let key = face.sorted();
        let orient_wrt_key = key.is_same(face);
        if let Some(&(idx, stored_orient_wrt_key)) = new_faces.get(&key) {
            (idx, orient_wrt_key == stored_orient_wrt_key)
        } else {
            let idx = if D == 2 {
                let nodes = [key.get(0), key.get(1)];
                res.insert_face(&nodes, 0)
            } else {
                let nodes = [key.get(0), key.get(1), key.get(2)];
                res.insert_face(&nodes, 0)
            };
            new_faces.insert(key, (idx, true));
            (idx, orient_wrt_key)
        }
    }

    /// Split one parent edge-face of a 2D dual element into a cone triangle.
    ///
    /// Algorithm:
    /// 1. Orient the base edge according to `orient_parent`.
    /// 2. Skip if the cone is topologically invalid or geometrically flat.
    /// 3. Build one triangle from:
    ///    - original base face `(i_face, orient_parent)`
    ///    - two side edges `(v -> apex)` and `(apex -> u)`.
    /// 4. Deduplicate side edges across the current split element and compute
    ///    orientation flags from topology only.
    fn add_simplex_from_face_and_vert_2d(
        &self,
        res: &mut SimplePolyMesh<D>,
        new_faces: &mut FxHashMap<<Self::C as Simplex>::FACE, (usize, bool)>,
        mut face: <Self::C as Simplex>::FACE,
        parent_face: (usize, bool),
        i_vert: usize,
        etag: Tag,
    ) {
        if face.contains(i_vert) {
            return;
        }

        let (i_face, orient_parent) = parent_face;

        // Base edge oriented as seen by the parent element.
        if !orient_parent {
            face.invert();
        }

        let mut elem_faces = [(0, false); 3];
        elem_faces[0] = (i_face, orient_parent);

        // Triangle edges in local order: (u->v), (v->i_vert), (i_vert->u)
        for (k, node_face) in face.faces().enumerate() {
            let mut nodes = <Self::C as Simplex>::FACE::from_vert_and_face(i_vert, &node_face);
            if k == 0 {
                // Keep the same local orientation as the previous explicit construction.
                nodes.invert();
            }
            let (i_new_face, orient) = Self::get_or_insert_split_face(res, new_faces, &nodes);
            elem_faces[k + 1] = (i_new_face, orient);
        }

        res.insert_elem(elem_faces, etag);
    }

    /// Split one parent triangle-face of a 3D dual element into a cone tetrahedron.
    ///
    /// Algorithm:
    /// 1. Orient the base triangle according to `orient_parent`.
    /// 2. Skip if the cone is topologically invalid or geometrically flat.
    /// 3. Build one tetrahedron from:
    ///    - original base face `(i_face, orient_parent)`
    ///    - three side triangles `(apex, v, u)`, `(apex, w, v)`, `(apex, u, w)`.
    /// 4. Deduplicate side triangles across the current split element and compute
    ///    orientation flags from topology only.
    fn add_simplex_from_face_and_vert_3d(
        &self,
        res: &mut SimplePolyMesh<D>,
        new_faces: &mut FxHashMap<<Self::C as Simplex>::FACE, (usize, bool)>,
        mut face: <Self::C as Simplex>::FACE,
        parent_face: (usize, bool),
        i_vert: usize,
        etag: Tag,
    ) {
        if face.contains(i_vert) {
            return;
        }

        let (i_face, orient_parent) = parent_face;

        // Base triangle oriented as seen by the parent element.
        if !orient_parent {
            face.invert();
        }

        let mut elem_faces = [(0, false); 4];
        elem_faces[0] = (i_face, orient_parent);

        for (k, mut edge_face) in face.faces().enumerate() {
            edge_face.invert();
            let nodes = <Self::C as Simplex>::FACE::from_vert_and_face(i_vert, &edge_face);
            let (i_new_face, orient) = Self::get_or_insert_split_face(res, new_faces, &nodes);
            elem_faces[k + 1] = (i_new_face, orient);
        }

        res.insert_elem(elem_faces, etag);
    }

    /// Split elements whose centers is outside the element into simplices by adding a vertex at the vertices of the primal mesh
    /// and connecting it to the faces
    fn split_elements(&self, primal: &impl Mesh<D, C = Self::C>) -> SimplePolyMesh<D> {
        let poly_type = match D {
            2 => PolyMeshType::Polygons,
            3 => PolyMeshType::Polyhedra,
            _ => unreachable!(),
        };

        let mut res = SimplePolyMesh::empty(poly_type, PolyFaceType::Simplices);

        // add the vertices and faces from self
        for v in self.verts() {
            res.insert_vert(v);
        }
        for (f, t) in self.faces().zip(self.ftags()) {
            let f = f.collect::<Vec<_>>();
            res.insert_face(&f, t);
        }

        let mut new_faces =
            FxHashMap::<<Self::C as Simplex>::FACE, (usize, bool)>::with_hasher(FxBuildHasher);

        for i in 0..self.n_elems() {
            let center = self.elem_center_c::<<Self::C as Simplex>::FACE>(i);
            if self.is_vertex_in_elem_c::<<Self::C as Simplex>::FACE>(&center, i) {
                // keep element unchanged: face ids are the same as in self
                res.insert_elem(self.elem(i), self.etag(i));
                continue;
            }

            let elem_faces = self.elem(i).collect::<Vec<_>>();
            let has_boundary_face = elem_faces.iter().any(|(i_face, _)| self.ftag(*i_face) != 0);
            let primal_vert = primal.vert(i);

            // split the element by coning each boundary simplex face to a new vertex
            let i_vert = if has_boundary_face {
                let mut i_vert = None;
                for (i_face, _) in &elem_faces {
                    for j in self.face(*i_face) {
                        if (res.vert(j) - primal_vert).norm() <= 1e-14 {
                            i_vert = Some(j);
                            break;
                        }
                    }
                    if i_vert.is_some() {
                        break;
                    }
                }
                i_vert.unwrap_or_else(|| res.insert_vert(primal_vert))
            } else {
                res.insert_vert(primal_vert)
            };
            let etag = self.etag(i);
            new_faces.clear();
            for (i_face, _) in &elem_faces {
                if self.ftag(*i_face) != 0 {
                    let face = <Self::C as Simplex>::FACE::from_iter(res.face(*i_face));
                    let key = face.sorted();
                    if let std::collections::hash_map::Entry::Vacant(entry) = new_faces.entry(key) {
                        entry.insert((*i_face, key.is_same(&face)));
                    }
                }
            }

            for (i_face, orient_parent) in elem_faces {
                let face = <Self::C as Simplex>::FACE::from_iter(self.face(i_face));
                if D == 2 {
                    self.add_simplex_from_face_and_vert_2d(
                        &mut res,
                        &mut new_faces,
                        face,
                        (i_face, orient_parent),
                        i_vert,
                        etag,
                    );
                } else if D == 3 {
                    self.add_simplex_from_face_and_vert_3d(
                        &mut res,
                        &mut new_faces,
                        face,
                        (i_face, orient_parent),
                        i_vert,
                        etag,
                    );
                } else {
                    unreachable!();
                }
            }
        }

        res
    }
}
