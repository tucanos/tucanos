mod curvature;
mod orient;
use crate::{
    Dim, Result, Tag, TopoTag,
    geometry::curvature::{compute_curvature, write_curvature},
    mesh::{MeshTopology, Topology},
};
use log::debug;
pub use orient::orient_geometry;
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};
use tmesh::{
    Vertex,
    mesh::{
        Edge, GSimplex, GenericMesh, Idx, Mesh, QuadraticEdge, QuadraticTriangle, Simplex, SubMesh,
        Triangle,
        to_quadratic::{to_quadratic_edge_mesh, to_quadratic_triangle_mesh},
    },
    spatialindex::ObjectIndex,
};

/// Representation of a D-dimensional geometry
pub trait Geometry<const D: usize>: Send + Sync {
    /// Check that the geometry is consistent with a Topology
    fn check(&self, topo: &Topology) -> Result<()>;

    /// Project a vertex, associated with a given `TopoTag`, onto the geometry
    /// The distance between the original vertex and its projection if returned
    fn project(&self, pt: &mut Vertex<D>, tag: &TopoTag) -> f64;

    /// Compute the angle between a vector n and the normal at the projection of pt onto the geometry
    fn angle(&self, pt: &Vertex<D>, n: &Vertex<D>, tag: &TopoTag) -> f64;

    /// Compute the max distance between the face centers and the geometry normals
    fn project_vertices<M: Mesh<D>>(&self, mesh: &mut M, topo: &MeshTopology) -> f64 {
        let mut d_max = 0.0;
        for (p, tag) in mesh.verts_mut().zip(topo.vtags()) {
            if tag.0 < D as Dim {
                let d = self.project(p, tag);
                d_max = f64::max(d_max, d);
            }
        }

        d_max
    }

    /// Compute the max distance between the face centers and the geometry normals
    fn max_distance<M: Mesh<D>>(&self, mesh: &M) -> f64 {
        let mut d_max = 0.0;
        if M::C::DIM == D {
            for (gf, tag) in mesh.gfaces().zip(mesh.ftags()) {
                let mut c = gf.center();
                let d = self.project(&mut c, &(<M::C as Simplex>::FACE::DIM as Dim, tag));
                d_max = f64::max(d_max, d);
            }
            d_max
        } else if M::C::DIM == D - 1 {
            for (gf, tag) in mesh.gelems().zip(mesh.etags()) {
                let mut c = gf.center();
                let d = self.project(&mut c, &(<M::C as Simplex>::DIM as Dim, tag));
                d_max = f64::max(d_max, d);
            }
            d_max
        } else {
            unreachable!();
        }
    }

    /// Compute the max angle between the face normals and the geometry normals
    fn max_normal_angle<M: Mesh<D>>(&self, mesh: &M) -> f64 {
        let mut a_max = 0.0;
        if M::C::DIM == D {
            for (gf, tag) in mesh.gfaces().zip(mesh.ftags()) {
                if tag > 0 {
                    let c = gf.center();
                    let n = gf.normal(None).normalize();
                    let a = self.angle(&c, &n, &(<M::C as Simplex>::FACE::DIM as Dim, tag));
                    a_max = f64::max(a_max, a);
                }
            }
        } else if M::C::DIM == D - 1 {
            for (gf, tag) in mesh.gelems().zip(mesh.etags()) {
                if tag > 0 {
                    let c = gf.center();
                    let n = gf.normal(None).normalize();
                    let a = self.angle(&c, &n, &(M::C::DIM as Dim, tag));
                    a_max = f64::max(a_max, a);
                }
            }
        } else {
            unreachable!();
        }

        a_max
    }

    /// Convert a linear mesh to a quadratic mesh (triangles)
    fn to_quadratic_triangle_mesh<M: Mesh<D, C = QuadraticTriangle<impl Idx>>>(
        &self,
        mesh: &impl Mesh<D, C = Triangle<impl Idx>>,
    ) -> M {
        let mut res = to_quadratic_triangle_mesh(mesh);
        let topo = MeshTopology::new(&res);

        self.project_vertices(&mut res, &topo);
        res
    }

    /// Convert a linear mesh to a quadratic mesh (edges)
    fn to_quadratic_edge_mesh<M: Mesh<D, C = QuadraticEdge<impl Idx>>>(
        &self,
        mesh: &impl Mesh<D, C = Edge<impl Idx>>,
    ) -> M {
        let mut res = to_quadratic_edge_mesh(mesh);
        let topo = MeshTopology::new(&res);
        self.project_vertices(&mut res, &topo);
        res
    }
}

/// No geometric model
pub struct NoGeometry<const D: usize>();

impl<const D: usize> Geometry<D> for NoGeometry<D> {
    fn check(&self, _topo: &Topology) -> Result<()> {
        Ok(())
    }

    fn project(&self, _pt: &mut Vertex<D>, tag: &TopoTag) -> f64 {
        assert!(tag.0 < D as Dim);
        0.
    }

    fn angle(&self, _pt: &Vertex<D>, _n: &Vertex<D>, tag: &TopoTag) -> f64 {
        assert_eq!(tag.0, D as Dim - 1);
        0.
    }
}

/// Geometry for a patch of faces with a constant tag
struct MeshedPatchGeometry<const D: usize, M: Mesh<D>> {
    /// The ObjectIndex
    tree: ObjectIndex<D, M>,
}

impl<const D: usize, M: Mesh<D>> MeshedPatchGeometry<D, M> {
    /// Create a `LinearPatchGeometry` from a `SimplexMesh`
    pub fn new(mesh: M) -> Self {
        Self {
            tree: ObjectIndex::new(mesh),
        }
    }

    // Perform projection
    fn project(&self, pt: &Vertex<D>) -> (f64, Vertex<D>) {
        self.tree.project(pt)
    }
}

/// Geometry for a patch of faces with a constant tag, with curvature information
struct MeshedPatchGeometryWithCurvature<const D: usize, M: Mesh<D>> {
    /// The ObjectIndex
    tree: ObjectIndex<D, M>,
    /// Optionally, the first principal curvature direction
    u: Vec<Vertex<D>>,
    /// Optionally, the second principal curvature direction (3D only)
    v: Option<Vec<Vertex<D>>>,
}

impl<const D: usize, M: Mesh<D>> MeshedPatchGeometryWithCurvature<D, M> {
    /// Create a `LinearPatchGeometry` from a `SimplexMesh`
    pub fn new(mut mesh: M) -> Self {
        mesh.fix().unwrap();
        let (u, v) = compute_curvature(&mesh);

        let tree = ObjectIndex::new(mesh);

        Self { tree, u, v }
    }

    // Perform projection
    fn project(&self, pt: &Vertex<D>) -> (f64, Vertex<D>) {
        self.tree.project(pt)
    }

    fn curvature(&self, pt: &Vertex<D>) -> (Vertex<D>, Option<Vertex<D>>) {
        let i_elem = self.tree.nearest_elem(pt);

        self.v.as_ref().map_or_else(
            || (self.u[i_elem], None),
            |v| (self.u[i_elem], Some(v[i_elem])),
        )
    }
}

/// Meshed (stl-like) representation of a geometry
/// doc TODO
pub struct MeshedGeometry<const D: usize, M: Mesh<D>> {
    /// The surface patches
    patches: FxHashMap<Tag, MeshedPatchGeometryWithCurvature<D, M>>,
    /// The edges
    edges: FxHashMap<Tag, MeshedPatchGeometry<D, GenericMesh<D, <M::C as Simplex>::FACE>>>,
    /// Topology map (for edges in 3d)
    edge2faces: FxHashMap<Tag, Vec<Tag>>,
    /// Topology map (for edges in 3d)
    edge_map: FxHashMap<Tag, Tag>,
}

impl<const D: usize, M: Mesh<D>> MeshedGeometry<D, M> {
    /// Create a `MeshedGeometry`
    /// For triangle meshes,
    ///     - the edges of `bdy` mut be properly tagged, e.g. with `bdy.fix()`
    ///     - to set the edge tag map from a given mesh to the current geometry, use `set_topo_map`
    pub fn new(bdy: &M) -> Result<Self> {
        assert!(<M::C as Simplex>::GEOM::<D>::has_normal());

        let face_tags: FxHashSet<Tag> = bdy.etags().collect();

        let mut patches = FxHashMap::default();
        for tag in face_tags.iter().copied() {
            debug!("Create LinearPatchGeometryWithCurvature for patch {tag}");
            let submesh = SubMesh::new(bdy, |t| t == tag).mesh;
            assert_ne!(submesh.n_verts(), 0, "Geometry mesh empty for tag {tag}");

            patches.insert(tag, MeshedPatchGeometryWithCurvature::new(submesh));
        }

        // Edges
        let mut edges = FxHashMap::default();
        let mut edge2faces = FxHashMap::default();

        if M::C::DIM == 2 {
            let topo = MeshTopology::new(bdy);
            let topo = topo.topo().clone();
            let (bdy_edges, _) = bdy.boundary::<GenericMesh<D, <M::C as Simplex>::FACE>>();

            let edge_tags: FxHashSet<Tag> = bdy_edges.etags().collect();

            for tag in edge_tags {
                debug!("Create LinearPatchGeometry for edge {tag}");
                let submesh = SubMesh::new(&bdy_edges, |t| t == tag).mesh;
                let bdy_topo_node = topo
                    .get((<M::C as Simplex>::FACE::DIM as Dim, tag))
                    .unwrap();
                let parents = &bdy_topo_node.parents;
                edges.insert(tag, MeshedPatchGeometry::new(submesh));
                edge2faces.insert(tag, parents.iter().copied().collect());
            }
        }

        let geom = Self {
            patches,
            edges,
            edge2faces,
            edge_map: FxHashMap::with_hasher(FxBuildHasher),
        };

        Ok(geom)
    }

    pub fn set_topo_map(&mut self, mesh_topo: &Topology) {
        self.edge_map.clear();
        for &tag in self.edges.keys() {
            let parents = self.edge2faces.get(&tag).unwrap();
            let mesh_topo_node = mesh_topo
                .get_from_parents_iter(<M::C as Simplex>::FACE::DIM as Dim, parents.iter().copied())
                .unwrap();
            self.edge_map.insert(mesh_topo_node.tag.1, tag);
        }
    }

    #[must_use]
    pub fn curvature(&self, pt: &Vertex<D>, tag: Tag) -> (Vertex<D>, Option<Vertex<D>>) {
        self.patches.get(&tag).unwrap().curvature(pt)
    }

    pub fn write_curvature(&self, fname: &str) -> Result<()> {
        for (tag, patch) in &self.patches {
            write_curvature(
                patch.tree.mesh(),
                &String::from(fname).replace(".vtu", &format!("_{tag}.vtu")),
            )?;
        }

        Ok(())
    }
}

impl<const D: usize, M: Mesh<D>> Geometry<D> for MeshedGeometry<D, M> {
    fn check(&self, _topo: &Topology) -> Result<()> {
        // The check is performed during creation
        Ok(())
    }

    fn project(&self, pt: &mut Vertex<D>, tag: &TopoTag) -> f64 {
        assert!(tag.0 < D as Dim);

        let (dist, p) = if tag.1 < 0 {
            (0.0, *pt)
        } else if tag.0 == M::C::DIM as Dim {
            let patch = self.patches.get(&tag.1).unwrap_or_else(|| {
                panic!(
                    "Invalid face tag {tag:?}. Available tags are {:?}",
                    self.patches.keys().collect::<Vec<_>>()
                )
            });
            patch.project(pt)
        } else if tag.0 == 0 {
            (0.0, *pt)
        } else if tag.0 == M::C::DIM as Dim - 1 {
            // after 0 to make sure that if is used only for E=Triangle
            let tag = if self.edge_map.is_empty() {
                tag.1
            } else {
                *self.edge_map.get(&tag.1).unwrap()
            };
            let edge = self.edges.get(&tag).unwrap_or_else(|| {
                panic!(
                    "Invalid edge tag {tag:?}. Available tags are {:?}",
                    self.edges.keys().collect::<Vec<_>>()
                )
            });
            edge.project(pt)
        } else {
            unreachable!("{:?}", tag)
        };

        *pt = p;
        dist
    }

    fn angle(&self, pt: &Vertex<D>, n: &Vertex<D>, tag: &TopoTag) -> f64 {
        assert_eq!(tag.0, D as Dim - 1);

        let patch = self.patches.get(&tag.1).unwrap();
        let idx = patch.tree.nearest_elem(pt);
        let n_ref = patch
            .tree
            .mesh()
            .gelem(&patch.tree.mesh().elem(idx))
            .normal(None)
            .normalize();
        let cos_a = n.dot(&n_ref).clamp(-1.0, 1.0);
        f64::acos(cos_a).to_degrees()
    }
}

#[cfg(test)]
mod tests {
    use tmesh::{
        Vertex,
        mesh::{BoundaryMesh2d, BoundaryMesh3d, Mesh, read_stl},
    };

    use super::{Geometry, MeshedGeometry};
    use crate::{
        Result,
        mesh::{
            MeshTopology,
            test_meshes::{test_mesh_2d, test_mesh_3d, write_stl_file},
        },
    };
    use std::fs::remove_file;

    #[test]
    fn test_stl() -> Result<()> {
        write_stl_file("cube2.stl")?;
        let geom: BoundaryMesh3d = read_stl("cube2.stl")?;
        remove_file("cube2.stl")?;

        let geom = MeshedGeometry::new(&geom)?;
        let mut p = Vertex::<3>::new(2., 0.5, 0.5);
        let d = geom.project(&mut p, &(2, 1));
        assert!(f64::abs(d - 1.) < 1e-12);
        assert!((p - Vertex::<3>::new(1., 0.5, 0.5)).norm() < 1e-12);

        let mut p = Vertex::<3>::new(0.5, 0.75, 0.5);
        let d = geom.project(&mut p, &(2, 1));
        assert!(f64::abs(d - 0.25) < 1e-12);
        assert!((p - Vertex::<3>::new(0.5, 1., 0.5)).norm() < 1e-12);

        Ok(())
    }

    #[test]
    fn test_linear_geometry_2d() -> Result<()> {
        let mut mesh = test_mesh_2d().split().split();
        mesh.fix().unwrap();

        let (bdy, _) = mesh.boundary::<BoundaryMesh2d>();
        let geom = MeshedGeometry::new(&bdy)?;

        let mut pt = Vertex::<2>::new(0.75, 0.5);
        let d = geom.project(&mut pt, &(1, 1));
        assert!(f64::abs(d - 0.5) < 1e-12);

        let mut pt = Vertex::<2>::new(0.75, 0.5);
        let d = geom.project(&mut pt, &(1, 2));
        assert!(f64::abs(d - 0.25) < 1e-12);

        let mut pt = Vertex::<2>::new(0.75, 0.5);
        let d = geom.project(&mut pt, &(1, 3));
        assert!(f64::abs(d - 0.5) < 1e-12);

        let mut pt = Vertex::<2>::new(0.75, 0.5);
        let d = geom.project(&mut pt, &(1, 4));
        assert!(f64::abs(d - 0.75) < 1e-12);

        Ok(())
    }

    #[test]
    fn test_linear_geometry_3d() -> Result<()> {
        let mut mesh = test_mesh_3d().split().split();
        mesh.fix().unwrap();
        let topo = MeshTopology::new(&mesh);

        let (mut bdy, _) = mesh.boundary::<BoundaryMesh3d>();
        bdy.fix().unwrap();
        let mut geom = MeshedGeometry::new(&bdy)?;
        geom.set_topo_map(topo.topo());

        let mut pt = Vertex::<3>::new(0.75, 0.5, 0.25);
        let _ = geom.project(&mut pt, &(2, 1));
        assert!(f64::abs(pt[0] - 0.75) < 1e-12);
        assert!(f64::abs(pt[1] - 0.5) < 1e-12);
        assert!(f64::abs(pt[2] - 0.0) < 1e-12);

        let mut pt = Vertex::<3>::new(0.75, 0.5, 0.25);
        let _ = geom.project(&mut pt, &(2, 2));
        assert!(f64::abs(pt[0] - 0.75) < 1e-12);
        assert!(f64::abs(pt[1] - 0.5) < 1e-12);
        assert!(f64::abs(pt[2] - 1.0) < 1e-12);

        let mut pt = Vertex::<3>::new(0.75, 0.5, 0.25);
        let _ = geom.project(&mut pt, &(2, 3));
        assert!(f64::abs(pt[0] - 0.75) < 1e-12);
        assert!(f64::abs(pt[1] - 0.0) < 1e-12);
        assert!(f64::abs(pt[2] - 0.25) < 1e-12);

        let mut pt = Vertex::<3>::new(0.75, 0.5, 0.25);
        let _ = geom.project(&mut pt, &(2, 4));
        assert!(f64::abs(pt[0] - 0.75) < 1e-12);
        assert!(f64::abs(pt[1] - 1.0) < 1e-12);
        assert!(f64::abs(pt[2] - 0.25) < 1e-12);

        let mut pt = Vertex::<3>::new(0.75, 0.5, 0.25);
        let _ = geom.project(&mut pt, &(2, 5));
        assert!(f64::abs(pt[0] - 1.0) < 1e-12);
        assert!(f64::abs(pt[1] - 0.5) < 1e-12);
        assert!(f64::abs(pt[2] - 0.25) < 1e-12);

        let mut pt = Vertex::<3>::new(0.75, 0.5, 0.25);
        let _ = geom.project(&mut pt, &(2, 6));
        assert!(f64::abs(pt[0] - 0.0) < 1e-12);
        assert!(f64::abs(pt[1] - 0.5) < 1e-12);
        assert!(f64::abs(pt[2] - 0.25) < 1e-12);

        let topo = MeshTopology::new(&mesh);

        let topo_node = topo.topo().get_from_parents(1, &[6, 3]).unwrap();
        let mut pt = Vertex::<3>::new(0.75, 0.5, 0.25);
        let _ = geom.project(&mut pt, &(1, topo_node.tag.1));
        assert!(f64::abs(pt[0] - 0.0) < 1e-12);
        assert!(f64::abs(pt[1] - 0.0) < 1e-12);
        assert!(f64::abs(pt[2] - 0.25) < 1e-12);

        Ok(())
    }
}
