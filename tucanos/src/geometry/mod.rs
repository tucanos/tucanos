mod curvature;
mod orient;

use crate::{
    Dim, Error, Result, Tag, TopoTag,
    geometry::curvature::HasCurvature,
    mesh::{MeshTopology, Topology},
};
use log::{debug, warn};
pub use orient::orient_geometry;
use rustc_hash::{FxHashMap, FxHashSet};
use tmesh::{
    Vertex,
    io::VTUFile,
    mesh::{GSimplex, GenericMesh, Mesh, Simplex, SubMesh},
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
    fn project_vertices<C: Simplex, M: Mesh<D, C>>(
        &self,
        mesh: &mut M,
        topo: &MeshTopology,
    ) -> f64 {
        let mut d_max = 0.0;
        for (p, tag) in mesh.verts_mut().zip(topo.vtags()) {
            if tag.0 < C::DIM as Dim {
                let d = self.project(p, tag);
                d_max = f64::max(d_max, d);
            }
        }

        d_max
    }

    /// Compute the max distance between the face centers and the geometry normals
    fn max_distance<C: Simplex, M: Mesh<D, C>>(&self, mesh: &M) -> f64 {
        let mut d_max = 0.0;
        for (gf, tag) in mesh.gfaces().zip(mesh.ftags()) {
            let mut c = gf.center();
            let d = self.project(&mut c, &(C::FACE::DIM as Dim, tag));
            d_max = f64::max(d_max, d);
        }
        d_max
    }

    /// Compute the max angle between the face normals and the geometry normals
    fn max_normal_angle<C: Simplex, M: Mesh<D, C>>(&self, mesh: &M) -> f64 {
        let mut a_max = 0.0;
        if C::DIM == D {
            for (gf, tag) in mesh.gfaces().zip(mesh.ftags()) {
                if tag > 0 {
                    let c = gf.center();
                    let n = gf.normal().normalize();
                    let a = self.angle(&c, &n, &(C::FACE::DIM as Dim, tag));
                    a_max = f64::max(a_max, a);
                }
            }
        } else if C::DIM == D - 1 {
            for (gf, tag) in mesh.gelems().zip(mesh.etags()) {
                if tag > 0 {
                    let c = gf.center();
                    let n = gf.normal().normalize();
                    let a = self.angle(&c, &n, &(C::DIM as Dim, tag));
                    a_max = f64::max(a_max, a);
                }
            }
        } else {
            unreachable!();
        }

        a_max
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
struct LinearPatchGeometry<const D: usize> {
    /// The ObjectIndex
    tree: ObjectIndex<D>,
}

impl<const D: usize> LinearPatchGeometry<D> {
    /// Create a `LinearPatchGeometry` from a `SimplexMesh`
    pub fn new<C: Simplex, M: Mesh<D, C>>(mesh: &M) -> Self {
        let tree = ObjectIndex::new(mesh);

        Self { tree }
    }

    // Perform projection
    fn project(&self, pt: &Vertex<D>) -> (f64, Vertex<D>) {
        self.tree.project(pt)
    }
}

/// Geometry for a patch of faces with a constant tag, with curvature information
struct LinearPatchGeometryWithCurvature<const D: usize, C: Simplex>
where
    GenericMesh<D, C>: HasCurvature<D>,
{
    /// The face mesh
    mesh: GenericMesh<D, C>,
    /// The ObjectIndex
    tree: ObjectIndex<D>,
    /// Optionally, the first principal curvature direction
    u: Option<Vec<Vertex<D>>>,
    /// Optionally, the second principal curvature direction (3D only)
    v: Option<Vec<Vertex<D>>>,
}

impl<const D: usize, C: Simplex> LinearPatchGeometryWithCurvature<D, C>
where
    GenericMesh<D, C>: HasCurvature<D>,
{
    /// Create a `LinearPatchGeometry` from a `SimplexMesh`
    pub fn new(mut mesh: GenericMesh<D, C>) -> Self {
        mesh.fix().unwrap();

        let tree = ObjectIndex::new(&mesh);

        Self {
            mesh,
            tree,
            u: None,
            v: None,
        }
    }

    // Perform projection
    fn project(&self, pt: &Vertex<D>) -> (f64, Vertex<D>) {
        self.tree.project(pt)
    }

    fn compute_curvature(&mut self) {
        let (u, v) = self.mesh.compute_curvature();

        self.u = Some(u);
        self.v = v;
    }

    fn curvature(&self, pt: &Vertex<D>) -> Result<(Vertex<D>, Option<Vertex<D>>)> {
        if self.u.is_none() {
            return Err(Error::from("LinearGeometry: compute_curvature not called"));
        }

        let i_elem = self.tree.nearest_elem(pt);
        let u = self.u.as_ref().unwrap();
        self.v.as_ref().map_or_else(
            || Ok((u[i_elem], None)),
            |v| Ok((u[i_elem], Some(v[i_elem]))),
        )
    }

    pub fn write_curvature(&self, fname: &str) -> Result<()> {
        let mut writer = VTUFile::from_mesh(&self.mesh, tmesh::io::VTUEncoding::Binary);
        writer.add_point_data("u", D, self.u.as_ref().unwrap().iter().flatten().copied());
        writer.add_point_data("v", D, self.v.as_ref().unwrap().iter().flatten().copied());
        writer.export(fname)?;

        Ok(())
    }
}

/// Piecewise linear (stl-like) representation of a geometry
/// doc TODO
pub struct LinearGeometry<const D: usize, C: Simplex>
where
    GenericMesh<D, C>: HasCurvature<D>,
{
    /// The surface patches
    patches: FxHashMap<Tag, LinearPatchGeometryWithCurvature<D, C>>,
    /// The edges
    edges: FxHashMap<Tag, LinearPatchGeometry<D>>,
}

impl<const D: usize, C: Simplex> LinearGeometry<D, C>
where
    GenericMesh<D, C>: HasCurvature<D>,
{
    /// Create a `LinearGeometry` for the boundary of `mesh` (with positive tags) from a
    /// `SimplexMesh` representation of the boundary
    pub fn new<C2: Simplex>(
        mesh: &impl Mesh<D, C2>,
        topo: &MeshTopology,
        mut bdy: GenericMesh<D, C>,
    ) -> Result<Self> {
        assert!(C2::DIM >= C::DIM);

        let mesh_topo = topo.topo();

        // Faces
        let mesh_face_tags = mesh_topo.tags(C::DIM as Dim);
        let face_tags: FxHashSet<Tag> = if C2::DIM == C::DIM + 1 {
            mesh.ftags().filter(|&t| t > 0).collect()
        } else if C2::DIM == C::DIM {
            mesh.etags().filter(|&t| t > 0).collect()
        } else {
            unimplemented!("mesh dimension {}, bdy dimension {}", C::DIM, C2::DIM);
        };

        let mut patches = FxHashMap::default();
        for tag in face_tags.iter().copied() {
            debug!("Create LinearPatchGeometryWithCurvature for patch {tag}");
            if mesh_topo.get((C::DIM as Dim, tag)).is_none() {
                return Err(Error::from(&format!(
                    "LinearGeometry: face tag {tag:?} not found in topo (mesh: {mesh_face_tags:?}, bdy: {face_tags:?})"
                )));
            }
            let submesh = SubMesh::new(&bdy, |t| t == tag).mesh;
            assert_ne!(submesh.n_verts(), 0, "Geometry mesh empty for tag {tag}");
            patches.insert(tag, LinearPatchGeometryWithCurvature::new(submesh));
        }

        // Edges
        let mut edges = FxHashMap::default();
        if C::DIM == 2 {
            bdy.fix().unwrap();
            let topo = MeshTopology::new(&bdy);
            let bdy_topo = topo.topo();
            let (bdy_edges, _) = bdy.boundary::<GenericMesh<D, C::FACE>>();

            let edge_tags: FxHashSet<Tag> = bdy_edges.etags().collect();

            for tag in edge_tags {
                debug!("Create LinearPatchGeometry for edge {tag}");
                // find the edge tag in mesh_topo
                let bdy_topo_node = bdy_topo.get((C::FACE::DIM as Dim, tag)).unwrap();
                let bdy_parents = &bdy_topo_node.parents;
                let mesh_topo_node = mesh_topo
                    .get_from_parents_iter(C::FACE::DIM as Dim, bdy_parents.iter().copied());

                if let Some(mesh_topo_node) = mesh_topo_node {
                    let submesh = SubMesh::new(&bdy_edges, |t| t == tag).mesh;
                    if mesh_topo_node.tag.1 > 0 {
                        edges.insert(mesh_topo_node.tag.1, LinearPatchGeometry::new(&submesh));
                    }
                }
            }
        }

        let geom = Self { patches, edges };

        let max_angle = geom.max_normal_angle(mesh);
        if max_angle > 45.0 {
            warn!(
                "Max normal angle between the mesh boundary and the geometry is {max_angle} degrees"
            );
        }
        Ok(geom)
    }

    pub fn compute_curvature(&mut self) {
        for (&i, patch) in &mut self.patches {
            debug!("Compute curvature for patch {i}");
            patch.compute_curvature();
        }
    }

    pub fn curvature(&self, pt: &Vertex<D>, tag: Tag) -> Result<(Vertex<D>, Option<Vertex<D>>)> {
        self.patches.get(&tag).unwrap().curvature(pt)
    }

    pub fn write_curvature(&self, fname: &str) -> Result<()> {
        for (tag, patch) in &self.patches {
            patch.write_curvature(&String::from(fname).replace(".vtu", &format!("_{tag}.vtu")))?;
        }

        Ok(())
    }
}

impl<const D: usize, C: Simplex> Geometry<D> for LinearGeometry<D, C>
where
    GenericMesh<D, C>: HasCurvature<D>,
{
    fn check(&self, _topo: &Topology) -> Result<()> {
        // The check is performed during creation
        Ok(())
    }

    fn project(&self, pt: &mut Vertex<D>, tag: &TopoTag) -> f64 {
        assert!(tag.0 < D as Dim);

        let (dist, p) = if tag.1 < 0 {
            (0.0, *pt)
        } else if tag.0 == C::DIM as Dim {
            let patch = self.patches.get(&tag.1).unwrap_or_else(|| {
                panic!(
                    "Invalid face tag {tag:?}. Available tags are {:?}",
                    self.patches.keys().collect::<Vec<_>>()
                )
            });
            patch.project(pt)
        } else if tag.0 == 0 {
            (0.0, *pt)
        } else if tag.0 == C::DIM as Dim - 1 {
            // after 0 to make sure that if is used only for E=Triangle
            let edge = self.edges.get(&tag.1).unwrap_or_else(|| {
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
        let n_ref = patch.mesh.gelem(&patch.mesh.elem(idx)).normal().normalize();
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

    use super::{Geometry, LinearGeometry};
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

        let mesh = geom.clone();
        let topo = MeshTopology::new(&mesh);

        let geom = LinearGeometry::new(&mesh, &topo, geom)?;
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
        let topo = MeshTopology::new(&mesh);

        let (bdy, _) = mesh.boundary::<BoundaryMesh2d>();
        let geom = LinearGeometry::new(&mesh, &topo, bdy)?;

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

        let (bdy, _) = mesh.boundary::<BoundaryMesh3d>();
        let geom = LinearGeometry::new(&mesh, &topo, bdy)?;

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
