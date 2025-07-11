mod curvature;
mod orient;

use crate::{
    geometry::curvature::HasCurvature,
    mesh::{Elem, GElem, HasTmeshImpl, Point, SimplexMesh, Topology},
    Dim, Error, Idx, Result, Tag, TopoTag,
};
use log::{debug, warn};
pub use orient::orient_geometry;
use rustc_hash::{FxHashMap, FxHashSet};
use tmesh::spatialindex::ObjectIndex;

/// Representation of a D-dimensional geometry
pub trait Geometry<const D: usize>: Send + Sync {
    /// Check that the geometry is consistent with a Topology
    fn check(&self, topo: &Topology) -> Result<()>;

    /// Project a vertex, associated with a given `TopoTag`, onto the geometry
    /// The distance between the original vertex and its projection if returned
    fn project(&self, pt: &mut Point<D>, tag: &TopoTag) -> f64;

    /// Compute the angle between a vector n and the normal at the projection of pt onto the geometry
    fn angle(&self, pt: &Point<D>, n: &Point<D>, tag: &TopoTag) -> f64;

    /// Compute the max distance between the face centers and the geometry normals
    fn project_vertices<E: Elem>(&self, mesh: &mut SimplexMesh<D, E>) -> f64 {
        let vtags = mesh.get_vertex_tags().unwrap().to_vec();

        let mut d_max = 0.0;
        for (p, tag) in mesh.mut_verts().zip(vtags.iter()) {
            if tag.0 < E::DIM as Dim {
                let d = self.project(p, tag);
                d_max = f64::max(d_max, d);
            }
        }

        d_max
    }

    /// Compute the max distance between the face centers and the geometry normals
    fn max_distance<E2: Elem>(&self, mesh: &SimplexMesh<D, E2>) -> f64 {
        let mut d_max = 0.0;
        for (gf, tag) in mesh.gfaces().zip(mesh.ftags()) {
            let mut c = gf.center();
            let d = self.project(&mut c, &(E2::Face::DIM as Dim, tag));
            d_max = f64::max(d_max, d);
        }
        d_max
    }

    /// Compute the max angle between the face normals and the geometry normals
    fn max_normal_angle<E2: Elem>(&self, mesh: &SimplexMesh<D, E2>) -> f64 {
        let mut a_max = 0.0;
        for (gf, tag) in mesh.gfaces().zip(mesh.ftags()) {
            if tag > 0 {
                let c = gf.center();
                let n = gf.normal();
                let a = self.angle(&c, &n, &(E2::Face::DIM as Dim, tag));
                a_max = f64::max(a_max, a);
            }
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

    fn project(&self, _pt: &mut Point<D>, tag: &TopoTag) -> f64 {
        assert!(tag.0 < D as Dim);
        0.
    }

    fn angle(&self, _pt: &Point<D>, _n: &Point<D>, tag: &TopoTag) -> f64 {
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
    pub fn new<E: Elem>(mesh: &SimplexMesh<D, E>) -> Self
    where
        SimplexMesh<D, E>: HasTmeshImpl<D, E>,
    {
        let tree = mesh.elem_tree();

        Self { tree }
    }

    // Perform projection
    fn project(&self, pt: &Point<D>) -> (f64, Point<D>) {
        self.tree.project(pt)
    }
}

/// Geometry for a patch of faces with a constant tag, with curvature information
struct LinearPatchGeometryWithCurvature<const D: usize, E: Elem> {
    /// The face mesh
    mesh: SimplexMesh<D, E>,
    /// The ObjectIndex
    tree: ObjectIndex<D>,
    /// Optionally, the first principal curvature direction
    u: Option<Vec<Point<D>>>,
    /// Optionally, the second principal curvature direction (3D only)
    v: Option<Vec<Point<D>>>,
}

impl<const D: usize, E: Elem> LinearPatchGeometryWithCurvature<D, E>
where
    SimplexMesh<D, E>: HasCurvature<D> + HasTmeshImpl<D, E>,
{
    /// Create a `LinearPatchGeometry` from a `SimplexMesh`
    pub fn new(mut mesh: SimplexMesh<D, E>) -> Self {
        let tree = mesh.elem_tree();
        mesh.compute_face_to_elems();
        mesh.add_boundary_faces();
        mesh.clear_face_to_elems();

        Self {
            mesh,
            tree,
            u: None,
            v: None,
        }
    }

    // Perform projection
    fn project(&self, pt: &Point<D>) -> (f64, Point<D>) {
        self.tree.project(pt)
    }

    fn compute_curvature(&mut self) {
        self.mesh.add_boundary_faces();
        self.mesh.compute_elem_to_elems();

        let (u, v) = self.mesh.compute_curvature();

        self.mesh.clear_elem_to_elems();
        self.u = Some(u);
        self.v = v;
    }

    fn curvature(&self, pt: &Point<D>) -> Result<(Point<D>, Option<Point<D>>)> {
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
        let mut writer = self.mesh.vtu_writer();
        writer.add_point_data("u", D, self.u.as_ref().unwrap().iter().flatten().copied());
        writer.add_point_data("v", D, self.v.as_ref().unwrap().iter().flatten().copied());
        writer.export(fname)?;

        Ok(())
    }
}

/// Piecewise linear (stl-like) representation of a geometry
/// doc TODO
pub struct LinearGeometry<const D: usize, E: Elem>
where
    SimplexMesh<D, E>: HasCurvature<D>,
{
    /// The surface patches
    patches: FxHashMap<Tag, LinearPatchGeometryWithCurvature<D, E>>,
    /// The edges
    edges: FxHashMap<Tag, LinearPatchGeometry<D>>,
}

impl<const D: usize, E: Elem> LinearGeometry<D, E>
where
    SimplexMesh<D, E>: HasCurvature<D> + HasTmeshImpl<D, E>,
    SimplexMesh<D, E::Face>: HasTmeshImpl<D, E::Face>,
{
    /// Create a `LinearGeometry` for the boundary of `mesh` (with positive tags) from a
    /// `SimplexMesh` representation of the boundary
    pub fn new<E2: Elem>(mesh: &SimplexMesh<D, E2>, mut bdy: SimplexMesh<D, E>) -> Result<Self> {
        assert!(E2::DIM >= E::DIM);

        let mesh_topo = mesh.get_topology()?;

        // Faces
        let mesh_face_tags = mesh_topo.tags(E::DIM as Dim);
        let face_tags: FxHashSet<Tag> = if E2::DIM == E::DIM + 1 {
            mesh.ftags().filter(|&t| t > 0).collect()
        } else if E2::DIM == E::DIM {
            mesh.etags().filter(|&t| t > 0).collect()
        } else {
            unimplemented!("mesh dimension {}, bdy dimension {}", E::DIM, E2::DIM);
        };

        let mut patches = FxHashMap::default();
        for tag in face_tags.iter().copied() {
            debug!("Create LinearPatchGeometryWithCurvature for patch {tag}");
            if mesh_topo.get((E::DIM as Dim, tag)).is_none() {
                return Err(Error::from(&format!(
                    "LinearGeometry: face tag {tag:?} not found in topo (mesh: {mesh_face_tags:?}, bdy: {face_tags:?})"
                )));
            }
            let submesh = bdy.extract_tag(tag).mesh;
            assert_ne!(submesh.n_verts(), 0, "Geometry mesh empty for tag {tag}");
            patches.insert(tag, LinearPatchGeometryWithCurvature::new(submesh));
        }

        // Edges
        let mut edges = FxHashMap::default();
        if E::DIM == 2 {
            bdy.add_boundary_faces();
            let bdy_topo = bdy.compute_topology().clone();
            let (bdy_edges, _) = bdy.boundary();

            let edge_tags: FxHashSet<Tag> = bdy_edges.etags().filter(|&t| t > 0).collect();

            for tag in edge_tags {
                debug!("Create LinearPatchGeometry for edge {tag}");
                // find the edge tag in mesh_topo
                let bdy_topo_node = bdy_topo.get((E::Face::DIM as Dim, tag)).unwrap();
                let bdy_parents = &bdy_topo_node.parents;
                let mesh_topo_node = mesh_topo
                    .get_from_parents_iter(E::Face::DIM as Dim, bdy_parents.iter().copied());

                if let Some(mesh_topo_node) = mesh_topo_node {
                    let submesh = bdy_edges.extract_tag(tag).mesh;
                    edges.insert(mesh_topo_node.tag.1, LinearPatchGeometry::new(&submesh));
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

    pub fn curvature(&self, pt: &Point<D>, tag: Tag) -> Result<(Point<D>, Option<Point<D>>)> {
        self.patches.get(&tag).unwrap().curvature(pt)
    }

    pub fn write_curvature(&self, fname: &str) -> Result<()> {
        for (tag, patch) in &self.patches {
            patch.write_curvature(&String::from(fname).replace(".vtu", &format!("_{tag}.vtu")))?;
        }

        Ok(())
    }
}

impl<const D: usize, E: Elem> Geometry<D> for LinearGeometry<D, E>
where
    SimplexMesh<D, E>: HasCurvature<D> + HasTmeshImpl<D, E>,
{
    fn check(&self, _topo: &Topology) -> Result<()> {
        // The check is performed during creation
        Ok(())
    }

    fn project(&self, pt: &mut Point<D>, tag: &TopoTag) -> f64 {
        assert!(tag.0 < D as Dim);

        let (dist, p) = if tag.1 < 0 {
            (0.0, *pt)
        } else if tag.0 == E::DIM as Dim {
            let patch = self
                .patches
                .get(&tag.1)
                .unwrap_or_else(|| panic!("Invalid face tag {tag:?}"));
            patch.project(pt)
        } else if tag.0 == 0 {
            (0.0, *pt)
        } else if tag.0 == E::DIM as Dim - 1 {
            // after 0 to make sure that if is used only for E=Triangle
            let edge = self
                .edges
                .get(&tag.1)
                .unwrap_or_else(|| panic!("Invalid edge tag {tag:?}"));
            edge.project(pt)
        } else {
            unreachable!("{:?}", tag)
        };

        *pt = p;
        dist
    }

    fn angle(&self, pt: &Point<D>, n: &Point<D>, tag: &TopoTag) -> f64 {
        assert_eq!(tag.0, D as Dim - 1);

        let patch = self.patches.get(&tag.1).unwrap();
        let idx = patch.tree.nearest_elem(pt);
        let n_ref = patch.mesh.gelem(patch.mesh.elem(idx as Idx)).normal();
        let cos_a = n.dot(&n_ref).clamp(-1.0, 1.0);
        f64::acos(cos_a).to_degrees()
    }
}

#[cfg(test)]
mod tests {
    use tmesh::mesh::{read_stl, Mesh};

    use super::{Geometry, LinearGeometry};
    use crate::{
        mesh::{
            test_meshes::{test_mesh_2d, test_mesh_3d, write_stl_file},
            Point, SimplexMesh, Triangle,
        },
        Result,
    };
    use std::fs::remove_file;

    #[test]
    fn test_stl() -> Result<()> {
        write_stl_file("cube2.stl")?;
        let geom: SimplexMesh<3, Triangle> = read_stl("cube2.stl")?;
        remove_file("cube2.stl")?;

        let mut mesh = geom.clone();
        mesh.compute_topology();

        let geom = LinearGeometry::new(&mesh, geom)?;
        let mut p = Point::<3>::new(2., 0.5, 0.5);
        let d = geom.project(&mut p, &(2, 1));
        assert!(f64::abs(d - 1.) < 1e-12);
        assert!((p - Point::<3>::new(1., 0.5, 0.5)).norm() < 1e-12);

        let mut p = Point::<3>::new(0.5, 0.75, 0.5);
        let d = geom.project(&mut p, &(2, 1));
        assert!(f64::abs(d - 0.25) < 1e-12);
        assert!((p - Point::<3>::new(0.5, 1., 0.5)).norm() < 1e-12);

        Ok(())
    }

    #[test]
    fn test_linear_geometry_2d() -> Result<()> {
        let mut mesh = test_mesh_2d().split().split();
        mesh.add_boundary_faces();
        mesh.compute_topology();

        let (bdy, _) = mesh.boundary();
        let geom = LinearGeometry::new(&mesh, bdy)?;

        let mut pt = Point::<2>::new(0.75, 0.5);
        let d = geom.project(&mut pt, &(1, 1));
        assert!(f64::abs(d - 0.5) < 1e-12);

        let mut pt = Point::<2>::new(0.75, 0.5);
        let d = geom.project(&mut pt, &(1, 2));
        assert!(f64::abs(d - 0.25) < 1e-12);

        let mut pt = Point::<2>::new(0.75, 0.5);
        let d = geom.project(&mut pt, &(1, 3));
        assert!(f64::abs(d - 0.5) < 1e-12);

        let mut pt = Point::<2>::new(0.75, 0.5);
        let d = geom.project(&mut pt, &(1, 4));
        assert!(f64::abs(d - 0.75) < 1e-12);

        Ok(())
    }

    #[test]
    fn test_linear_geometry_3d() -> Result<()> {
        let mut mesh = test_mesh_3d().split().split();
        mesh.add_boundary_faces();
        mesh.compute_topology();

        let (bdy, _) = mesh.boundary();
        let geom = LinearGeometry::new(&mesh, bdy)?;

        let mut pt = Point::<3>::new(0.75, 0.5, 0.25);
        let _ = geom.project(&mut pt, &(2, 1));
        assert!(f64::abs(pt[0] - 0.75) < 1e-12);
        assert!(f64::abs(pt[1] - 0.5) < 1e-12);
        assert!(f64::abs(pt[2] - 0.0) < 1e-12);

        let mut pt = Point::<3>::new(0.75, 0.5, 0.25);
        let _ = geom.project(&mut pt, &(2, 2));
        assert!(f64::abs(pt[0] - 0.75) < 1e-12);
        assert!(f64::abs(pt[1] - 0.5) < 1e-12);
        assert!(f64::abs(pt[2] - 1.0) < 1e-12);

        let mut pt = Point::<3>::new(0.75, 0.5, 0.25);
        let _ = geom.project(&mut pt, &(2, 3));
        assert!(f64::abs(pt[0] - 0.75) < 1e-12);
        assert!(f64::abs(pt[1] - 0.0) < 1e-12);
        assert!(f64::abs(pt[2] - 0.25) < 1e-12);

        let mut pt = Point::<3>::new(0.75, 0.5, 0.25);
        let _ = geom.project(&mut pt, &(2, 4));
        assert!(f64::abs(pt[0] - 0.75) < 1e-12);
        assert!(f64::abs(pt[1] - 1.0) < 1e-12);
        assert!(f64::abs(pt[2] - 0.25) < 1e-12);

        let mut pt = Point::<3>::new(0.75, 0.5, 0.25);
        let _ = geom.project(&mut pt, &(2, 5));
        assert!(f64::abs(pt[0] - 1.0) < 1e-12);
        assert!(f64::abs(pt[1] - 0.5) < 1e-12);
        assert!(f64::abs(pt[2] - 0.25) < 1e-12);

        let mut pt = Point::<3>::new(0.75, 0.5, 0.25);
        let _ = geom.project(&mut pt, &(2, 6));
        assert!(f64::abs(pt[0] - 0.0) < 1e-12);
        assert!(f64::abs(pt[1] - 0.5) < 1e-12);
        assert!(f64::abs(pt[2] - 0.25) < 1e-12);

        let topo_node = mesh
            .get_topology()
            .unwrap()
            .get_from_parents(1, &[6, 3])
            .unwrap();
        let mut pt = Point::<3>::new(0.75, 0.5, 0.25);
        let _ = geom.project(&mut pt, &(1, topo_node.tag.1));
        assert!(f64::abs(pt[0] - 0.0) < 1e-12);
        assert!(f64::abs(pt[1] - 0.0) < 1e-12);
        assert!(f64::abs(pt[2] - 0.25) < 1e-12);

        Ok(())
    }
}
