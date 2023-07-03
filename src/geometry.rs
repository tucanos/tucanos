use crate::{
    curvature::HasCurvature,
    geom_elems::GElem,
    mesh::{Point, SimplexMesh},
    topo_elems::Elem,
    topology::Topology,
    Dim, Error, Mesh, Result, Tag, TopoTag,
};
use log::info;
use rustc_hash::{FxHashMap, FxHashSet};
use std::{collections::HashMap, f64::consts::PI, hash::BuildHasherDefault};

/// Representation of a D-dimensional geometry
pub trait Geometry<const D: usize> {
    /// Check that the geometry is consistent with a Topology
    fn check(&self, topo: &Topology) -> Result<()>;

    /// Project a vertex, associated with a given `TopoTag`, onto the geometry
    /// The distance between the original vertex and its projection if returned
    fn project(&self, pt: &mut Point<D>, tag: &TopoTag) -> f64;

    /// Compute the angle between a vector n and the normal at the projection of pt onto the geometry
    fn angle(&self, pt: &mut Point<D>, n: &Point<D>, tag: &TopoTag) -> f64;

    /// Compute the max distance between the face centers and the geometry normals
    fn max_distance<E2: Elem>(&self, mesh: &SimplexMesh<D, E2>) -> f64 {
        let mut d_max = 0.0;
        for i_face in 0..mesh.n_faces() {
            let mut c = mesh.face_center(i_face);
            let tag = mesh.ftags[i_face as usize];
            let d = self.project(&mut c, &(E2::Face::DIM as Dim, tag));
            d_max = f64::max(d_max, d);
        }
        d_max
    }

    /// Compute the max angle between the face normals and the geometry normals
    fn max_normal_angle<E2: Elem>(&self, mesh: &SimplexMesh<D, E2>) -> f64 {
        let mut a_max = 0.0;
        for i_face in 0..mesh.n_faces() {
            let mut c = mesh.face_center(i_face);
            let tag = mesh.ftags[i_face as usize];
            let n = mesh.gface(i_face).normal();
            let a = self.angle(&mut c, &n, &(E2::Face::DIM as Dim, tag));
            a_max = f64::max(a_max, a);
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

    fn angle(&self, _pt: &mut Point<D>, _n: &Point<D>, tag: &TopoTag) -> f64 {
        assert_eq!(tag.0, D as Dim - 1);
        0.
    }
}

/// Geometry for a patch of faces with a constant tag
struct LinearPatchGeometry<const D: usize, E: Elem> {
    /// The face mesh
    mesh: SimplexMesh<D, E>,
}

impl<const D: usize, E: Elem> LinearPatchGeometry<D, E> {
    /// Create a `LinearPatchGeometry` from a `SimplexMesh`
    pub fn new(mut mesh: SimplexMesh<D, E>) -> Self {
        mesh.compute_octree();

        Self { mesh }
    }

    // Perform projection
    fn project(&self, pt: &mut Point<D>) -> (f64, Point<D>) {
        self.mesh.tree.as_ref().unwrap().project(pt)
    }
}

/// Geometry for a patch of faces with a constant tag, with curvature information
struct LinearPatchGeometryWithCurvature<const D: usize, E: Elem> {
    /// The face mesh
    mesh: SimplexMesh<D, E>,
    /// Optionally, the first principal curvature direction
    u: Option<Vec<Point<D>>>,
    /// Optionally, the second principal curvature direction (3D only)
    v: Option<Vec<Point<D>>>,
}

impl<const D: usize, E: Elem> LinearPatchGeometryWithCurvature<D, E>
where
    SimplexMesh<D, E>: HasCurvature<D>,
{
    /// Create a `LinearPatchGeometry` from a `SimplexMesh`
    pub fn new(mut mesh: SimplexMesh<D, E>) -> Self {
        mesh.compute_octree();
        mesh.compute_face_to_elems();
        mesh.add_boundary_faces();
        mesh.clear_face_to_elems();

        Self {
            mesh,
            u: None,
            v: None,
        }
    }

    // Perform projection
    fn project(&self, pt: &mut Point<D>) -> (f64, Point<D>) {
        self.mesh.tree.as_ref().unwrap().project(pt)
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

        let tree = self.mesh.tree.as_ref().unwrap();
        let i_elem = tree.nearest(pt) as usize;
        let u = self.u.as_ref().unwrap();
        if let Some(v) = self.v.as_ref() {
            Ok((u[i_elem], Some(v[i_elem])))
        } else {
            Ok((u[i_elem], None))
        }
    }

    pub fn write_curvature(&self, fname: &str) -> Result<()> {
        // Export the curvature
        let u1 = self
            .u
            .as_ref()
            .unwrap()
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        let v1 = self
            .v
            .as_ref()
            .unwrap()
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        let mut data = HashMap::new();
        data.insert(String::from("u"), u1.as_slice());
        data.insert(String::from("v"), v1.as_slice());

        self.mesh.write_vtk(fname, None, Some(data))?;

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
    edges: FxHashMap<Tag, LinearPatchGeometry<D, E::Face>>,
}

impl<const D: usize, E: Elem> LinearGeometry<D, E>
where
    SimplexMesh<D, E>: HasCurvature<D>,
{
    /// Create a `LinearGeometry` from a `SimplexMesh`
    pub fn new<E2: Elem>(mesh: &SimplexMesh<D, E2>, mut bdy: SimplexMesh<D, E>) -> Result<Self> {
        assert!(E2::DIM >= E::DIM);
        if mesh.topo.is_none() {
            return Err(Error::from("Mesh topology not computed"));
        }

        let mesh_topo = mesh.topo.as_ref().unwrap();

        // Faces
        let mesh_face_tags = mesh_topo.tags(E::DIM as Dim);
        let face_tags: FxHashSet<Tag> = bdy.etags().collect();
        if face_tags.len() != mesh_face_tags.len() {
            return Err(Error::from(&format!(
                "LinearGeometry: invalid # of tags (mesh: {mesh_face_tags:?}, bdy: {face_tags:?})"
            )));
        }

        let mut patches = FxHashMap::with_hasher(BuildHasherDefault::default());
        for tag in face_tags.iter().copied() {
            info!("Create LinearPatchGeometryWithCurvature for patch {tag}");
            if mesh_topo.get((E::DIM as Dim, tag)).is_none() {
                return Err(Error::from(&format!("LinearGeometry: face tag {tag:?} not found in topo (mesh: {mesh_face_tags:?}, bdy: {face_tags:?})")));
            }
            let (submesh, _, _, _) = bdy.extract(tag);
            patches.insert(tag, LinearPatchGeometryWithCurvature::new(submesh));
        }

        // Edges
        let mut edges = FxHashMap::with_hasher(BuildHasherDefault::default());
        if E::DIM == 2 {
            bdy.add_boundary_faces();
            let (bdy_topo, _) = Topology::from_mesh(&bdy);
            let (bdy_edges, _) = bdy.boundary();

            let edge_tags: FxHashSet<Tag> = bdy_edges.etags().collect();

            for tag in edge_tags {
                info!("Create LinearPatchGeometry for edge {tag}");
                // find the edge tag in mesh_topo
                let bdy_topo_node = bdy_topo.get((E::Face::DIM as Dim, tag)).unwrap();
                let bdy_parents = &bdy_topo_node.parents;
                let mesh_topo_node = mesh_topo
                    .get_from_parents_iter(E::Face::DIM as Dim, bdy_parents.iter().copied())
                    .unwrap();

                let (submesh, _, _, _) = bdy_edges.extract(tag);
                edges.insert(mesh_topo_node.tag.1, LinearPatchGeometry::new(submesh));
            }
        }
        Ok(Self { patches, edges })
    }

    pub fn compute_curvature(&mut self) {
        for (&i, patch) in self.patches.iter_mut() {
            info!("Compute curvature for patch {i}");
            patch.compute_curvature();
        }
    }

    pub fn curvature(&self, pt: &Point<D>, tag: Tag) -> Result<(Point<D>, Option<Point<D>>)> {
        self.patches.get(&tag).unwrap().curvature(pt)
    }

    pub fn write_curvature(&self, fname: &str) -> Result<()> {
        for (tag, patch) in self.patches.iter() {
            patch
                .write_curvature(&String::from(fname).replace(".vtu", &format!("_{}.vtu", tag)))?;
        }

        Ok(())
    }
}

impl<const D: usize, E: Elem> Geometry<D> for LinearGeometry<D, E>
where
    SimplexMesh<D, E>: HasCurvature<D>,
{
    fn check(&self, _topo: &Topology) -> Result<()> {
        // The check is performed during creation
        Ok(())
    }

    fn project(&self, pt: &mut Point<D>, tag: &TopoTag) -> f64 {
        assert!(tag.0 < D as Dim);

        let (dist, p) = if tag.0 == E::DIM as Dim {
            let patch = self.patches.get(&tag.1).unwrap();
            patch.project(pt)
        } else if tag.0 == 0 {
            (0.0, *pt)
        } else if tag.0 == E::DIM as Dim - 1 {
            // after 0 to make sure that if is used only for E=Triangle
            let edge = self.edges.get(&tag.1).unwrap();
            edge.project(pt)
        } else {
            unreachable!("{:?}", tag)
        };

        *pt = p;
        dist
    }

    fn angle(&self, pt: &mut Point<D>, n: &Point<D>, tag: &TopoTag) -> f64 {
        assert_eq!(tag.0, D as Dim - 1);

        let patch = self.patches.get(&tag.1).unwrap();
        let tree = patch.mesh.tree.as_ref().unwrap();
        let idx = tree.nearest(pt);
        let n_ref = patch.mesh.gelem(idx).normal();
        f64::acos(n.dot(&n_ref)) * 180. / PI
    }
}

#[cfg(test)]
mod tests {
    use super::{Geometry, LinearGeometry};
    use crate::{
        mesh::Point,
        mesh_stl::read_stl,
        test_meshes::write_stl_file,
        test_meshes::{test_mesh_2d, test_mesh_3d},
        Result,
    };
    use std::fs::remove_file;

    #[test]
    fn test_stl() -> Result<()> {
        write_stl_file("cube2.stl")?;
        let geom = read_stl("cube2.stl");
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
            .topo
            .as_ref()
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
